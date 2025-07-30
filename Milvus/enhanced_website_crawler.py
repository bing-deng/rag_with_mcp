#!/usr/bin/env python3
"""
增强版网站爬虫
集成智能分块处理，专门优化为関電工网站爬取
"""

import os
import re
import time
import hashlib
import requests
import threading
from queue import Queue, Empty
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

# HTML 解析
from bs4 import BeautifulSoup

# 并发处理
from concurrent.futures import ThreadPoolExecutor, as_completed

# 增强版处理器
from enhanced_html_to_milvus import EnhancedHTMLToMilvusProcessor

@dataclass
class EnhancedCrawlConfig:
    """增强版爬虫配置"""
    base_url: str
    max_pages: int = 500
    max_depth: int = 3
    concurrent_workers: int = 3
    delay_between_requests: float = 2.0
    timeout: int = 30
    collection_name: str = "kandenko_website_smart"
    quality_threshold: float = 0.4

class EnhancedWebsiteCrawler:
    """增强版网站爬虫，集成智能分块"""
    
    def __init__(self, config: EnhancedCrawlConfig):
        self.config = config
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.url_queue = Queue()
        self.results = []
        self.lock = threading.Lock()
        
        # 解析基础域名
        parsed = urlparse(config.base_url)
        self.base_domain = parsed.netloc
        self.base_scheme = parsed.scheme
        
        # 初始化增强版处理器
        self.processor = EnhancedHTMLToMilvusProcessor(
            collection_name=config.collection_name
        )
        
        # 统计信息
        self.stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'smart_chunks_created': 0,
            'high_quality_chunks': 0,
            'start_time': None,
            'end_time': None,
            'languages_detected': {'ja': 0, 'zh': 0, 'en': 0}
        }
        
        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8,zh;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc == self.base_domain and
                not any(exclude in url.lower() for exclude in [
                    'javascript:', 'mailto:', 'tel:', '.pdf', '.doc', '.zip',
                    'logout', 'login', 'admin', 'api/', '/search?'
                ])
            )
        except:
            return False
    
    def _fetch_page(self, url: str) -> Optional[Tuple[str, str]]:
        """获取页面内容"""
        try:
            print(f"🌐 正在获取: {url}")
            
            response = self.session.get(
                url, 
                timeout=self.config.timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # 尝试检测编码
                encoding = response.encoding
                if encoding.lower() in ['iso-8859-1', 'ascii']:
                    # 检测日文编码
                    import chardet
                    detected = chardet.detect(response.content)
                    if detected['encoding']:
                        encoding = detected['encoding']
                
                response.encoding = encoding
                content = response.text
                
                # 提取标题
                soup = BeautifulSoup(content, 'html.parser')
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else ""
                
                print(f"✅ 成功获取: {url} (编码: {encoding})")
                return content, title
            else:
                print(f"❌ HTTP错误 {response.status_code}: {url}")
                return None
                
        except Exception as e:
            print(f"❌ 获取失败 {url}: {e}")
            return None
    
    def _extract_links(self, html_content: str, base_url: str) -> Set[str]:
        """提取页面中的链接"""
        links = set()
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if href:
                    absolute_url = urljoin(base_url, href)
                    # 移除片段标识符
                    absolute_url = absolute_url.split('#')[0]
                    
                    if self._is_valid_url(absolute_url):
                        links.add(absolute_url)
            
            print(f"🔗 发现 {len(links)} 个有效链接")
            return links
            
        except Exception as e:
            print(f"❌ 提取链接失败: {e}")
            return set()
    
    def _crawl_page(self, url: str, depth: int) -> bool:
        """爬取单个页面"""
        if depth > self.config.max_depth:
            return False
        
        with self.lock:
            if url in self.visited_urls or len(self.visited_urls) >= self.config.max_pages:
                return False
            self.visited_urls.add(url)
        
        # 获取页面内容
        result = self._fetch_page(url)
        if not result:
            with self.lock:
                self.failed_urls.add(url)
                self.stats['pages_failed'] += 1
            return False
        
        html_content, title = result
        
        # 使用增强版处理器进行智能分块
        try:
            items = self.processor.process_html_page(url, html_content, title)
            
            if items:
                # 批量保存
                if self.processor.save_items(items):
                    with self.lock:
                        self.stats['pages_crawled'] += 1
                        self.stats['smart_chunks_created'] += len(items)
                        
                        # 统计高质量块
                        high_quality = sum(1 for item in items if item['quality_score'] >= self.config.quality_threshold)
                        self.stats['high_quality_chunks'] += high_quality
                        
                        # 语言统计
                        for item in items:
                            lang = item['language']
                            if lang in self.stats['languages_detected']:
                                self.stats['languages_detected'][lang] += 1
                
                print(f"✅ 页面处理完成: {url} ({len(items)} 个智能块)")
            else:
                print(f"⚠️  页面无有效内容: {url}")
        
        except Exception as e:
            print(f"❌ 处理页面失败 {url}: {e}")
            with self.lock:
                self.failed_urls.add(url)
                self.stats['pages_failed'] += 1
            return False
        
        # 提取新链接（仅在浅层时）
        if depth < self.config.max_depth - 1:
            new_links = self._extract_links(html_content, url)
            
            for link in new_links:
                if link not in self.visited_urls and link not in self.failed_urls:
                    self.url_queue.put((link, depth + 1))
        
        # 请求间延时
        time.sleep(self.config.delay_between_requests)
        return True
    
    def crawl(self) -> Dict:
        """执行智能爬取"""
        print(f"🚀 开始智能爬取: {self.config.base_url}")
        print(f"📊 配置: 最大页面={self.config.max_pages}, 最大深度={self.config.max_depth}")
        
        self.stats['start_time'] = time.time()
        
        # 连接到Milvus
        if not self.processor.connect():
            raise Exception("无法连接到Milvus")
        
        # 创建集合
        if not self.processor.create_collection():
            raise Exception("无法创建Milvus集合")
        
        # 添加起始URL
        self.url_queue.put((self.config.base_url, 0))
        
        # 多线程爬取
        with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
            futures = []
            
            while not self.url_queue.empty() or futures:
                # 提交新任务
                while len(futures) < self.config.concurrent_workers and not self.url_queue.empty():
                    try:
                        url, depth = self.url_queue.get_nowait()
                        future = executor.submit(self._crawl_page, url, depth)
                        futures.append(future)
                    except Empty:
                        break
                
                # 检查完成的任务
                if futures:
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            completed_futures.append(future)
                    
                    for future in completed_futures:
                        futures.remove(future)
                        try:
                            future.result()
                        except Exception as e:
                            print(f"❌ 任务执行失败: {e}")
                
                # 短暂等待
                time.sleep(0.1)
        
        self.stats['end_time'] = time.time()
        
        # 创建索引
        print("🔗 创建优化索引...")
        self.processor.create_index()
        
        # 完成统计
        duration = self.stats['end_time'] - self.stats['start_time']
        success_rate = self.stats['pages_crawled'] / (self.stats['pages_crawled'] + self.stats['pages_failed']) if (self.stats['pages_crawled'] + self.stats['pages_failed']) > 0 else 0
        
        report = {
            'config': {
                'base_url': self.config.base_url,
                'collection_name': self.config.collection_name,
                'max_pages': self.config.max_pages
            },
            'stats': {
                'duration_seconds': duration,
                'pages_crawled': self.stats['pages_crawled'],
                'pages_failed': self.stats['pages_failed'],
                'smart_chunks_created': self.stats['smart_chunks_created'],
                'high_quality_chunks': self.stats['high_quality_chunks'],
                'success_rate': success_rate,
                'languages_detected': self.stats['languages_detected']
            },
            'processor_stats': self.processor.get_stats()
        }
        
        # 打印统计
        self.print_final_stats()
        self.processor.print_stats()
        
        return report
    
    def print_final_stats(self):
        """打印最终统计信息"""
        print("\n🎯 智能爬取最终统计:")
        print("=" * 60)
        duration = self.stats['end_time'] - self.stats['start_time']
        print(f"⏱️  总耗时: {duration:.1f} 秒")
        print(f"📄 成功页面: {self.stats['pages_crawled']}")
        print(f"❌ 失败页面: {self.stats['pages_failed']}")
        print(f"🧩 智能块总数: {self.stats['smart_chunks_created']}")
        print(f"⭐ 高质量块: {self.stats['high_quality_chunks']}")
        
        if self.stats['smart_chunks_created'] > 0:
            quality_rate = self.stats['high_quality_chunks'] / self.stats['smart_chunks_created']
            print(f"📈 质量率: {quality_rate:.1%}")
        
        print(f"🌐 语言分布:")
        for lang, count in self.stats['languages_detected'].items():
            if count > 0:
                lang_name = {'ja': '日语', 'zh': '中文', 'en': '英语'}.get(lang, lang)
                print(f"   {lang_name}: {count} 块")
        
        success_rate = self.stats['pages_crawled'] / (self.stats['pages_crawled'] + self.stats['pages_failed']) if (self.stats['pages_crawled'] + self.stats['pages_failed']) > 0 else 0
        print(f"🎯 成功率: {success_rate:.1%}")
    
    def save_report(self) -> str:
        """保存爬取报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_crawl_report_{self.base_domain}_{timestamp}.json"
        
        report = {
            'config': self.config.__dict__,
            'stats': self.stats,
            'processor_stats': self.processor.get_stats(),
            'visited_urls': list(self.visited_urls),
            'failed_urls': list(self.failed_urls)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📋 报告已保存: {filename}")
        return filename
    
    def disconnect(self):
        """断开连接"""
        self.processor.disconnect()

def create_kandenko_smart_config() -> EnhancedCrawlConfig:
    """创建関電工智能爬取配置"""
    return EnhancedCrawlConfig(
        base_url="https://www.kandenko.co.jp",
        max_pages=300,
        max_depth=3,
        concurrent_workers=2,
        delay_between_requests=2.0,
        collection_name="kandenko_website_smart",
        quality_threshold=0.5
    )

if __name__ == "__main__":
    config = create_kandenko_smart_config()
    crawler = EnhancedWebsiteCrawler(config)
    
    try:
        report = crawler.crawl()
        crawler.save_report()
        print("🎉 智能爬取完成！")
    except Exception as e:
        print(f"❌ 爬取失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        crawler.disconnect()