#!/usr/bin/env python3
"""
完整网站爬虫系统
爬取整个网站的所有页面，解析内容并存储到 Milvus 向量数据库中
支持多线程、去重、错误处理等高级功能
"""

import os
import re
import time
import hashlib
import requests
import threading
from queue import Queue, Empty
from urllib.parse import urljoin, urlparse, urlencode
from urllib.robotparser import RobotFileParser
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# HTML 解析
from bs4 import BeautifulSoup

# 并发处理
from concurrent.futures import ThreadPoolExecutor, as_completed

# Milvus 和向量化
from html_to_milvus import HTMLToMilvusProcessor

@dataclass
class CrawlConfig:
    """爬虫配置"""
    base_url: str
    max_pages: int = 1000
    max_depth: int = 3
    concurrent_workers: int = 5
    delay_between_requests: float = 1.0
    timeout: int = 30
    respect_robots_txt: bool = True
    allowed_domains: List[str] = None
    exclude_patterns: List[str] = None
    include_patterns: List[str] = None
    collection_name: str = "website_content"

class WebsiteCrawler:
    """完整网站爬虫"""
    
    def __init__(self, config: CrawlConfig):
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
        
        # 允许的域名
        if config.allowed_domains is None:
            self.allowed_domains = {self.base_domain}
        else:
            self.allowed_domains = set(config.allowed_domains)
            self.allowed_domains.add(self.base_domain)
        
        # 初始化 robots.txt 检查
        self.rp = None
        if config.respect_robots_txt:
            self._load_robots_txt()
        
        # 初始化 Milvus 处理器
        self.milvus_processor = HTMLToMilvusProcessor(
            collection_name=config.collection_name
        )
        
        # 统计信息
        self.stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'content_blocks_extracted': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _load_robots_txt(self):
        """加载并解析 robots.txt"""
        try:
            robots_url = urljoin(self.config.base_url, '/robots.txt')
            self.rp = RobotFileParser()
            self.rp.set_url(robots_url)
            self.rp.read()
            print(f"✅ 已加载 robots.txt: {robots_url}")
        except Exception as e:
            print(f"⚠️  无法加载 robots.txt: {e}")
            self.rp = None
    
    def _can_fetch(self, url: str) -> bool:
        """检查是否可以爬取该URL"""
        if self.rp is None:
            return True
        return self.rp.can_fetch('*', url)
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        if not url or url in self.visited_urls:
            return False
        
        parsed = urlparse(url)
        
        # 检查域名
        if parsed.netloc not in self.allowed_domains:
            return False
        
        # 检查协议
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # 检查排除模式
        if self.config.exclude_patterns:
            for pattern in self.config.exclude_patterns:
                if re.search(pattern, url):
                    return False
        
        # 检查包含模式
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if re.search(pattern, url):
                    return True
            return False
        
        # 排除常见的非内容URL
        exclude_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.rar', '.exe']
        if any(url.lower().endswith(ext) for ext in exclude_extensions):
            return False
        
        return True
    
    def _extract_links(self, html_content: str, base_url: str) -> List[str]:
        """从HTML中提取链接"""
        links = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取所有链接
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href:
                    continue
                
                # 处理相对链接
                absolute_url = urljoin(base_url, href)
                
                # 移除锚点和查询参数（可选）
                parsed = urlparse(absolute_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                if self._is_valid_url(clean_url):
                    links.append(clean_url)
        
        except Exception as e:
            print(f"⚠️  提取链接失败: {e}")
        
        return list(set(links))  # 去重
    
    def _fetch_page(self, url: str) -> Optional[Dict]:
        """获取单个页面"""
        if not self._can_fetch(url):
            print(f"🚫 Robots.txt 禁止访问: {url}")
            return None
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ja,en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=self.config.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                print(f"⚠️  跳过非HTML内容: {url} ({content_type})")
                return None
            
            # 确保正确的编码处理，特别是日文内容
            if response.encoding is None or response.encoding.lower() in ['iso-8859-1', 'ascii']:
                response.encoding = response.apparent_encoding or 'utf-8'
            
            return {
                'url': url,
                'content': response.text,
                'status_code': response.status_code,
                'content_length': len(response.text),
                'timestamp': time.time()
            }
        
        except Exception as e:
            print(f"❌ 获取页面失败: {url} - {e}")
            with self.lock:
                self.failed_urls.add(url)
                self.stats['pages_failed'] += 1
            return None
    
    def _process_page(self, page_data: Dict, depth: int) -> List[str]:
        """处理单个页面，返回发现的新链接"""
        url = page_data['url']
        html_content = page_data['content']
        
        print(f"📄 处理页面 (深度 {depth}): {url}")
        
        try:
            # 解析HTML内容
            content_blocks = self.milvus_processor.parse_html(html_content, base_url=url)
            
            # 记录统计信息
            with self.lock:
                self.stats['pages_crawled'] += 1
                self.stats['content_blocks_extracted'] += len(content_blocks)
                self.visited_urls.add(url)
            
            # 提取链接
            new_links = []
            if depth < self.config.max_depth:
                links = self._extract_links(html_content, url)
                new_links = [link for link in links if link not in self.visited_urls]
            
            # 存储页面结果
            self.results.append({
                'url': url,
                'content_blocks': content_blocks,
                'links_found': len(new_links),
                'depth': depth,
                'timestamp': page_data['timestamp'],
                'content_length': page_data['content_length']
            })
            
            return new_links
        
        except Exception as e:
            print(f"❌ 处理页面失败: {url} - {e}")
            return []
    
    def _worker(self, depth: int):
        """工作线程函数"""
        while True:
            try:
                url = self.url_queue.get(timeout=10)
                
                # 检查是否已访问
                with self.lock:
                    if url in self.visited_urls:
                        self.url_queue.task_done()
                        continue
                
                # 延迟请求
                time.sleep(self.config.delay_between_requests)
                
                # 获取页面
                page_data = self._fetch_page(url)
                if page_data is None:
                    self.url_queue.task_done()
                    continue
                
                # 处理页面
                new_links = self._process_page(page_data, depth)
                
                # 添加新链接到队列
                if depth < self.config.max_depth:
                    for link in new_links:
                        with self.lock:
                            if (link not in self.visited_urls and 
                                len(self.visited_urls) < self.config.max_pages):
                                self.url_queue.put(link)
                
                self.url_queue.task_done()
                
            except Empty:
                break
            except Exception as e:
                print(f"❌ 工作线程错误: {e}")
                self.url_queue.task_done()
    
    def crawl(self) -> Dict:
        """开始爬取网站"""
        print(f"🚀 开始爬取网站: {self.config.base_url}")
        print(f"📊 配置: 最大页面数={self.config.max_pages}, 最大深度={self.config.max_depth}, 并发数={self.config.concurrent_workers}")
        
        self.stats['start_time'] = time.time()
        
        # 添加起始URL
        self.url_queue.put(self.config.base_url)
        
        # 启动工作线程
        threads = []
        for i in range(self.config.concurrent_workers):
            t = threading.Thread(target=self._worker, args=(0,))
            t.daemon = True
            t.start()
            threads.append(t)
        
        try:
            # 等待所有任务完成
            self.url_queue.join()
            
            # 处理更深层次的链接
            for depth in range(1, self.config.max_depth + 1):
                if len(self.visited_urls) >= self.config.max_pages:
                    break
                
                print(f"\n🔍 开始处理深度 {depth} 的页面...")
                
                # 启动新的工作线程处理这个深度
                depth_threads = []
                for i in range(self.config.concurrent_workers):
                    t = threading.Thread(target=self._worker, args=(depth,))
                    t.daemon = True
                    t.start()
                    depth_threads.append(t)
                
                # 等待这个深度的任务完成
                self.url_queue.join()
                
                # 等待线程结束
                for t in depth_threads:
                    t.join(timeout=1)
        
        except KeyboardInterrupt:
            print("\n⚠️  用户中断爬取过程")
        
        self.stats['end_time'] = time.time()
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """生成爬取报告"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        report = {
            'config': asdict(self.config),
            'stats': {
                **self.stats,
                'duration_seconds': duration,
                'pages_per_second': self.stats['pages_crawled'] / duration if duration > 0 else 0,
                'total_urls_discovered': len(self.visited_urls) + len(self.failed_urls),
                'success_rate': self.stats['pages_crawled'] / (self.stats['pages_crawled'] + self.stats['pages_failed']) if (self.stats['pages_crawled'] + self.stats['pages_failed']) > 0 else 0
            },
            'visited_urls': list(self.visited_urls),
            'failed_urls': list(self.failed_urls),
            'sample_results': self.results[:5]  # 只保存前5个结果作为示例
        }
        
        return report
    
    def save_to_milvus(self) -> bool:
        """将爬取的内容保存到 Milvus"""
        if not self.results:
            print("❌ 没有数据需要保存")
            return False
        
        print(f"💾 开始保存数据到 Milvus...")
        
        try:
            # 连接到 Milvus
            if not self.milvus_processor.connect_to_milvus():
                return False
            
            # 创建集合
            self.milvus_processor.create_collection()
            self.milvus_processor.create_index()
            self.milvus_processor.load_collection()
            
            # 收集所有内容块
            all_content_blocks = []
            for result in self.results:
                all_content_blocks.extend(result['content_blocks'])
            
            # 批量插入到 Milvus
            success = self.milvus_processor.insert_html_content(all_content_blocks)
            
            if success:
                print(f"✅ 成功保存 {len(all_content_blocks)} 个内容块到 Milvus")
                
                # 显示统计信息
                stats = self.milvus_processor.get_statistics()
                print(f"📊 Milvus 统计信息:")
                print(f"   集合名称: {stats.get('collection_name')}")
                print(f"   总内容块: {stats.get('total_blocks')}")
                
                if 'content_type_distribution' in stats:
                    print("   内容类型分布:")
                    for content_type, count in stats['content_type_distribution'].items():
                        print(f"     {content_type}: {count}")
            
            return success
            
        except Exception as e:
            print(f"❌ 保存到 Milvus 失败: {e}")
            return False
        
        finally:
            self.milvus_processor.disconnect()
    
    def save_report(self, filename: str = None) -> str:
        """保存爬取报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = urlparse(self.config.base_url).netloc.replace('.', '_')
            filename = f"crawl_report_{domain}_{timestamp}.json"
        
        report = self._generate_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"📋 爬取报告已保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
            return ""

def create_kandenko_config() -> CrawlConfig:
    """为 Kandenko 网站创建专用配置"""
    config = CrawlConfig(
        base_url="https://www.kandenko.co.jp/",
        max_pages=200,  # 限制页面数量，避免过度爬取
        max_depth=3,    # 爬取深度
        concurrent_workers=3,  # 并发数，不要太高以免给服务器造成压力
        delay_between_requests=2.0,  # 请求间隔2秒，更礼貌
        timeout=30,
        respect_robots_txt=True,
        collection_name="kandenko_website",
        # 只爬取同域名下的页面
        allowed_domains=["www.kandenko.co.jp"],
        # 排除一些不需要的页面类型
        exclude_patterns=[
            r'/print/',  # 打印页面
            r'/search\?',  # 搜索结果页
            r'\.pdf$',   # PDF文件
            r'/contact/.*send',  # 表单提交页面
            r'/admin/',  # 管理页面
        ]
    )
    return config

def main():
    """主函数"""
    print("🕷️  完整网站爬虫系统")
    print("=" * 60)
    
    # 创建 Kandenko 专用配置
    config = create_kandenko_config()
    
    print(f"🎯 目标网站: {config.base_url}")
    print(f"📊 爬取配置:")
    print(f"   最大页面数: {config.max_pages}")
    print(f"   最大深度: {config.max_depth}")
    print(f"   并发工作器: {config.concurrent_workers}")
    print(f"   请求间隔: {config.delay_between_requests}秒")
    
    # 确认开始
    confirm = input(f"\n是否开始爬取 {config.base_url}？(y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 用户取消")
        return
    
    # 创建爬虫
    crawler = WebsiteCrawler(config)
    
    try:
        # 开始爬取
        report = crawler.crawl()
        
        # 显示结果
        print(f"\n📋 爬取完成报告:")
        print(f"   爬取耗时: {report['stats']['duration_seconds']:.2f} 秒")
        print(f"   成功页面: {report['stats']['pages_crawled']}")
        print(f"   失败页面: {report['stats']['pages_failed']}")
        print(f"   内容块数: {report['stats']['content_blocks_extracted']}")
        print(f"   成功率: {report['stats']['success_rate']:.2%}")
        
        # 保存报告
        report_file = crawler.save_report()
        
        # 保存到 Milvus
        print(f"\n💾 保存数据到 Milvus...")
        if crawler.save_to_milvus():
            print(f"✅ 数据已成功保存到 Milvus 集合: {config.collection_name}")
            print(f"\n🎉 现在可以使用以下命令进行 AI 查询:")
            print(f"   python query_milvus.py  # 使用集合名称: {config.collection_name}")
            print(f"   python llama_query.py   # LLaMA 智能问答")
        else:
            print(f"❌ 保存到 Milvus 失败")
        
    except Exception as e:
        print(f"❌ 爬取过程出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 