#!/usr/bin/env python3
"""
增强版 HTML 内容向量化存储系统
集成智能分块处理，提升内容质量
"""

import os
import re
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Tuple, Optional
import numpy as np

# HTML 解析相关
from bs4 import BeautifulSoup

# Milvus 相关
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, 
    DataType, Collection, MilvusException
)

# 智能分块处理
from smart_chunker import SmartChunker, ChunkMetadata

# 向量化相关
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    print("✅ 使用 sentence-transformers 进行语义向量化")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers 未安装，使用简化向量化")

class EnhancedHTMLToMilvusProcessor:
    """增强版 HTML 内容到 Milvus 的处理器，集成智能分块"""
    
    def __init__(self, host='localhost', port='19530', collection_name='enhanced_content'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.dimension = 384
        
        # 初始化智能分块器 - 严格控制在Milvus字段限制内
        self.chunker = SmartChunker(
            min_chunk_size=150,  # 最小150字符
            max_chunk_size=1500, # 最大1500字符（留500字符安全缓冲）
            overlap_size=30,     # 减少重叠到30字符
            target_chunk_size=600 # 目标600字符
        )
        
        # 初始化向量化模型
        if HAS_SENTENCE_TRANSFORMERS:
            print("🔧 加载增强版语义向量化模型...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ 增强版语义模型加载完成")
        else:
            self.model = None
            print("⚠️  将使用简化向量化")
        
        # 统计信息
        self.stats = {
            'total_pages_processed': 0,
            'total_chunks_created': 0,
            'high_quality_chunks': 0,
            'filtered_low_quality': 0,
            'multilingual_detection': {'ja': 0, 'zh': 0, 'en': 0}
        }
    
    def connect(self) -> bool:
        """连接到 Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ 成功连接到 Milvus: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接 Milvus 失败: {e}")
            return False
    
    def create_collection(self) -> bool:
        """创建增强版集合"""
        try:
            # 删除现有集合
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"🗑️  删除现有集合: {self.collection_name}")
            
            # 定义增强版字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chunk_index", dtype=DataType.INT32),
                FieldSchema(name="total_chunks", dtype=DataType.INT32),
                FieldSchema(name="quality_score", dtype=DataType.FLOAT),
                FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
                FieldSchema(name="word_count", dtype=DataType.INT32),
                FieldSchema(name="timestamp", dtype=DataType.INT64)
            ]
            
            schema = CollectionSchema(fields, "Enhanced website content with smart chunking")
            self.collection = Collection(self.collection_name, schema)
            
            print(f"✅ 创建增强版集合: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ 创建集合失败: {e}")
            return False
    
    def create_index(self):
        """创建优化索引"""
        try:
            # 向量索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 48, "efConstruction": 500}
            }
            
            self.collection.create_index("embedding", index_params)
            
            # 标量字段索引
            self.collection.create_index("language")
            self.collection.create_index("content_type") 
            self.collection.create_index("quality_score")
            
            print("✅ 创建优化索引完成")
            
        except Exception as e:
            print(f"❌ 创建索引失败: {e}")
    
    def text_to_vector(self, text: str) -> List[float]:
        """文本转向量"""
        if self.model:
            try:
                embedding = self.model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
            except Exception as e:
                print(f"❌ 向量化失败: {e}")
                return None
        else:
            # 简化版向量化
            return self._simple_text_vector(text)
    
    def _simple_text_vector(self, text: str) -> List[float]:
        """简化版文本向量化"""
        # 基于字符统计的简单向量化
        vector = [0.0] * self.dimension
        
        if not text:
            return vector
        
        # 基本特征
        vector[0] = len(text) / 1000.0  # 长度特征
        vector[1] = len(text.split()) / 100.0  # 词数特征
        
        # 字符频率特征
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # 填充向量的其余部分
        for i, (char, count) in enumerate(sorted(char_counts.items())[:self.dimension-2]):
            if i + 2 < self.dimension:
                vector[i + 2] = count / len(text)
        
        return vector
    
    def process_html_page(self, url: str, html_content: str, title: str = "") -> List[Dict]:
        """处理HTML页面，返回智能分块的内容"""
        print(f"🔄 智能处理页面: {url}")
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取和合并文本内容
            page_content = self._extract_page_content(soup)
            
            if not page_content:
                print(f"⚠️  页面无有效内容: {url}")
                return []
            
            # 使用智能分块器处理内容
            metadata = {
                'url': url,
                'title': title,
                'content_type': 'enhanced_page_content'
            }
            
            chunks = self.chunker.chunk_content(page_content, metadata)
            
            if not chunks:
                print(f"⚠️  智能分块无结果: {url}")
                self.stats['filtered_low_quality'] += 1
                return []
            
            # 准备存储的数据
            processed_items = []
            current_time = int(time.time())
            
            for chunk_text, chunk_metadata in chunks:
                # 向量化
                vector = self.text_to_vector(chunk_text)
                if not vector:
                    continue
                
                # 质量评估
                if chunk_metadata.quality_score >= 0.4:  # 高质量阈值
                    self.stats['high_quality_chunks'] += 1
                else:
                    self.stats['filtered_low_quality'] += 1
                    continue  # 跳过低质量块
                
                # 语言统计
                lang = chunk_metadata.language
                if lang in self.stats['multilingual_detection']:
                    self.stats['multilingual_detection'][lang] += 1
                
                item = {
                    'content': chunk_text,
                    'embedding': vector,
                    'url': url,
                    'title': title,
                    'content_type': 'smart_chunk',
                    'chunk_index': chunk_metadata.chunk_index,
                    'total_chunks': chunk_metadata.total_chunks,
                    'quality_score': chunk_metadata.quality_score,
                    'language': chunk_metadata.language,
                    'word_count': chunk_metadata.word_count,
                    'timestamp': current_time
                }
                
                processed_items.append(item)
            
            self.stats['total_pages_processed'] += 1
            self.stats['total_chunks_created'] += len(processed_items)
            
            print(f"✅ 智能分块完成: {len(processed_items)} 个高质量块")
            return processed_items
            
        except Exception as e:
            print(f"❌ 处理页面失败 {url}: {e}")
            return []
    
    def _extract_page_content(self, soup: BeautifulSoup) -> str:
        """提取页面的主要文本内容"""
        # 移除脚本和样式
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # 优先提取主要内容区域
        main_content = ""
        
        # 尝试找到主要内容区域
        main_selectors = [
            'main', 'article', '.content', '.main-content', 
            '#content', '#main', '.post-content', '.entry-content'
        ]
        
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = ' '.join([elem.get_text().strip() for elem in elements])
                break
        
        # 如果没有找到主要区域，提取body内容
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text()
        
        # 清理文本
        main_content = re.sub(r'\s+', ' ', main_content).strip()
        
        return main_content
    
    def save_items(self, items: List[Dict]) -> bool:
        """批量保存处理好的项目到Milvus"""
        if not items:
            return True
        
        try:
            # 过滤超长内容并截断
            filtered_items = []
            for item in items:
                content = item['content']
                if len(content) > 7900:
                    print(f"⚠️  内容过长，截断: {len(content)} -> 7900 字符")
                    content = content[:7900] + "..."  # 截断并添加省略号
                    item = item.copy()
                    item['content'] = content
                filtered_items.append(item)
            
            # 准备批量插入数据
            entities = [
                [item['content'] for item in filtered_items],
                [item['embedding'] for item in filtered_items],
                [item['url'] for item in filtered_items],
                [item['title'] for item in filtered_items], 
                [item['content_type'] for item in filtered_items],
                [item['chunk_index'] for item in filtered_items],
                [item['total_chunks'] for item in filtered_items],
                [item['quality_score'] for item in filtered_items],
                [item['language'] for item in filtered_items],
                [item['word_count'] for item in filtered_items],
                [item['timestamp'] for item in filtered_items]
            ]
            
            # 插入到Milvus
            mr = self.collection.insert(entities)
            
            print(f"✅ 成功保存 {len(filtered_items)} 个智能分块到Milvus")
            return True
            
        except Exception as e:
            print(f"❌ 保存到Milvus失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取处理统计信息"""
        return self.stats.copy()
    
    def print_stats(self):
        """打印统计信息"""
        print("\n📊 智能分块处理统计:")
        print("=" * 50)
        print(f"📄 处理页面数: {self.stats['total_pages_processed']}")
        print(f"🧩 生成块数: {self.stats['total_chunks_created']}")
        print(f"⭐ 高质量块: {self.stats['high_quality_chunks']}")
        print(f"🗑️ 过滤低质量: {self.stats['filtered_low_quality']}")
        print(f"🌐 语言分布:")
        for lang, count in self.stats['multilingual_detection'].items():
            lang_name = {'ja': '日语', 'zh': '中文', 'en': '英语'}.get(lang, lang)
            print(f"   {lang_name}: {count} 块")
        
        if self.stats['total_chunks_created'] > 0:
            quality_rate = self.stats['high_quality_chunks'] / self.stats['total_chunks_created']
            print(f"📈 质量率: {quality_rate:.1%}")
    
    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect("default")
            print("🔌 已断开Milvus连接")
        except:
            pass

def test_enhanced_processor():
    """测试增强版处理器"""
    processor = EnhancedHTMLToMilvusProcessor(collection_name="test_enhanced")
    
    if not processor.connect():
        return
    
    if not processor.create_collection():
        return
    
    # 测试HTML内容
    test_html = """
    <html>
    <body>
        <main>
            <h1>株式会社関電工について</h1>
            <p>株式会社関電工は、1944年に設立された日本の電気工事会社です。</p>
            <p>主な事業内容には、電力工事、電気設備工事、情報通信工事、土木工事などがあります。</p>
            <p>関電工は、持続可能な社会の実現に向けて、技術革新と環境保護に取り組んでいます。</p>
        </main>
    </body>
    </html>
    """
    
    # 処理
    items = processor.process_html_page("https://test.com", test_html, "テストページ")
    
    if items:
        processor.save_items(items)
    
    processor.print_stats()
    processor.disconnect()

if __name__ == "__main__":
    test_enhanced_processor()