#!/usr/bin/env python3
"""
HTML 内容向量化存储系统
将 HTML 网页内容解析、清理、向量化并存储到 Milvus 向量数据库中
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

# 向量化相关 (如果没有安装会提供替代方案)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    print("✅ 使用 sentence-transformers 进行语义向量化")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers 未安装，使用简化向量化")
    print("   安装命令: pip install sentence-transformers beautifulsoup4 requests")

class HTMLToMilvusProcessor:
    """HTML 内容到 Milvus 的处理器"""
    
    def __init__(self, host='localhost', port='19530', collection_name='html_content'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.dimension = 384
        
        # 初始化向量化模型
        if HAS_SENTENCE_TRANSFORMERS:
            print("🔧 加载语义向量化模型...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ 语义模型加载完成")
        else:
            self.model = None
    
    def connect_to_milvus(self):
        """连接到 Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ 成功连接到 Milvus: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def create_collection(self):
        """创建 HTML 内容集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),  # paragraph, heading, list等
            FieldSchema(name="word_count", dtype=DataType.INT64),
            FieldSchema(name="url_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(fields, "HTML 内容向量化存储集合")
        
        if utility.has_collection(self.collection_name):
            print(f"📝 集合 {self.collection_name} 已存在，正在删除...")
            utility.drop_collection(self.collection_name)
        
        self.collection = Collection(self.collection_name, schema)
        print(f"✅ 集合 {self.collection_name} 创建成功")
        return self.collection
    
    def create_index(self):
        """创建向量索引"""
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        print("🔧 正在创建索引...")
        self.collection.create_index("embedding", index_params)
        print("✅ 索引创建完成")
    
    def load_collection(self):
        """加载集合到内存"""
        print("📥 正在加载集合到内存...")
        self.collection.load()
        print("✅ 集合加载完成")
    
    def fetch_html(self, url: str) -> Optional[str]:
        """获取 HTML 内容"""
        try:
            print(f"🌐 正在获取网页: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            print(f"✅ 成功获取网页内容 ({len(response.text)} 字符)")
            return response.text
        except Exception as e:
            print(f"❌ 获取网页失败: {e}")
            return None
    
    def parse_html(self, html_content: str, base_url: str = "") -> List[Dict]:
        """解析 HTML 内容并提取结构化信息"""
        print("🔍 正在解析 HTML 内容...")
        
        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        
        # 移除不需要的标签
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # 提取网页标题
        title = soup.find('title')
        page_title = title.get_text().strip() if title else "无标题"
        
        extracted_content = []
        
        # 1. 提取标题内容 (h1-h6)
        for level in range(1, 7):
            headings = soup.find_all(f'h{level}')
            for heading in headings:
                text = self._clean_text(heading.get_text())
                if text and len(text) > 5:  # 过滤太短的标题
                    extracted_content.append({
                        'content': text,
                        'content_type': f'heading_h{level}',
                        'word_count': len(text.split())
                    })
        
        # 2. 提取段落内容
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = self._clean_text(p.get_text())
            if text and len(text) > 20:  # 过滤太短的段落
                # 如果段落太长，分割成多个块
                chunks = self._split_long_text(text, max_length=1000)
                for i, chunk in enumerate(chunks):
                    content_type = 'paragraph' if len(chunks) == 1 else f'paragraph_chunk_{i+1}'
                    extracted_content.append({
                        'content': chunk,
                        'content_type': content_type,
                        'word_count': len(chunk.split())
                    })
        
        # 3. 提取列表内容
        lists = soup.find_all(['ul', 'ol'])
        for lst in lists:
            items = lst.find_all('li')
            if len(items) > 1:  # 至少有2个列表项才处理
                list_text = []
                for item in items:
                    item_text = self._clean_text(item.get_text())
                    if item_text:
                        list_text.append(f"• {item_text}")
                
                if list_text:
                    combined_text = '\n'.join(list_text[:10])  # 最多取前10项
                    extracted_content.append({
                        'content': combined_text,
                        'content_type': 'list',
                        'word_count': len(combined_text.split())
                    })
        
        # 4. 提取表格内容 (简化处理)
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) > 1:  # 至少有表头和一行数据
                table_text = []
                for row in rows[:5]:  # 最多取前5行
                    cells = row.find_all(['td', 'th'])
                    row_text = ' | '.join([self._clean_text(cell.get_text()) for cell in cells])
                    if row_text.strip():
                        table_text.append(row_text)
                
                if table_text:
                    combined_text = '\n'.join(table_text)
                    extracted_content.append({
                        'content': combined_text,
                        'content_type': 'table',
                        'word_count': len(combined_text.split())
                    })
        
        # 5. 提取定义列表内容 (dt/dd) - 重要的企业信息通常在这里
        dts = soup.find_all('dt')
        dds = soup.find_all('dd')
        if dts and dds:
            definition_pairs = []
            for i, dt in enumerate(dts):
                dt_text = self._clean_text(dt.get_text())
                if i < len(dds):
                    dd = dds[i]
                    dd_text = self._clean_text(dd.get_text())
                    if dt_text and dd_text:
                        definition_pairs.append(f"{dt_text}: {dd_text}")
            
            if definition_pairs:
                # 将定义列表作为一个完整的内容块 
                combined_text = '\n'.join(definition_pairs)
                extracted_content.append({
                    'content': combined_text,
                    'content_type': 'definition_list',
                    'word_count': len(combined_text.split())
                })
                
                # 对于重要的定义项，也单独创建内容块（便于精确搜索）
                for pair in definition_pairs:
                    if any(keyword in pair for keyword in ['設立', '創立', '資本金', '従業員', '売上', '本社', '会社名']):
                        extracted_content.append({
                            'content': pair,
                            'content_type': 'company_info',
                            'word_count': len(pair.split())
                        })
        
        # 6. 提取地址信息 (address标签)
        addresses = soup.find_all('address')
        for addr in addresses:
            addr_text = self._clean_text(addr.get_text())
            if addr_text and len(addr_text) > 10:
                extracted_content.append({
                    'content': addr_text,
                    'content_type': 'address',
                    'word_count': len(addr_text.split())
                })
        
        # 7. 提取引用内容 (blockquote标签)
        blockquotes = soup.find_all('blockquote')
        for quote in blockquotes:
            quote_text = self._clean_text(quote.get_text())
            if quote_text and len(quote_text) > 20:
                extracted_content.append({
                    'content': quote_text,
                    'content_type': 'quote',
                    'word_count': len(quote_text.split())
                })
        
        # 8. 提取强调内容 (strong, em, b标签中的重要信息)
        emphasis_tags = soup.find_all(['strong', 'em', 'b'])
        emphasis_texts = []
        for tag in emphasis_tags:
            text = self._clean_text(tag.get_text())
            if text and len(text) > 3 and len(text) < 100:  # 避免太长的内容
                emphasis_texts.append(text)
        
        if emphasis_texts:
            # 合并强调内容
            combined_emphasis = ' | '.join(list(set(emphasis_texts))[:10])  # 去重并限制数量
            extracted_content.append({
                'content': combined_emphasis,
                'content_type': 'emphasis',
                'word_count': len(combined_emphasis.split())
            })
        
        # 9. 提取语义化内容 (section, article标签)
        semantic_tags = soup.find_all(['section', 'article'])
        for tag in semantic_tags:
            # 获取section/article的直接文本内容（排除嵌套的其他标签）
            tag_text = ""
            for string in tag.stripped_strings:
                if len(tag_text + string) < 500:  # 限制长度
                    tag_text += string + " "
                else:
                    break
            
            tag_text = self._clean_text(tag_text)
            if tag_text and len(tag_text) > 30:
                extracted_content.append({
                    'content': tag_text,
                    'content_type': f'semantic_{tag.name}',
                    'word_count': len(tag_text.split())
                })
        
        # 10. 提取预格式化内容 (pre, code标签)
        pre_tags = soup.find_all(['pre', 'code'])
        for tag in pre_tags:
            pre_text = tag.get_text()
            if pre_text and len(pre_text.strip()) > 10:
                # 保持原始格式
                extracted_content.append({
                    'content': pre_text.strip(),
                    'content_type': f'code_{tag.name}',
                    'word_count': len(pre_text.split())
                })
        
        # 11. 提取图表标题 (figcaption标签)
        figcaptions = soup.find_all('figcaption')
        for cap in figcaptions:
            cap_text = self._clean_text(cap.get_text())
            if cap_text and len(cap_text) > 5:
                extracted_content.append({
                    'content': cap_text,
                    'content_type': 'figure_caption',
                    'word_count': len(cap_text.split())
                })
        
        # 12. 提取重要的div内容（基于class或id判断重要性）
        important_divs = soup.find_all('div', class_=re.compile(r'(info|profile|about|company|business|service|feature|highlight|summary|overview|description)', re.I))
        important_divs += soup.find_all('div', id=re.compile(r'(info|profile|about|company|business|service|feature|highlight|summary|overview|description)', re.I))
        
        for div in important_divs[:5]:  # 限制数量避免过多
            # 提取div的直接文本内容
            div_text = ""
            for string in div.stripped_strings:
                if len(div_text + string) < 800:
                    div_text += string + " "
                else:
                    break
            
            div_text = self._clean_text(div_text)
            if div_text and len(div_text) > 30:
                # 尝试从class或id获取更具体的类型
                class_names = div.get('class', [])
                div_id = div.get('id', '')
                content_type = 'important_div'
                
                if any('company' in cls.lower() for cls in class_names) or 'company' in div_id.lower():
                    content_type = 'company_div'
                elif any('profile' in cls.lower() for cls in class_names) or 'profile' in div_id.lower():
                    content_type = 'profile_div'
                elif any('business' in cls.lower() for cls in class_names) or 'business' in div_id.lower():
                    content_type = 'business_div'
                
                extracted_content.append({
                    'content': div_text,
                    'content_type': content_type,
                    'word_count': len(div_text.split())
                })
        
        # 13. 提取span中的关键信息（数字、日期、重要标识）
        spans = soup.find_all('span')
        important_spans = []
        for span in spans:
            span_text = self._clean_text(span.get_text())
            if span_text and len(span_text) > 2 and len(span_text) < 50:
                # 检查是否包含重要信息（数字、日期、货币等）
                if re.search(r'(\d{4}年|\d+億|\d+万|\d+円|\d+名|\d+%)', span_text):
                    important_spans.append(span_text)
        
        if important_spans:
            # 合并重要的span内容
            combined_spans = ' | '.join(list(set(important_spans))[:8])
            extracted_content.append({
                'content': combined_spans,
                'content_type': 'key_data',
                'word_count': len(combined_spans.split())
            })
        
        # 为每个内容块添加公共信息
        url_hash = hashlib.md5(base_url.encode()).hexdigest()
        timestamp = int(time.time())
        
        for content in extracted_content:
            content.update({
                'url': base_url,
                'title': page_title,
                'url_hash': url_hash,
                'timestamp': timestamp
            })
        
        print(f"✅ 解析完成，提取了 {len(extracted_content)} 个内容块")
        return extracted_content
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 移除特殊字符但保留中文、日文和基本标点
        # \u4e00-\u9fff: 中日韩汉字
        # \u3040-\u309f: 平假名
        # \u30a0-\u30ff: 片假名
        # \u3000-\u303f: 日文标点符号
        # \uff00-\uffef: 全角字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f\uff00-\uffef.,!?;:()"\'-]', '', text)
        
        return text
    
    def _split_long_text(self, text: str, max_length: int = 1000) -> List[str]:
        """将长文本分割成较短的块"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?。！？]', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:max_length]]
    
    def text_to_vector(self, text: str) -> List[float]:
        """将文本转换为向量"""
        if HAS_SENTENCE_TRANSFORMERS and self.model:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        else:
            # 简化向量化方法
            words = text.lower().split()
            vector = np.random.normal(0, 1, self.dimension)
            
            # 基于文本长度和内容调整向量
            length_factor = min(len(words) / 100.0, 1.0)
            vector = vector * length_factor
            
            # 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector.tolist()
    
    def insert_html_content(self, content_blocks: List[Dict]) -> bool:
        """将解析后的 HTML 内容插入 Milvus"""
        if not content_blocks:
            print("⚠️  没有内容需要插入")
            return False
        
        print(f"📝 正在向量化并插入 {len(content_blocks)} 个内容块...")
        
        # 准备数据
        urls = []
        titles = []
        contents = []
        content_types = []
        word_counts = []
        url_hashes = []
        timestamps = []
        embeddings = []
        
        for i, block in enumerate(content_blocks):
            print(f"  📊 向量化内容块 {i+1}/{len(content_blocks)}: {block['content_type']}")
            
            # 处理内容长度限制
            content = block['content']
            if len(content) > 15000:
                print(f"  ⚠️  内容过长 ({len(content)} 字符)，截断到 15000 字符")
                content = content[:14900] + "..."  # 留一些余量
            
            # 处理标题长度限制
            title = block['title']
            if len(title) > 500:
                title = title[:497] + "..."
            
            # 处理URL长度限制
            url = block['url']
            if len(url) > 1000:
                url = url[:997] + "..."
            
            urls.append(url)
            titles.append(title)
            contents.append(content)
            content_types.append(block['content_type'])
            word_counts.append(block['word_count'])
            url_hashes.append(block['url_hash'])
            timestamps.append(block['timestamp'])
            embeddings.append(self.text_to_vector(content))
        
        # 插入数据
        data = [urls, titles, contents, content_types, word_counts, url_hashes, timestamps, embeddings]
        
        try:
            insert_result = self.collection.insert(data)
            self.collection.flush()
            print(f"✅ 成功插入 {insert_result.insert_count} 个内容块")
            return True
        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
            return False
    
    def search_content(self, query: str, top_k: int = 5, content_type_filter: str = None) -> List[Dict]:
        """搜索内容"""
        print(f"🔍 搜索查询: '{query}'")
        
        query_vector = self.text_to_vector(query)
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}
        }
        
        # 构建过滤条件
        expr = None
        if content_type_filter:
            expr = f'content_type like "{content_type_filter}%"'
        
        try:
            results = self.collection.search(
                [query_vector],
                "embedding",
                search_params,
                limit=top_k,
                expr=expr,
                output_fields=["url", "title", "content", "content_type", "word_count"]
            )
            
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": float(hit.score),
                        "url": hit.entity.get("url"),
                        "title": hit.entity.get("title"),
                        "content": hit.entity.get("content"),
                        "content_type": hit.entity.get("content_type"),
                        "word_count": hit.entity.get("word_count")
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.collection:
            return {}
        
        try:
            stats = {
                "collection_name": self.collection.name,
                "total_blocks": self.collection.num_entities,
                "dimension": self.dimension,
                "has_semantic_model": HAS_SENTENCE_TRANSFORMERS
            }
            
            # 获取内容类型分布
            content_types = self.collection.query(
                expr="id >= 0",
                output_fields=["content_type"],
                limit=1000
            )
            
            type_counts = {}
            for item in content_types:
                ct = item.get('content_type', 'unknown')
                type_counts[ct] = type_counts.get(ct, 0) + 1
            
            stats["content_type_distribution"] = type_counts
            return stats
            
        except Exception as e:
            print(f"⚠️  获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def disconnect(self):
        """断开连接"""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        print("🔌 已断开连接")

def create_sample_html_files():
    """创建示例 HTML 文件用于测试"""
    html_files = {
        "tech_article.html": """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Python 机器学习技术指南</title>
        </head>
        <body>
            <h1>Python 机器学习完整指南</h1>
            <h2>什么是机器学习？</h2>
            <p>机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和做出决策。通过算法和统计模型，机器学习系统可以从数据中识别模式并做出预测。</p>
            
            <h2>Python 在机器学习中的优势</h2>
            <ul>
                <li>丰富的库生态系统：scikit-learn、pandas、numpy</li>
                <li>简单易学的语法</li>
                <li>强大的数据处理能力</li>
                <li>活跃的社区支持</li>
            </ul>
            
            <h3>常用机器学习库</h3>
            <table>
                <tr><th>库名</th><th>用途</th><th>特点</th></tr>
                <tr><td>scikit-learn</td><td>通用机器学习</td><td>易用、文档完善</td></tr>
                <tr><td>TensorFlow</td><td>深度学习</td><td>谷歌开发、功能强大</td></tr>
                <tr><td>PyTorch</td><td>深度学习</td><td>动态图、研究友好</td></tr>
            </table>
            
            <p>开始机器学习之旅的第一步是理解数据。数据是机器学习的燃料，没有高质量的数据，即使是最先进的算法也无法产生有意义的结果。数据预处理包括数据清洗、特征工程和数据转换等步骤。</p>
        </body>
        </html>
        """,
        
        "health_guide.html": """
        <!DOCTYPE html>
        <html>
        <head>
            <title>健康生活方式指南</title>
        </head>
        <body>
            <h1>健康生活方式完全指南</h1>
            
            <h2>均衡饮食的重要性</h2>
            <p>均衡饮食是健康生活的基石。它不仅为我们的身体提供必要的营养素，还有助于维持理想的体重和预防慢性疾病。一个均衡的饮食计划应该包含各种颜色的蔬菜水果、全谷物、瘦蛋白质和健康脂肪。</p>
            
            <h3>每日营养建议</h3>
            <ul>
                <li>蔬菜水果：每天至少5份不同颜色的蔬果</li>
                <li>全谷物：选择糙米、全麦面包等未精制谷物</li>
                <li>蛋白质：鱼类、豆类、坚果和适量瘦肉</li>
                <li>健康脂肪：橄榄油、鳄梨、坚果中的不饱和脂肪</li>
                <li>水分：每天至少8杯水</li>
            </ul>
            
            <h2>规律运动的益处</h2>
            <p>运动不仅能帮助我们保持健康的体重，还能增强心血管功能、提高免疫力、改善心理健康。世界卫生组织建议成年人每周至少进行150分钟的中等强度有氧运动。</p>
            
            <h3>推荐的运动类型</h3>
            <ol>
                <li>有氧运动：跑步、游泳、骑自行车</li>
                <li>力量训练：举重、俯卧撑、深蹲</li>
                <li>柔韧性训练：瑜伽、太极、拉伸</li>
                <li>平衡训练：单脚站立、平衡板练习</li>
            </ol>
        </body>
        </html>
        """,
        
        "cooking_tips.html": """
        <!DOCTYPE html>
        <html>
        <head>
            <title>烹饪技巧大全</title>
        </head>
        <body>
            <h1>烹饪技巧与美食制作</h1>
            
            <h2>基本烹饪技法</h2>
            <p>掌握基本的烹饪技法是成为优秀厨师的第一步。不同的烹饪方法会产生不同的口感和风味，了解每种技法的特点和适用场景非常重要。</p>
            
            <h3>中式烹饪技法</h3>
            <ul>
                <li>炒：高温快速烹饪，保持食材脆嫩</li>
                <li>煮：用水或汤汁加热，适合制作汤品</li>
                <li>蒸：利用蒸汽加热，保持食材原味</li>
                <li>炖：长时间小火慢煮，使食材软烂入味</li>
                <li>烧：调味后焖煮，汁浓味厚</li>
            </ul>
            
            <h2>食材选择与搭配</h2>
            <p>新鲜的食材是美味佳肴的基础。选择食材时要注意观察其外观、气味和触感。不同的食材有不同的最佳搭配，合理的搭配不仅能提升口味，还能增加营养价值。</p>
            
            <table>
                <tr><th>食材类型</th><th>选择要点</th><th>最佳搭配</th></tr>
                <tr><td>蔬菜</td><td>色泽鲜艳、质地脆嫩</td><td>荤素搭配、颜色搭配</td></tr>
                <tr><td>肉类</td><td>无异味、弹性好</td><td>香料调味、蔬菜平衡</td></tr>
                <tr><td>海鲜</td><td>眼睛明亮、鳃色鲜红</td><td>柠檬去腥、清淡调味</td></tr>
            </table>
            
            <h3>调味的艺术</h3>
            <p>调味是烹饪的灵魂。合适的调味能够突出食材的天然味道，而不是掩盖它。调味要遵循循序渐进的原则，先少量添加，品尝后再调整。记住，盐要后放，以免影响其他调料的吸收。</p>
        </body>
        </html>
        """
    }
    
    # 创建示例文件目录
    os.makedirs("sample_html", exist_ok=True)
    
    for filename, content in html_files.items():
        filepath = os.path.join("sample_html", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 创建示例文件: {filepath}")
    
    return list(html_files.keys())

def main():
    """主函数 - 演示 HTML 内容向量化存储"""
    print("🌐 HTML 内容向量化存储系统演示")
    print("=" * 60)
    
    # 创建处理器实例
    processor = HTMLToMilvusProcessor()
    
    try:
        # 1. 连接到 Milvus
        if not processor.connect_to_milvus():
            return
        
        # 2. 创建集合
        processor.create_collection()
        processor.create_index()
        processor.load_collection()
        
        # 3. 创建示例 HTML 文件
        print("\n📄 创建示例 HTML 文件...")
        sample_files = create_sample_html_files()
        
        # 4. 处理本地 HTML 文件
        all_content_blocks = []
        for filename in sample_files:
            filepath = os.path.join("sample_html", filename)
            print(f"\n🔄 处理文件: {filepath}")
            
            # 读取 HTML 文件
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 解析 HTML
            content_blocks = processor.parse_html(html_content, base_url=filepath)
            all_content_blocks.extend(content_blocks)
        
        # 5. 插入到 Milvus
        processor.insert_html_content(all_content_blocks)
        
        # 6. 显示统计信息
        print("\n📊 存储统计信息:")
        stats = processor.get_statistics()
        print(f"   集合名称: {stats.get('collection_name', 'N/A')}")
        print(f"   内容块总数: {stats.get('total_blocks', 'N/A')}")
        print(f"   向量维度: {stats.get('dimension', 'N/A')}")
        print(f"   语义模型: {'✅ 已启用' if stats.get('has_semantic_model') else '❌ 简化模式'}")
        
        if 'content_type_distribution' in stats:
            print("   内容类型分布:")
            for content_type, count in stats['content_type_distribution'].items():
                print(f"     - {content_type}: {count}")
        
        # 7. 演示搜索功能
        print(f"\n🔍 搜索演示:")
        print("-" * 40)
        
        test_queries = [
            "机器学习算法",
            "健康饮食建议", 
            "烹饪技巧",
            "Python 编程",
            "运动锻炼方法"
        ]
        
        for query in test_queries:
            print(f"\n🎯 查询: '{query}'")
            results = processor.search_content(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. 【{result['content_type']}】相似度: {result['score']:.4f}")
                    print(f"      来源: {result['url']}")
                    print(f"      内容: {result['content'][:80]}...")
            else:
                print("  未找到相关结果")
        
        # 8. 按内容类型搜索
        print(f"\n🏷️  按内容类型搜索演示:")
        print("-" * 40)
        
        query = "营养健康"
        content_type = "paragraph"
        print(f"在内容类型 '{content_type}' 中搜索: '{query}'")
        
        filtered_results = processor.search_content(query, top_k=3, content_type_filter=content_type)
        for i, result in enumerate(filtered_results, 1):
            print(f"  {i}. {result['content'][:100]}... (分数: {result['score']:.4f})")
        
        print(f"\n✅ HTML 内容向量化存储演示完成!")
        
        # 9. 提供进一步的使用建议
        print(f"\n💡 进一步使用建议:")
        print("   1. 修改 sample_html/ 目录下的 HTML 文件来测试不同内容")
        print("   2. 在 main() 函数中添加网络爬取功能 (使用 processor.fetch_html())")
        print("   3. 调整向量化模型以获得更好的语义理解")
        print("   4. 添加更多的内容类型和过滤条件")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        processor.disconnect()

def demo_web_crawling():
    """演示从网络爬取 HTML 内容 (可选功能)"""
    print("🌐 网络爬取演示 (请确保目标网站允许爬取)")
    
    # 示例网站列表 (请根据实际情况修改)
    urls = [
        "https://disclosure2.edinet-fsa.go.jp/WZEK0040.aspx?S100FWD6,,",  # 请替换为实际的网站URL
        # "https://your-blog.com/article1",
        # "https://docs.python.org/3/tutorial/",
    ]
    
    processor = HTMLToMilvusProcessor(collection_name="web_content")
    
    try:
        processor.connect_to_milvus()
        processor.create_collection()
        processor.create_index()
        processor.load_collection()
        
        all_content_blocks = []
        
        for url in urls:
            html_content = processor.fetch_html(url)
            if html_content:
                content_blocks = processor.parse_html(html_content, base_url=url)
                all_content_blocks.extend(content_blocks)
        
        if all_content_blocks:
            processor.insert_html_content(all_content_blocks)
            print("✅ 网络内容爬取和存储完成")
        else:
            print("⚠️  没有成功获取任何网络内容")
            
    except Exception as e:
        print(f"❌ 网络爬取失败: {e}")
    finally:
        processor.disconnect()

if __name__ == "__main__":
    # main()
    
    # 如果需要测试网络爬取功能，取消下面的注释
    # print("\n" + "="*60)
    demo_web_crawling() 