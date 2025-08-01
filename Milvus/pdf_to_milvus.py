#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF文档向量化处理工具
将PDF文档处理成文本块，生成向量嵌入并存储到Milvus数据库中
"""

import os
import re
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# PDF处理
import fitz  # PyMuPDF
from PIL import Image
import io

# 机器学习和向量化
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("警告: sentence-transformers 未安装，将使用简单向量化方法")

# Milvus相关
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PDFChunk:
    """PDF文档块数据结构"""
    id: str
    text: str
    page_number: int
    chunk_index: int
    pdf_path: str
    pdf_name: str
    chunk_type: str  # 'text', 'image_caption', 'table', 'header', 'footer'
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1000, overlap_size: int = 100):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从PDF中提取文本内容"""
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 提取文本
                text = page.get_text()
                
                # 提取图像信息
                image_list = page.get_images()
                images_info = []
                for img_index, img in enumerate(image_list):
                    try:
                        # 获取图像基本信息
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # 生成图像说明
                        image_caption = f"页面 {page_num + 1} 图像 {img_index + 1} (格式: {image_ext})"
                        images_info.append({
                            'caption': image_caption,
                            'size': len(image_bytes),
                            'format': image_ext
                        })
                    except Exception as e:
                        logger.warning(f"处理图像时出错: {e}")
                
                # 提取表格区域（简单方法）
                tables = self._extract_table_regions(page)
                
                pages_content.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'images': images_info,
                    'tables': tables,
                    'bbox': page.rect
                })
            
            doc.close()
            return pages_content
            
        except Exception as e:
            logger.error(f"处理PDF文件 {pdf_path} 时出错: {e}")
            return []
    
    def _extract_table_regions(self, page) -> List[Dict[str, Any]]:
        """简单的表格区域检测"""
        # 这是一个简化的表格检测方法
        # 在实际应用中，可能需要更复杂的表格检测算法
        tables = []
        try:
            # 查找可能的表格标识符
            text = page.get_text()
            if any(marker in text.lower() for marker in ['表', 'table', '┌', '├', '│']):
                tables.append({
                    'type': 'potential_table',
                    'text': 'detected_table_content',
                    'confidence': 0.5
                })
        except Exception as e:
            logger.warning(f"表格检测出错: {e}")
        
        return tables
    
    def chunk_text(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """将文本分割成合适大小的块"""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        # 清理文本
        cleaned_text = self._clean_text(text)
        
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', cleaned_text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果当前块加上新段落超过最大长度，保存当前块
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'page_number': page_number,
                        'chunk_index': chunk_index,
                        'chunk_type': 'text'
                    })
                    chunk_index += 1
                
                # 开始新块，保留重叠内容
                overlap_text = current_chunk[-self.overlap_size:] if len(current_chunk) > self.overlap_size else ""
                current_chunk = overlap_text + " " + paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph
        
        # 处理最后一个块
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append({
                'text': current_chunk.strip(),
                'page_number': page_number,
                'chunk_index': chunk_index,
                'chunk_type': 'text'
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留中文
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff.,!?;:()[\]{}"-]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def process_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """处理整个PDF文档"""
        logger.info(f"开始处理PDF文档: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return []
        
        pdf_name = os.path.basename(pdf_path)
        pages_content = self.extract_text_from_pdf(pdf_path)
        
        if not pages_content:
            logger.error(f"无法从PDF中提取内容: {pdf_path}")
            return []
        
        all_chunks = []
        global_chunk_index = 0
        
        for page_content in pages_content:
            page_number = page_content['page_number']
            
            # 处理文本内容
            text_chunks = self.chunk_text(page_content['text'], page_number)
            
            for chunk_data in text_chunks:
                chunk_id = self._generate_chunk_id(pdf_path, page_number, global_chunk_index)
                
                chunk = PDFChunk(
                    id=chunk_id,
                    text=chunk_data['text'],
                    page_number=page_number,
                    chunk_index=global_chunk_index,
                    pdf_path=pdf_path,
                    pdf_name=pdf_name,
                    chunk_type=chunk_data['chunk_type'],
                    metadata={
                        'text_length': len(chunk_data['text']),
                        'processing_time': time.time(),
                        'source_type': 'pdf'
                    }
                )
                
                all_chunks.append(chunk)
                global_chunk_index += 1
            
            # 处理图像信息
            for img_idx, img_info in enumerate(page_content['images']):
                chunk_id = self._generate_chunk_id(pdf_path, page_number, global_chunk_index)
                
                chunk = PDFChunk(
                    id=chunk_id,
                    text=img_info['caption'],
                    page_number=page_number,
                    chunk_index=global_chunk_index,
                    pdf_path=pdf_path,
                    pdf_name=pdf_name,
                    chunk_type='image_caption',
                    metadata={
                        'image_size': img_info['size'],
                        'image_format': img_info['format'],
                        'source_type': 'pdf_image'
                    }
                )
                
                all_chunks.append(chunk)
                global_chunk_index += 1
        
        logger.info(f"PDF处理完成，共生成 {len(all_chunks)} 个文本块")
        return all_chunks
    
    def _generate_chunk_id(self, pdf_path: str, page_number: int, chunk_index: int) -> str:
        """生成唯一的块ID"""
        content = f"{pdf_path}_{page_number}_{chunk_index}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()

class VectorEmbedder:
    """向量嵌入生成器"""
    
    def __init__(self, model_name: str = 'cl-nagoya/sup-simcse-ja-base', dimension: int = 768):
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化嵌入模型"""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"正在加载语义向量化模型: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("语义向量化模型加载完成")
            except Exception as e:
                logger.warning(f"加载语义模型失败: {e}，将使用简单向量化")
                self.model = None
        else:
            logger.info("使用简单向量化方法")
    
    def embed_text(self, text: str) -> List[float]:
        """生成文本的向量嵌入"""
        if self.model is not None:
            # 使用语义向量化
            try:
                embedding = self.model.encode(text)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"语义向量化失败: {e}，使用简单方法")
        
        # 简单向量化方法
        return self._simple_vectorize(text)
    
    def _simple_vectorize(self, text: str) -> List[float]:
        """简单的文本向量化方法"""
        # 基于字符频率的简单向量化
        text = text.lower()
        vector = [0.0] * self.dimension
        
        for i, char in enumerate(text[:self.dimension]):
            vector[i] = float(ord(char)) / 1000.0
        
        # 添加一些统计特征
        if len(text) > 0:
            vector[0] = len(text) / 1000.0  # 文本长度
            vector[1] = text.count(' ') / max(1, len(text))  # 空格比例
            vector[2] = sum(c.isdigit() for c in text) / max(1, len(text))  # 数字比例
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def embed_chunks(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """为所有文本块生成向量嵌入"""
        logger.info(f"开始为 {len(chunks)} 个文本块生成向量嵌入")
        
        for i, chunk in enumerate(chunks):
            try:
                chunk.embedding = self.embed_text(chunk.text)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"已处理 {i + 1}/{len(chunks)} 个文本块")
                    
            except Exception as e:
                logger.error(f"为文本块生成嵌入时出错: {e}")
                # 使用零向量作为后备
                chunk.embedding = [0.0] * self.dimension
        
        logger.info("向量嵌入生成完成")
        return chunks

class MilvusPDFManager:
    """Milvus PDF数据管理器"""
    
    def __init__(self, host: str = "localhost", port: str = "19530", 
                 collection_name: str = "pdf_documents", dimension: int = 384,
                 use_lite: bool = False, lite_uri: str = "./milvus_demo.db"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.use_lite = use_lite
        self.lite_uri = lite_uri
        self.collection = None
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """连接到Milvus服务器或Milvus Lite"""
        try:
            if self.use_lite:
                # 使用Milvus Lite
                connections.connect("default", uri=self.lite_uri)
                logger.info(f"成功连接到Milvus Lite: {self.lite_uri}")
            else:
                # 使用标准Milvus服务器
                connections.connect("default", host=self.host, port=self.port)
                logger.info(f"成功连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _ensure_collection(self):
        """确保集合存在"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"使用现有集合: {self.collection_name}")
        else:
            self._create_collection()
    
    def _create_collection(self):
        """创建新的集合"""
        # 定义字段模式
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="pdf_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="text_length", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        # 创建集合模式
        schema = CollectionSchema(fields, f"PDF文档向量化集合 - {self.collection_name}")
        
        # 创建集合
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"成功创建集合: {self.collection_name}")
        
        # 创建索引
        self._create_index()
    
    def _create_index(self):
        """创建向量索引"""
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        try:
            self.collection.create_index("embedding", index_params)
            logger.info("向量索引创建完成")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
    
    def insert_chunks(self, chunks: List[PDFChunk]) -> bool:
        """将PDF块插入到Milvus中"""
        if not chunks:
            logger.warning("没有要插入的数据块")
            return False
        
        try:
            # 准备数据
            data = [
                [chunk.id for chunk in chunks],  # id
                [chunk.text for chunk in chunks],  # text
                [chunk.pdf_name for chunk in chunks],  # pdf_name
                [chunk.pdf_path for chunk in chunks],  # pdf_path
                [chunk.page_number for chunk in chunks],  # page_number
                [chunk.chunk_index for chunk in chunks],  # chunk_index
                [chunk.chunk_type for chunk in chunks],  # chunk_type
                [len(chunk.text) for chunk in chunks],  # text_length
                [json.dumps(chunk.metadata) for chunk in chunks],  # metadata
                [chunk.embedding for chunk in chunks]  # embedding
            ]
            
            # 插入数据
            insert_result = self.collection.insert(data)
            logger.info(f"成功插入 {len(chunks)} 个文档块到Milvus")
            
            # 刷新数据
            self.collection.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"插入数据到Milvus失败: {e}")
            return False
    
    def load_collection(self):
        """加载集合到内存"""
        try:
            self.collection.load()
            logger.info("集合已加载到内存")
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
    
    def search_similar(self, query_text: str, top_k: int = 5, 
                      pdf_name_filter: Optional[str] = None, embedder: Optional[VectorEmbedder] = None) -> List[Dict[str, Any]]:
        """搜索相似的文档块"""
        if not self.collection:
            logger.error("集合未初始化")
            return []
        
        try:
            # 生成查询向量 - 使用传入的embedder或创建新的
            if embedder is None:
                embedder = VectorEmbedder(dimension=self.dimension)
            query_vector = embedder.embed_text(query_text)
            
            # 设置搜索参数
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # 构建过滤表达式
            expr = None
            if pdf_name_filter:
                expr = f'pdf_name like "{pdf_name_filter}%"'
            
            # 执行搜索
            results = self.collection.search(
                [query_vector],
                "embedding",
                search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "text", "pdf_name", "page_number", "chunk_type", "metadata"]
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.entity.get("id"),
                        "text": hit.entity.get("text"),
                        "pdf_name": hit.entity.get("pdf_name"),
                        "page_number": hit.entity.get("page_number"),
                        "chunk_type": hit.entity.get("chunk_type"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "score": hit.score
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = {
                "name": self.collection.name,
                "num_entities": self.collection.num_entities,
                "description": self.collection.description,
                "has_index": len(self.collection.indexes) > 0,
                "indexes": [{"field": idx.field_name, "params": idx.params} for idx in self.collection.indexes]
            }
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

def main():
    """主函数：演示PDF向量化流程"""
    # 配置参数
    PDF_PATH = "/Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/pdf/high_takusoukun_web_manual_separate.pdf"
    DIMENSION = 384
    
    try:
        # 1. 处理PDF文档
        logger.info("=== 步骤1: 处理PDF文档 ===")
        processor = PDFProcessor(min_chunk_size=100, max_chunk_size=800, overlap_size=50)
        chunks = processor.process_pdf(PDF_PATH)
        
        if not chunks:
            logger.error("PDF处理失败，无法继续")
            return
        
        # 2. 生成向量嵌入
        logger.info("=== 步骤2: 生成向量嵌入 ===")
        embedder = VectorEmbedder(dimension=DIMENSION)
        chunks = embedder.embed_chunks(chunks)
        
        # 3. 存储到Milvus
        logger.info("=== 步骤3: 存储到Milvus ===")
        milvus_manager = MilvusPDFManager(dimension=DIMENSION)
        
        success = milvus_manager.insert_chunks(chunks)
        if not success:
            logger.error("存储到Milvus失败")
            return
        
        # 4. 加载集合
        logger.info("=== 步骤4: 加载集合到内存 ===")
        milvus_manager.load_collection()
        
        # 5. 测试搜索
        logger.info("=== 步骤5: 测试向量搜索 ===")
        test_queries = [
            "电柱番号",
            "計量器について",
            "設備種目別記入例",
            "電圧調査",
            "引込線"
        ]
        
        for query in test_queries:
            logger.info(f"\n查询: {query}")
            results = milvus_manager.search_similar(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                logger.info(f"  结果 {i}: (相关度: {result['score']:.4f})")
                logger.info(f"    页码: {result['page_number']}")
                logger.info(f"    类型: {result['chunk_type']}")
                logger.info(f"    内容: {result['text'][:100]}...")
        
        # 6. 显示统计信息
        logger.info("=== 步骤6: 集合统计信息 ===")
        stats = milvus_manager.get_collection_stats()
        logger.info(f"集合统计: {stats}")
        
        logger.info("=== PDF向量化处理完成 ===")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()