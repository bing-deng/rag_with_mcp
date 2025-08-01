#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版PDF文档处理器
提供更好的性能和错误处理
"""

import os
import time
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass, asdict
import hashlib

# PDF处理
import fitz  # PyMuPDF
import numpy as np

# 机器学习和向量化
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Milvus相关
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_pages: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    vectorization_time: float = 0.0
    storage_time: float = 0.0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        return self
    
    def checkpoint(self, name: str):
        """记录检查点"""
        if self.start_time is None:
            self.start_time = time.time()
        self.checkpoints[name] = time.time() - self.start_time
    
    def get_elapsed(self, name: str = None) -> float:
        """获取经过的时间"""
        if name and name in self.checkpoints:
            return self.checkpoints[name]
        return time.time() - self.start_time if self.start_time else 0.0

class OptimizedVectorEmbedder:
    """优化的向量嵌入器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384, batch_size: int = 32):
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.model = None
        self._model_lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            if HAS_SENTENCE_TRANSFORMERS:
                logger.info(f"正在加载优化的语义向量化模型: {self.model_name}")
                with self._model_lock:
                    self.model = SentenceTransformer(self.model_name)
                logger.info("语义向量化模型加载完成")
            else:
                logger.warning("SentenceTransformers未安装，将使用简单向量化方法")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def embed_texts_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """批量向量化文本"""
        if not texts:
            return []
        
        if not self.model:
            # 简单向量化方法（fallback）
            return [self._simple_vectorize(text) for text in texts]
        
        try:
            # 分批处理
            all_embeddings = []
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                if show_progress and total_batches > 1:
                    logger.info(f"处理批次 {i//self.batch_size + 1}/{total_batches} ({len(batch_texts)} 个文本)")
                
                # 生成嵌入向量
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            # 降级到简单方法
            return [self._simple_vectorize(text) for text in texts]
    
    def _simple_vectorize(self, text: str) -> List[float]:
        """简单向量化方法（fallback）"""
        # 使用文本哈希生成固定维度向量
        text_hash = hashlib.md5(text.encode()).hexdigest()
        vector = []
        for i in range(0, min(len(text_hash), self.dimension), 2):
            vector.append(float(int(text_hash[i:i+2], 16)) / 255.0 - 0.5)
        
        # 填充到指定维度
        while len(vector) < self.dimension:
            vector.append(0.0)
        
        return vector[:self.dimension]

class OptimizedPDFProcessor:
    """优化的PDF处理器"""
    
    def __init__(self, 
                 min_chunk_size: int = 100, 
                 max_chunk_size: int = 800, 
                 overlap_size: int = 80,
                 max_workers: int = 4):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.max_workers = max_workers
        self.stats = ProcessingStats()
    
    def process_pdf_optimized(self, pdf_path: str) -> List[Dict[str, Any]]:
        """优化的PDF处理"""
        monitor = PerformanceMonitor().start()
        
        try:
            logger.info(f"开始优化处理PDF文档: {pdf_path}")
            
            # 打开PDF文档
            doc = fitz.open(pdf_path)
            self.stats.total_pages = len(doc)
            
            # 并行处理页面
            all_chunks = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交页面处理任务
                future_to_page = {
                    executor.submit(self._process_page, doc, page_num): page_num 
                    for page_num in range(len(doc))
                }
                
                # 收集结果
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        page_chunks = future.result()
                        all_chunks.extend(page_chunks)
                        
                        if page_num % 10 == 0:  # 每10页报告一次进度
                            logger.info(f"已处理页面: {page_num + 1}/{len(doc)}")
                            
                    except Exception as e:
                        logger.error(f"处理页面 {page_num} 时出错: {e}")
                        self.stats.error_count += 1
            
            doc.close()
            
            # 按页面顺序排序
            all_chunks.sort(key=lambda x: (x['page_number'], x['chunk_index']))
            
            self.stats.total_chunks = len(all_chunks)
            self.stats.processing_time = monitor.get_elapsed()
            
            logger.info(f"PDF处理完成，共生成 {len(all_chunks)} 个文本块")
            logger.info(f"处理时间: {self.stats.processing_time:.2f}秒")
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"PDF优化处理失败: {e}")
            raise
    
    def _process_page(self, doc, page_num: int) -> List[Dict[str, Any]]:
        """处理单个页面（线程安全）"""
        try:
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # 清理文本
            text = self._clean_text(text)
            
            if len(text.strip()) < self.min_chunk_size:
                return []
            
            # 分块处理
            chunks = self._chunk_text_optimized(text, page_num + 1)
            
            return chunks
            
        except Exception as e:
            logger.error(f"处理页面 {page_num} 失败: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """优化的文本清理"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 移除页面标记
        text = text.replace('page', '').strip()
        
        return text
    
    def _chunk_text_optimized(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """优化的文本分块"""
        if len(text) <= self.max_chunk_size:
            return [{
                'text': text,
                'page_number': page_number,
                'chunk_index': 0,
                'chunk_type': 'text',
                'char_count': len(text)
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            if end < len(text):
                # 寻找最佳分割点（句号、换行等）
                best_split = end
                for split_char in ['。', '．', '\n', '、', '，']:
                    split_pos = text.rfind(split_char, start, end)
                    if split_pos > start + self.min_chunk_size:
                        best_split = split_pos + 1
                        break
                
                chunk_text = text[start:best_split].strip()
            else:
                chunk_text = text[start:].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'page_number': page_number,
                    'chunk_index': chunk_index,
                    'chunk_type': 'text',
                    'char_count': len(chunk_text)
                })
                chunk_index += 1
            
            # 计算下一个起始位置（考虑重叠）
            if end >= len(text):
                break
            start = max(start + self.max_chunk_size - self.overlap_size, start + 1)
        
        return chunks

class OptimizedMilvusManager:
    """优化的Milvus管理器"""
    
    def __init__(self, 
                 use_lite: bool = True, 
                 lite_uri: str = "./milvus_demo.db",
                 collection_name: str = "pdf_documents",
                 dimension: int = 384,
                 batch_size: int = 100):
        
        self.use_lite = use_lite
        self.lite_uri = lite_uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.collection = None
        self.stats = ProcessingStats()
        
        self._connect_and_initialize()
    
    def _connect_and_initialize(self):
        """连接和初始化"""
        try:
            if self.use_lite:
                connections.connect("default", uri=self.lite_uri)
                logger.info(f"成功连接到Milvus Lite: {self.lite_uri}")
            else:
                connections.connect("default", host="localhost", port="19530")
                logger.info("成功连接到Milvus服务器")
            
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Milvus连接初始化失败: {e}")
            raise
    
    def _ensure_collection(self):
        """确保集合存在"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"集合已存在: {self.collection_name}")
        else:
            self._create_collection()
    
    def _create_collection(self):
        """创建集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="char_count", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(fields, f"优化的PDF文档向量化集合")
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"成功创建集合: {self.collection_name}")
        
        # 创建索引
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
    
    def insert_chunks_optimized(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """优化的批量插入"""
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            logger.error("块数据和向量数据不匹配")
            return False
        
        monitor = PerformanceMonitor().start()
        
        try:
            # 分批插入
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            inserted_count = 0
            
            for i in range(0, len(chunks), self.batch_size):
                batch_end = min(i + self.batch_size, len(chunks))
                batch_chunks = chunks[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                
                # 准备批次数据
                batch_data = [
                    [f"chunk_{chunk['page_number']}_{chunk['chunk_index']}_{i+j}" for j, chunk in enumerate(batch_chunks)],
                    [chunk['text'][:4000] for chunk in batch_chunks],  # 限制文本长度
                    [os.path.basename(chunk.get('pdf_path', 'unknown.pdf')) for chunk in batch_chunks],
                    [chunk['page_number'] for chunk in batch_chunks],
                    [chunk['chunk_type'] for chunk in batch_chunks],
                    [chunk.get('char_count', len(chunk['text'])) for chunk in batch_chunks],
                    batch_embeddings
                ]
                
                # 插入批次数据
                self.collection.insert(batch_data)
                inserted_count += len(batch_chunks)
                
                if total_batches > 1:
                    logger.info(f"已插入批次 {i//self.batch_size + 1}/{total_batches} ({inserted_count}/{len(chunks)} 个文档块)")
            
            # 刷新数据
            self.collection.flush()
            
            self.stats.storage_time = monitor.get_elapsed()
            logger.info(f"优化插入完成，共插入 {inserted_count} 个文档块")
            logger.info(f"存储时间: {self.stats.storage_time:.2f}秒")
            
            return True
            
        except Exception as e:
            logger.error(f"优化插入失败: {e}")
            return False
    
    def load_collection(self):
        """加载集合到内存"""
        try:
            self.collection.load()
            logger.info("集合已加载到内存")
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = {
                'num_entities': self.collection.num_entities,
                'collection_name': self.collection_name,
                'dimension': self.dimension
            }
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

def optimize_pdf_processing(pdf_path: str, output_stats: bool = True) -> Dict[str, Any]:
    """完整的优化PDF处理流程"""
    logger.info("=== 开始优化PDF处理流程 ===")
    
    total_monitor = PerformanceMonitor().start()
    
    try:
        # 1. PDF处理
        processor = OptimizedPDFProcessor(
            min_chunk_size=100,
            max_chunk_size=800,
            overlap_size=80,
            max_workers=4
        )
        chunks = processor.process_pdf_optimized(pdf_path)
        
        if not chunks:
            logger.error("PDF处理未生成任何文本块")
            return {"success": False, "error": "No chunks generated"}
        
        # 2. 向量化
        embedder = OptimizedVectorEmbedder(batch_size=64)
        texts = [chunk['text'] for chunk in chunks]
        
        vector_monitor = PerformanceMonitor().start()
        embeddings = embedder.embed_texts_batch(texts)
        vector_time = vector_monitor.get_elapsed()
        
        logger.info(f"向量化完成，处理时间: {vector_time:.2f}秒")
        
        # 3. 存储到Milvus
        milvus_manager = OptimizedMilvusManager()
        success = milvus_manager.insert_chunks_optimized(chunks, embeddings)
        
        if success:
            milvus_manager.load_collection()
        
        # 4. 生成统计报告
        total_time = total_monitor.get_elapsed()
        
        stats = {
            "success": success,
            "total_time": total_time,
            "processing_time": processor.stats.processing_time,
            "vectorization_time": vector_time,
            "storage_time": milvus_manager.stats.storage_time,
            "total_pages": processor.stats.total_pages,
            "total_chunks": len(chunks),
            "error_count": processor.stats.error_count,
            "performance_metrics": {
                "chunks_per_second": len(chunks) / total_time if total_time > 0 else 0,
                "pages_per_second": processor.stats.total_pages / processor.stats.processing_time if processor.stats.processing_time > 0 else 0,
                "vectorization_rate": len(chunks) / vector_time if vector_time > 0 else 0
            }
        }
        
        if output_stats:
            logger.info("=== 优化处理统计报告 ===")
            logger.info(f"总处理时间: {total_time:.2f}秒")
            logger.info(f"PDF解析时间: {processor.stats.processing_time:.2f}秒")
            logger.info(f"向量化时间: {vector_time:.2f}秒")
            logger.info(f"存储时间: {milvus_manager.stats.storage_time:.2f}秒")
            logger.info(f"处理页数: {processor.stats.total_pages}")
            logger.info(f"生成块数: {len(chunks)}")
            logger.info(f"错误数量: {processor.stats.error_count}")
            logger.info(f"处理速度: {stats['performance_metrics']['chunks_per_second']:.2f} 块/秒")
            logger.info("=== 优化处理完成 ===")
        
        return stats
        
    except Exception as e:
        logger.error(f"优化处理流程失败: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # 测试优化处理
    PDF_PATH = "/Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/pdf/high_takusoukun_web_manual_separate.pdf"
    
    if os.path.exists(PDF_PATH):
        result = optimize_pdf_processing(PDF_PATH)
        if result.get("success"):
            logger.info("✅ 优化处理成功完成")
        else:
            logger.error(f"❌ 优化处理失败: {result.get('error')}")
    else:
        logger.error(f"PDF文件不存在: {PDF_PATH}")