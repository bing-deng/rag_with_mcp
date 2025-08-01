#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理功能测试脚本 - 使用Milvus Lite
"""

import os
import logging
from pdf_to_milvus import PDFProcessor, VectorEmbedder

# Milvus Lite 版本
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusLiteManager:
    """Milvus Lite管理器"""
    
    def __init__(self, collection_name: str = "pdf_test", dimension: int = 384):
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection = None
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """连接到Milvus Lite"""
        try:
            # 使用Milvus Lite（嵌入式版本）
            connections.connect("default", uri="./milvus_demo.db")
            logger.info("成功连接到Milvus Lite")
        except Exception as e:
            logger.error(f"连接Milvus Lite失败: {e}")
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
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        # 创建集合模式
        schema = CollectionSchema(fields, f"PDF文档向量化测试集合")
        
        # 创建集合
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"成功创建集合: {self.collection_name}")
        
        # 创建索引
        self._create_index()
    
    def _create_index(self):
        """创建向量索引"""
        index_params = {
            "metric_type": "COSINE",
            "index_type": "FLAT"  # 对于小数据集使用FLAT索引
        }
        
        try:
            self.collection.create_index("embedding", index_params)
            logger.info("向量索引创建完成")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
    
    def insert_data(self, chunks):
        """插入数据"""
        if not chunks:
            logger.warning("没有要插入的数据块")
            return False
        
        try:
            # 准备数据
            data = [
                [chunk.id for chunk in chunks],  # id
                [chunk.text for chunk in chunks],  # text
                [chunk.pdf_name for chunk in chunks],  # pdf_name
                [chunk.page_number for chunk in chunks],  # page_number
                [chunk.chunk_type for chunk in chunks],  # chunk_type
                [chunk.embedding for chunk in chunks]  # embedding
            ]
            
            # 插入数据
            self.collection.insert(data)
            logger.info(f"成功插入 {len(chunks)} 个文档块")
            
            # 刷新数据
            self.collection.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            return False
    
    def load_collection(self):
        """加载集合到内存"""
        try:
            self.collection.load()
            logger.info("集合已加载到内存")
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
    
    def search_similar(self, query_embedding, top_k=3):
        """搜索相似的文档块"""
        try:
            search_params = {"metric_type": "COSINE"}
            
            results = self.collection.search(
                [query_embedding],
                "embedding",
                search_params,
                limit=top_k,
                output_fields=["id", "text", "pdf_name", "page_number", "chunk_type"]
            )
            
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.entity.get("id"),
                        "text": hit.entity.get("text"),
                        "pdf_name": hit.entity.get("pdf_name"),
                        "page_number": hit.entity.get("page_number"),
                        "chunk_type": hit.entity.get("chunk_type"),
                        "score": hit.score
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

def test_pdf_with_lite():
    """使用Milvus Lite测试PDF处理功能"""
    # 配置参数
    PDF_PATH = "/Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/pdf/high_takusoukun_web_manual_separate.pdf"
    DIMENSION = 384
    
    try:
        # 1. 测试PDF文档处理（只处理前几页）
        logger.info("=== 测试PDF文档处理 ===")
        processor = PDFProcessor(min_chunk_size=200, max_chunk_size=1000, overlap_size=100)
        
        # 提取PDF内容
        pages_content = processor.extract_text_from_pdf(PDF_PATH)
        logger.info(f"提取了 {len(pages_content)} 页内容")
        
        # 只处理前3页的内容来快速测试
        test_chunks = []
        for page_idx, page_content in enumerate(pages_content[:3]):  # 只处理前3页
            page_number = page_content['page_number']
            text_chunks = processor.chunk_text(page_content['text'], page_number)
            
            for chunk_data in text_chunks:
                chunk_id = f"test_chunk_{page_number}_{len(test_chunks)}"
                from pdf_to_milvus import PDFChunk
                
                chunk = PDFChunk(
                    id=chunk_id,
                    text=chunk_data['text'],
                    page_number=page_number,
                    chunk_index=len(test_chunks),
                    pdf_path=PDF_PATH,
                    pdf_name=os.path.basename(PDF_PATH),
                    chunk_type=chunk_data['chunk_type'],
                    metadata={'test': True}
                )
                test_chunks.append(chunk)
        
        logger.info(f"生成了 {len(test_chunks)} 个测试文本块")
        
        # 2. 测试向量化
        logger.info("=== 测试向量化 ===")
        embedder = VectorEmbedder(dimension=DIMENSION)
        
        # 向量化所有块
        embedded_chunks = embedder.embed_chunks(test_chunks)
        
        logger.info(f"向量化完成，处理了 {len(embedded_chunks)} 个文本块")
        
        # 3. 测试Milvus Lite存储
        logger.info("=== 测试Milvus Lite存储 ===")
        milvus_manager = MilvusLiteManager(
            collection_name="pdf_test_lite",
            dimension=DIMENSION
        )
        
        # 插入测试数据
        success = milvus_manager.insert_data(embedded_chunks)
        if success:
            logger.info("数据插入成功")
            
            # 加载集合
            milvus_manager.load_collection()
            
            # 4. 测试搜索
            logger.info("=== 测试搜索功能 ===")
            test_queries = [
                "電柱番号",
                "計量器",
                "設備種目",
                "申込項目"
            ]
            
            for query in test_queries:
                logger.info(f"\n搜索查询: {query}")
                
                # 生成查询向量
                query_embedding = embedder.embed_text(query)
                
                # 执行搜索
                results = milvus_manager.search_similar(query_embedding, top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  结果 {i}: (相关度: {result['score']:.4f})")
                        logger.info(f"    页码: {result['page_number']}")
                        logger.info(f"    内容: {result['text'][:80]}...")
                else:
                    logger.info("  未找到相关结果")
            
            logger.info("=== 测试完成 ===")
            
        else:
            logger.error("数据插入失败")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    test_pdf_with_lite()