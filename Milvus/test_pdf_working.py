#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理功能完整测试 - 降低参数以获取更多文本块
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
            connections.connect("default", uri="./milvus_demo.db")
            logger.info("成功连接到Milvus Lite")
        except Exception as e:
            logger.error(f"连接Milvus Lite失败: {e}")
            raise
    
    def _ensure_collection(self):
        """确保集合存在"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)  # 删除旧集合重新创建
            
        self._create_collection()
    
    def _create_collection(self):
        """创建新的集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(fields, f"PDF文档向量化测试集合")
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"成功创建集合: {self.collection_name}")
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "FLAT"
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
            data = [
                [chunk.id for chunk in chunks],
                [chunk.text for chunk in chunks],
                [chunk.pdf_name for chunk in chunks],
                [chunk.page_number for chunk in chunks],
                [chunk.chunk_type for chunk in chunks],
                [chunk.embedding for chunk in chunks]
            ]
            
            self.collection.insert(data)
            logger.info(f"成功插入 {len(chunks)} 个文档块")
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

def test_complete_pdf_processing():
    """完整测试PDF处理功能"""
    PDF_PATH = "/Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/pdf/high_takusoukun_web_manual_separate.pdf"
    DIMENSION = 384
    
    try:
        # 1. PDF文档处理 - 降低min_chunk_size以获取更多块
        logger.info("=== 测试PDF文档处理 ===")
        processor = PDFProcessor(min_chunk_size=50, max_chunk_size=500, overlap_size=50)  # 降低参数
        
        pages_content = processor.extract_text_from_pdf(PDF_PATH)
        logger.info(f"提取了 {len(pages_content)} 页内容")
        
        # 打印前几页的文本内容以调试
        for i, page in enumerate(pages_content[:3]):
            logger.info(f"页面 {page['page_number']} 文本长度: {len(page['text'])}")
            logger.info(f"页面 {page['page_number']} 前100字符: {page['text'][:100]}...")
        
        # 处理前5页内容
        test_chunks = []
        for page_idx, page_content in enumerate(pages_content[:5]):
            page_number = page_content['page_number']
            text_chunks = processor.chunk_text(page_content['text'], page_number)
            
            logger.info(f"页面 {page_number} 生成了 {len(text_chunks)} 个文本块")
            
            for chunk_data in text_chunks:
                chunk_id = f"chunk_{page_number}_{len(test_chunks)}"
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
        
        logger.info(f"总共生成了 {len(test_chunks)} 个测试文本块")
        
        if len(test_chunks) == 0:
            # 如果没有生成文本块，手动创建一些测试数据
            logger.info("手动创建测试数据...")
            from pdf_to_milvus import PDFChunk
            
            test_data = [
                "電柱番号について説明します。電柱には固有の番号が付けられています。",
                "計量器は電力の使用量を計測する装置です。定期的な点検が必要です。",
                "設備種目別記入例および注意事項を確認してください。",
                "申込項目と内容種別の一覧表をご参照ください。",
                "引込線・電線についての注意事項があります。"
            ]
            
            for i, text in enumerate(test_data):
                chunk = PDFChunk(
                    id=f"manual_chunk_{i}",
                    text=text,
                    page_number=1,
                    chunk_index=i,
                    pdf_path=PDF_PATH,
                    pdf_name=os.path.basename(PDF_PATH),
                    chunk_type='text',
                    metadata={'manual': True}
                )
                test_chunks.append(chunk)
            
            logger.info(f"手动创建了 {len(test_chunks)} 个测试文本块")
        
        # 2. 向量化
        logger.info("=== 测试向量化 ===")
        embedder = VectorEmbedder(dimension=DIMENSION)
        embedded_chunks = embedder.embed_chunks(test_chunks)
        logger.info(f"向量化完成，处理了 {len(embedded_chunks)} 个文本块")
        
        # 验证向量质量
        for i, chunk in enumerate(embedded_chunks[:3]):
            logger.info(f"文本块 {i+1}: {chunk.text[:50]}...")
            logger.info(f"  向量维度: {len(chunk.embedding)}")
            logger.info(f"  向量前5位: {chunk.embedding[:5]}")
        
        # 3. Milvus Lite存储
        logger.info("=== 测试Milvus Lite存储 ===")
        milvus_manager = MilvusLiteManager(
            collection_name="pdf_working_test",
            dimension=DIMENSION
        )
        
        success = milvus_manager.insert_data(embedded_chunks)
        if success:
            logger.info("数据插入成功")
            milvus_manager.load_collection()
            
            # 4. 搜索测试
            logger.info("=== 测试搜索功能 ===")
            test_queries = [
                "電柱番号",
                "計量器",  
                "設備種目",
                "申込項目",
                "引込線"
            ]
            
            for query in test_queries:
                logger.info(f"\n🔍 搜索查询: '{query}'")
                
                query_embedding = embedder.embed_text(query)
                results = milvus_manager.search_similar(query_embedding, top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  📄 结果 {i}: (相关度: {result['score']:.4f})")
                        logger.info(f"     页码: {result['page_number']}")
                        logger.info(f"     类型: {result['chunk_type']}")  
                        logger.info(f"     内容: {result['text'][:100]}...")
                        logger.info("")
                else:
                    logger.info("  ❌ 未找到相关结果")
            
            # 5. 统计信息
            logger.info("=== 集合统计信息 ===")
            num_entities = milvus_manager.collection.num_entities
            logger.info(f"📊 集合中共有 {num_entities} 个文档块")
            
            logger.info("🎉 === 测试完成 ===")
            
        else:
            logger.error("❌ 数据插入失败")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    test_complete_pdf_processing()