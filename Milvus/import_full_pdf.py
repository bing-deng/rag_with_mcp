#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的PDF数据导入脚本
"""

import os
import logging
from pdf_to_milvus import PDFProcessor, VectorEmbedder
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusManager:
    """Milvus数据库管理器"""
    
    def __init__(self, collection_name: str = "pdf_documents", dimension: int = 384):
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
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _ensure_collection(self):
        """确保集合存在"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"使用现有集合: {self.collection_name}")
            logger.info(f"集合中现有文档数量: {self.collection.num_entities}")
        else:
            self._create_collection()
    
    def _create_collection(self):
        """创建新的集合"""
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
        
        schema = CollectionSchema(fields, f"PDF文档向量化集合")
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
                [str(chunk.metadata) for chunk in chunks],  # metadata
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
    
    def search_similar(self, query_embedding, top_k=5):
        """搜索相似的文档块"""
        try:
            search_params = {"metric_type": "COSINE"}
            
            results = self.collection.search(
                [query_embedding],
                "embedding",
                search_params,
                limit=top_k,
                output_fields=["id", "text", "pdf_name", "page_number", "chunk_type", "text_length"]
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
                        "text_length": hit.entity.get("text_length"),
                        "score": hit.score
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

def import_full_pdf():
    """导入完整的PDF文档"""
    # 配置参数
    PDF_PATH = "/Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/pdf/high_takusoukun_web_manual_separate.pdf"
    DIMENSION = 384
    
    try:
        # 1. 处理PDF文档 - 使用更小的chunk来获取更多数据
        logger.info("=== 开始处理PDF文档 ===")
        processor = PDFProcessor(min_chunk_size=100, max_chunk_size=800, overlap_size=80)
        
        # 处理整个PDF文档
        all_chunks = processor.process_pdf(PDF_PATH)
        logger.info(f"PDF处理完成，共生成 {len(all_chunks)} 个文本块")
        
        if len(all_chunks) == 0:
            logger.error("没有生成任何文本块，退出")
            return
        
        # 显示前几个文本块的示例
        for i, chunk in enumerate(all_chunks[:3]):
            logger.info(f"文本块 {i+1}: {chunk.text[:100]}...")
        
        # 2. 生成向量嵌入
        logger.info("=== 开始生成向量嵌入 ===")
        embedder = VectorEmbedder(dimension=DIMENSION)
        
        # 分批处理以避免内存问题
        batch_size = 50
        all_embedded_chunks = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}, 包含 {len(batch)} 个文本块")
            
            embedded_batch = embedder.embed_chunks(batch)
            all_embedded_chunks.extend(embedded_batch)
        
        logger.info(f"向量化完成，共处理 {len(all_embedded_chunks)} 个文本块")
        
        # 3. 存储到Milvus
        logger.info("=== 存储到Milvus数据库 ===")
        milvus_manager = MilvusManager(
            collection_name="pdf_documents",
            dimension=DIMENSION
        )
        
        # 如果集合已存在且有数据，询问是否重新导入
        current_count = milvus_manager.collection.num_entities
        if current_count > 0:
            logger.info(f"集合中已有 {current_count} 个文档，将追加新数据")
        
        # 分批插入数据
        insert_batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(all_embedded_chunks), insert_batch_size):
            batch = all_embedded_chunks[i:i+insert_batch_size]
            success = milvus_manager.insert_data(batch)
            
            if success:
                total_inserted += len(batch)
                logger.info(f"已插入 {total_inserted}/{len(all_embedded_chunks)} 个文档块")
            else:
                logger.error(f"批次 {i//insert_batch_size + 1} 插入失败")
                break
        
        if total_inserted == len(all_embedded_chunks):
            logger.info("所有数据插入成功")
            
            # 4. 加载集合并测试搜索
            logger.info("=== 加载集合并测试搜索 ===")
            milvus_manager.load_collection()
            
            # 测试搜索功能
            test_queries = [
                "電柱番号について",
                "計量器の調査",
                "設備種目別記入例",
                "引込線・電線",
                "電圧調査"
            ]
            
            for query in test_queries:
                logger.info(f"\n🔍 测试搜索: '{query}'")
                query_embedding = embedder.embed_text(query)
                results = milvus_manager.search_similar(query_embedding, top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  📄 结果 {i}: (相关度: {result['score']:.4f})")
                        logger.info(f"     页码: {result['page_number']}, 长度: {result['text_length']}")
                        logger.info(f"     内容: {result['text'][:80]}...")
                else:
                    logger.info("  ❌ 未找到相关结果")
            
            # 5. 显示最终统计
            final_count = milvus_manager.collection.num_entities
            logger.info(f"\n📊 导入完成统计:")
            logger.info(f"   总文档块数: {final_count}")
            logger.info(f"   本次新增: {total_inserted}")
            logger.info(f"   数据库文件: ./milvus_demo.db")
            
            logger.info("🎉 === PDF数据导入完成 ===")
            
        else:
            logger.error("数据导入不完整，请检查错误信息")
        
    except Exception as e:
        logger.error(f"导入过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    import_full_pdf()