#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复搜索问题的脚本
清理重复数据并重新构建索引
"""

import logging
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder
from pymilvus import utility, connections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_search_issue():
    """修复搜索问题"""
    try:
        # 1. 连接到数据库
        logger.info("连接到Milvus Lite数据库...")
        connections.connect("default", uri="./milvus_demo.db")
        
        # 2. 检查当前状态
        collection_name = "pdf_documents"
        if utility.has_collection(collection_name):
            from pymilvus import Collection
            collection = Collection(collection_name)
            current_count = collection.num_entities
            logger.info(f"当前数据库中有 {current_count} 个实体")
            
            # 3. 清理集合并重新创建
            logger.info("删除现有集合以清理重复数据...")
            utility.drop_collection(collection_name)
            logger.info("集合已删除")
        
        # 4. 重新创建管理器
        logger.info("重新创建Milvus管理器...")
        manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_demo.db",
            collection_name=collection_name,
            dimension=384
        )
        
        # 5. 重新导入数据
        logger.info("重新导入PDF数据...")
        from pdf_to_milvus import PDFProcessor
        
        PDF_PATH = "/Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/pdf/high_takusoukun_web_manual_separate.pdf"
        
        # 处理PDF
        processor = PDFProcessor(min_chunk_size=100, max_chunk_size=800, overlap_size=80)
        chunks = processor.process_pdf(PDF_PATH)
        logger.info(f"生成了 {len(chunks)} 个文本块")
        
        # 向量化
        embedder = VectorEmbedder(dimension=384)
        chunks = embedder.embed_chunks(chunks)
        logger.info(f"向量化完成")
        
        # 插入数据
        success = manager.insert_chunks(chunks)
        if success:
            logger.info("数据插入成功")
            manager.load_collection()
            
            # 6. 测试搜索
            logger.info("测试搜索功能...")
            test_queries = ["弊社設備について", "計量器について", "電柱"]
            
            for query in test_queries:
                logger.info(f"\n🔍 测试搜索: '{query}'")
                results = manager.search_similar(query, top_k=3, embedder=embedder)
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  📄 结果 {i}: (相关度: {result['score']:.4f})")
                        logger.info(f"     页码: {result['page_number']}")
                        logger.info(f"     内容: {result['text'][:100]}...")
                else:
                    logger.info("  ❌ 未找到结果")
            
            logger.info(f"\n✅ 修复完成！数据库现在有 {manager.collection.num_entities} 个实体")
            
        else:
            logger.error("❌ 数据插入失败")
            
    except Exception as e:
        logger.error(f"修复过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    fix_search_issue()