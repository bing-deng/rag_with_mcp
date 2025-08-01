#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理功能快速测试脚本
"""

import os
import logging
from pdf_to_milvus import PDFProcessor, VectorEmbedder, MilvusPDFManager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pdf_processing():
    """测试PDF处理功能"""
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
        
        # 只处理前5页的内容来快速测试
        test_chunks = []
        for page_idx, page_content in enumerate(pages_content[:5]):  # 只处理前5页
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
        
        # 2. 测试向量化（只处理前10个块）
        logger.info("=== 测试向量化 ===")
        embedder = VectorEmbedder(dimension=DIMENSION)
        
        # 只向量化前10个块进行快速测试
        test_sample = test_chunks[:10]
        embedded_chunks = embedder.embed_chunks(test_sample)
        
        logger.info(f"向量化完成，处理了 {len(embedded_chunks)} 个文本块")
        
        # 检查向量质量
        for i, chunk in enumerate(embedded_chunks[:3]):
            logger.info(f"文本块 {i+1}: {chunk.text[:100]}...")
            logger.info(f"向量维度: {len(chunk.embedding)}")
            logger.info(f"向量示例: {chunk.embedding[:5]}...")
        
        # 3. 测试Milvus存储
        logger.info("=== 测试Milvus存储 ===")
        milvus_manager = MilvusPDFManager(
            collection_name="test_pdf_collection",
            dimension=DIMENSION
        )
        
        # 插入测试数据
        success = milvus_manager.insert_chunks(embedded_chunks)
        if success:
            logger.info("数据插入成功")
            
            # 加载集合
            milvus_manager.load_collection()
            
            # 4. 测试搜索
            logger.info("=== 测试搜索功能 ===")
            test_queries = [
                "電柱番号",
                "計量器",
                "設備種目"
            ]
            
            for query in test_queries:
                logger.info(f"\n搜索查询: {query}")
                results = milvus_manager.search_similar(query, top_k=3)
                
                for i, result in enumerate(results, 1):
                    logger.info(f"  结果 {i}: (相关度: {result['score']:.4f})")
                    logger.info(f"    页码: {result['page_number']}")
                    logger.info(f"    内容: {result['text'][:80]}...")
            
            # 5. 显示统计信息
            logger.info("=== 集合统计 ===")
            stats = milvus_manager.get_collection_stats()
            logger.info(f"统计信息: {stats}")
            
        else:
            logger.error("数据插入失败")
        
        logger.info("=== 测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    test_pdf_processing()