#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试搜索功能，查看原始搜索结果
"""

import logging
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_search():
    """调试搜索功能"""
    try:
        # 初始化系统
        milvus_manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_demo.db",
            collection_name="pdf_documents",
            dimension=384
        )
        
        vector_embedder = VectorEmbedder(dimension=384)
        milvus_manager.load_collection()
        
        # 测试多个相关搜索词
        test_queries = [
            "計量装置",
            "計量器", 
            "傾いている",
            "例３",
            "例3",
            "原因",
            "計量装置 傾いている",
            "計量器 傾斜"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"搜索词: '{query}'")
            print('='*60)
            
            # 执行搜索，不过滤任何结果
            results = milvus_manager.search_similar(
                query_text=query,
                top_k=10,
                embedder=vector_embedder
            )
            
            if results:
                print(f"找到 {len(results)} 个结果:")
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    page_num = result.get('page_number', 0)
                    text = result.get('text', '')
                    chunk_type = result.get('chunk_type', '')
                    
                    print(f"\n结果 {i}: (分数: {score:.4f}, 页码: {page_num}, 类型: {chunk_type})")
                    print(f"内容: {text[:200]}{'...' if len(text) > 200 else ''}")
                    
            else:
                print("未找到结果")
                
        # 额外检查：查看所有文本类型的内容
        print(f"\n{'='*60}")
        print("查看文本类型内容样本")
        print('='*60)
        
        # 搜索包含"計量"的内容
        results = milvus_manager.search_similar(
            query_text="計量",
            top_k=20,
            embedder=vector_embedder
        )
        
        text_results = []
        for result in results or []:
            text = result.get('text', '')
            chunk_type = result.get('chunk_type', '')
            if chunk_type == 'text' and len(text.strip()) > 30:
                text_results.append(result)
        
        print(f"找到 {len(text_results)} 个文本类型的结果:")
        for i, result in enumerate(text_results[:5], 1):
            score = result.get('score', 0)
            page_num = result.get('page_number', 0)
            text = result.get('text', '')
            
            print(f"\n文本结果 {i}: (分数: {score:.4f}, 页码: {page_num})")
            print(f"内容: {text}")
            
    except Exception as e:
        logger.error(f"调试失败: {e}")

if __name__ == "__main__":
    debug_search()