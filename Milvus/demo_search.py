#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF搜索系统演示脚本
直接演示搜索功能而不需要启动web服务
"""

import sys
import os
import logging
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_search_system():
    """演示搜索系统功能"""
    logger.info("🚀 启动PDF搜索系统演示")
    
    try:
        # 1. 初始化系统组件
        logger.info("📚 初始化系统组件...")
        
        # 使用现有的数据库
        milvus_manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_demo.db",
            collection_name="pdf_documents",
            dimension=384
        )
        
        vector_embedder = VectorEmbedder(dimension=384)
        
        # 加载集合到内存
        milvus_manager.load_collection()
        
        # 2. 获取系统统计信息
        logger.info("📊 获取系统统计信息...")
        stats = milvus_manager.get_collection_stats()
        logger.info(f"   数据库实体数量: {stats.get('num_entities', 0)}")
        logger.info(f"   集合名称: {stats.get('collection_name', 'unknown')}")
        logger.info(f"   向量维度: {stats.get('dimension', 0)}")
        
        # 3. 演示搜索功能
        logger.info("\n🔍 演示搜索功能...")
        
        test_queries = [
            {"query": "電柱番号", "description": "电线杆编号"},
            {"query": "計量器", "description": "计量器"},
            {"query": "設備種目", "description": "设备类别"},
            {"query": "申込項目", "description": "申请项目"},
            {"query": "引込線", "description": "引入线"}
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\n--- 搜索测试 {i}: {test_case['description']} ---")
            logger.info(f"查询: '{test_case['query']}'")
            
            try:
                # 执行搜索
                results = milvus_manager.search_similar(
                    query_text=test_case['query'],
                    top_k=3,
                    embedder=vector_embedder
                )
                
                if results:
                    logger.info(f"✅ 找到 {len(results)} 个相关结果:")
                    
                    for j, result in enumerate(results, 1):
                        score = result.get('score', 0)
                        page_num = result.get('page_number', 0)
                        text = result.get('text', '')
                        
                        # 截取文本预览
                        text_preview = text[:80] + ('...' if len(text) > 80 else '')
                        
                        logger.info(f"   结果 {j}: 相关度 {score:.4f}")
                        logger.info(f"            页码: {page_num}")
                        logger.info(f"            内容: {text_preview}")
                        
                        # 显示相关性评级
                        if score > 0.8:
                            relevance = "🟢 高"
                        elif score > 0.6:
                            relevance = "🟡 中"
                        else:
                            relevance = "🔴 低"
                        logger.info(f"            相关性: {relevance}")
                        
                else:
                    logger.warning("⚠️  未找到相关结果")
                    
            except Exception as e:
                logger.error(f"❌ 搜索失败: {e}")
        
        # 4. 演示问答功能
        logger.info("\n💬 演示问答功能...")
        
        test_questions = [
            "電柱番号について教えてください",
            "計量器の故障時の対応方法は？",
            "設備の種目にはどのようなものがありますか？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- 问答测试 {i} ---")
            logger.info(f"问题: {question}")
            
            try:
                # 搜索相关内容作为上下文
                search_results = milvus_manager.search_similar(
                    query_text=question,
                    top_k=3,
                    embedder=vector_embedder
                )
                
                if search_results:
                    # 计算平均相关度
                    avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
                    
                    # 获取最相关的结果
                    best_result = search_results[0]
                    best_text = best_result.get('text', '')
                    
                    # 生成简单答案
                    if avg_score > 0.7:
                        answer = f"根据文档内容：{best_text[:200]}{'...' if len(best_text) > 200 else ''}"
                        confidence = "🟢 高"
                    elif avg_score > 0.5:
                        answer = f"在相关文档中找到：{best_text[:150]}{'...' if len(best_text) > 150 else ''}"
                        confidence = "🟡 中"
                    else:
                        answer = "抱歉，没有找到充分的相关信息来回答这个问题。"
                        confidence = "🔴 低"
                    
                    logger.info(f"✅ 答案: {answer}")
                    logger.info(f"   置信度: {confidence}")
                    logger.info(f"   参考来源: {len(search_results)} 个文档片段")
                    
                else:
                    logger.warning("⚠️  没有找到相关信息")
                    
            except Exception as e:
                logger.error(f"❌ 问答失败: {e}")
        
        # 5. 系统性能统计
        logger.info("\n📈 系统性能统计:")
        logger.info(f"   总文档块数: {stats.get('num_entities', 0)}")
        logger.info(f"   向量维度: 384")
        logger.info(f"   搜索引擎: Milvus Lite")
        logger.info(f"   向量化模型: SentenceTransformers (all-MiniLM-L6-v2)")
        logger.info(f"   数据库大小: {os.path.getsize('./milvus_demo.db')/1024/1024:.2f} MB")
        
        logger.info("\n🎉 演示完成！系统运行正常")
        return True
        
    except Exception as e:
        logger.error(f"❌ 演示过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PDF文档智能搜索系统 - 功能演示")
    logger.info("=" * 60)
    
    success = demo_search_system()
    
    if success:
        logger.info("\n✅ 演示成功完成")
        logger.info("系统已准备就绪，可以处理PDF文档搜索和问答任务")
    else:
        logger.error("\n❌ 演示失败")
        logger.error("请检查系统配置和依赖")
    
    logger.info("=" * 60)