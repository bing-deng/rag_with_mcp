#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日语问答系统
"""

import logging
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_to_milvus import VectorEmbedder, MilvusPDFManager
from enhanced_qa_service import EnhancedQASystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_japanese_qa():
    """测试日语问答系统"""
    try:
        logger.info("🚀 测试日语问答系统")
        
        # 初始化组件
        logger.info("初始化系统组件...")
        embedder = VectorEmbedder(
            model_name='cl-nagoya/sup-simcse-ja-base',
            dimension=768
        )
        
        milvus_manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_japanese.db",
            collection_name="pdf_documents_japanese",
            dimension=768
        )
        
        # 加载集合
        milvus_manager.load_collection()
        
        # 创建增强问答系统
        qa_system = EnhancedQASystem(milvus_manager, embedder)
        
        # 测试问题列表
        test_questions = [
            "計量装置が傾いている原因として、例３では何が挙げられていますか？",
            "例３について教えてください",
            "板が腐った場合どうなりますか？",
            "計量装置の損傷について説明してください",
            "電気は止まっていない場合はどうですか？"
        ]
        
        logger.info("🔍 开始测试问答...")
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- 测试问题 {i} ---")
            logger.info(f"问题: {question}")
            
            # 使用增强问答系统
            result = qa_system.answer_question(question)
            
            if result.get('status') == 'success':
                logger.info(f"✅ 回答: {result['answer']}")
                logger.info(f"   置信度: {result['confidence']}")
                logger.info(f"   找到关键词: {result.get('keywords_found', [])}")
                logger.info(f"   结果总数: {result.get('total_results', 0)}")
                
                if result.get('sources'):
                    logger.info("   参考来源:")
                    for j, source in enumerate(result['sources'][:2], 1):
                        logger.info(f"     {j}. 页码: {source['page_number']}, 匹配类型: {source.get('match_type', 'unknown')}, 分数: {source['score']:.4f}")
            else:
                logger.error(f"❌ 回答失败: {result.get('error', '未知错误')}")
        
        # 特别测试关键问题
        logger.info("\n🎯 特别测试核心问题")
        key_question = "計量装置が傾いている原因として、例３では何が挙げられていますか？"
        result = qa_system.answer_question(key_question)
        
        if result.get('status') == 'success':
            answer = result['answer']
            keywords = result.get('keywords_found', [])
            
            # 检查答案质量
            if '例３' in keywords and ('板が腐って' in answer or '傾いている' in answer):
                logger.info("🎉 核心问题回答正确！")
                logger.info(f"   正确识别关键词: {keywords}")
                logger.info(f"   答案包含关键信息: {answer[:100]}...")
            else:
                logger.warning("⚠️ 核心问题回答可能不够准确")
                logger.info(f"   找到关键词: {keywords}")
                logger.info(f"   答案内容: {answer}")
        
        logger.info("\n✅ 日语问答系统测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎯 日语问答系统测试")
    logger.info("=" * 60)
    
    if test_japanese_qa():
        logger.info("🎉 测试成功！")
    else:
        logger.error("❌ 测试失败")
    
    logger.info("=" * 60)