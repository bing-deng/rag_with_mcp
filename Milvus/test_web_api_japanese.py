#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日语优化的Web API
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_japanese_web_api():
    """测试日语Web API"""
    base_url = "http://localhost:5001"
    
    try:
        logger.info("🚀 测试日语优化Web API")
        
        # 1. 健康检查
        logger.info("1️⃣ 健康检查...")
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ 服务状态: {health_data['status']}")
            logger.info(f"   数据库连接: {health_data['milvus_connected']}")
            logger.info(f"   文档数量: {health_data['collection_stats'].get('num_entities', 0)}")
        else:
            logger.error(f"❌ 健康检查失败: {response.status_code}")
            return False
        
        # 2. 获取系统统计
        logger.info("2️⃣ 获取系统统计...")
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats_data = response.json()
            config = stats_data.get('config', {})
            logger.info(f"✅ 系统配置:")
            logger.info(f"   模型: {config.get('model', 'unknown')}")
            logger.info(f"   维度: {config.get('dimension', 'unknown')}")
            logger.info(f"   集合: {config.get('collection_name', 'unknown')}")
        
        # 3. 测试搜索功能
        logger.info("3️⃣ 测试搜索功能...")
        search_query = {
            "query": "計量装置",
            "top_k": 3
        }
        
        response = requests.post(
            f"{base_url}/api/search",
            json=search_query,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            search_data = response.json()
            logger.info(f"✅ 搜索成功，找到 {search_data['results_count']} 个结果")
            for i, result in enumerate(search_data['results'][:2], 1):
                logger.info(f"   结果 {i}: 分数={result['score']:.4f}, 内容={result['text'][:50]}...")
        else:
            logger.error(f"❌ 搜索失败: {response.status_code}")
        
        # 4. 测试关键问题的问答功能
        logger.info("4️⃣ 测试关键问题问答...")
        qa_query = {
            "question": "計量装置が傾いている原因として、例３では何が挙げられていますか？"
        }
        
        response = requests.post(
            f"{base_url}/api/answer",
            json=qa_query,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            qa_data = response.json()
            logger.info(f"✅ 问答成功!")
            logger.info(f"   问题: {qa_data['question']}")
            logger.info(f"   答案: {qa_data['answer']}")
            logger.info(f"   置信度: {qa_data['confidence']}")
            logger.info(f"   关键词: {qa_data.get('keywords_found', [])}")
            logger.info(f"   结果数: {qa_data.get('total_results', 0)}")
            
            # 检查答案质量
            answer = qa_data['answer']
            keywords = qa_data.get('keywords_found', [])
            
            if '例３' in keywords and ('板が腐って' in answer or '傾いている' in answer):
                logger.info("🎉 关键问题回答正确！日语优化模型工作良好")
            else:
                logger.warning("⚠️ 答案可能需要进一步优化")
                
            if qa_data.get('sources'):
                logger.info("   参考来源:")
                for i, source in enumerate(qa_data['sources'][:2], 1):
                    match_type = source.get('match_type', 'unknown')
                    logger.info(f"     {i}. 页码: {source['page_number']}, 匹配: {match_type}, 分数: {source['score']:.4f}")
        else:
            logger.error(f"❌ 问答失败: {response.status_code}")
            try:
                error_data = response.json()
                logger.error(f"   错误信息: {error_data.get('error', 'unknown')}")
            except:
                logger.error(f"   响应内容: {response.text}")
        
        # 5. 测试其他日语问题
        logger.info("5️⃣ 测试其他日语问题...")
        test_questions = [
            "例３について教えてください",
            "板が腐った場合どうなりますか？",
            "計量装置の損傷について説明してください"
        ]
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"   测试问题 {i}: {question}")
            
            qa_query = {"question": question}
            response = requests.post(
                f"{base_url}/api/answer",
                json=qa_query,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                qa_data = response.json()
                answer = qa_data['answer']
                confidence = qa_data['confidence']
                logger.info(f"   ✅ 答案: {answer[:80]}...")
                logger.info(f"   置信度: {confidence}")
            else:
                logger.error(f"   ❌ 问题 {i} 失败")
        
        logger.info("🎉 日语Web API测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎯 日语优化Web API测试")
    logger.info("=" * 60)
    
    if test_japanese_web_api():
        logger.info("🎉 测试成功！")
    else:
        logger.error("❌ 测试失败")
    
    logger.info("=" * 60)