#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PDF搜索API功能
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_endpoints():
    """测试API端点功能"""
    base_url = "http://localhost:5001"
    
    # 测试数据
    test_queries = [
        {"query": "電柱番号", "description": "电线杆编号"},
        {"query": "計量器", "description": "计量器"},
        {"query": "設備種目", "description": "设备类别"},
        {"query": "申込項目", "description": "申请项目"},
        {"query": "引込線", "description": "引入线"}
    ]
    
    test_questions = [
        {"question": "電柱番号について教えてください", "description": "关于电线杆编号的信息"},
        {"question": "計量器の故障時の対応方法は？", "description": "计量器故障时的处理方法"},
        {"question": "設備の種目にはどのようなものがありますか？", "description": "设备类别有哪些"}
    ]
    
    logger.info("=== 开始API功能测试 ===")
    
    # 1. 测试健康检查
    logger.info("\n1. 测试健康检查端点...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ 健康检查通过")
            health_data = response.json()
            logger.info(f"   状态: {health_data.get('status')}")
            logger.info(f"   Milvus连接: {health_data.get('milvus_connected')}")
            if 'collection_stats' in health_data:
                stats = health_data['collection_stats']
                logger.info(f"   文档数量: {stats.get('num_entities', 0)}")
        else:
            logger.error(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 健康检查请求失败: {e}")
        return False
    
    # 2. 测试搜索功能
    logger.info("\n2. 测试搜索功能...")
    for i, test_case in enumerate(test_queries, 1):
        logger.info(f"\n   测试 {i}: {test_case['description']} - '{test_case['query']}'")
        try:
            payload = {
                "query": test_case['query'],
                "top_k": 3
            }
            response = requests.post(
                f"{base_url}/api/search",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                search_data = response.json()
                if search_data.get('status') == 'success':
                    results = search_data.get('results', [])
                    logger.info(f"   ✅ 找到 {len(results)} 个相关结果")
                    
                    for j, result in enumerate(results[:2], 1):  # 只显示前2个结果
                        logger.info(f"      结果 {j}: 相关度 {result['score']:.4f}")
                        logger.info(f"               页码: {result['page_number']}")
                        logger.info(f"               内容: {result['text'][:60]}...")
                else:
                    logger.warning(f"   ⚠️  搜索返回错误: {search_data.get('error', '未知错误')}")
            else:
                logger.error(f"   ❌ 搜索请求失败: {response.status_code}")
                logger.error(f"      响应: {response.text}")
                
        except Exception as e:
            logger.error(f"   ❌ 搜索请求异常: {e}")
    
    # 3. 测试问答功能
    logger.info("\n3. 测试问答功能...")
    for i, test_case in enumerate(test_questions, 1):
        logger.info(f"\n   测试 {i}: {test_case['description']}")
        logger.info(f"   问题: {test_case['question']}")
        try:
            payload = {
                "question": test_case['question'],
                "context_size": 3
            }
            response = requests.post(
                f"{base_url}/api/answer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                answer_data = response.json()
                if answer_data.get('status') == 'success':
                    logger.info(f"   ✅ 问答成功")
                    logger.info(f"   置信度: {answer_data.get('confidence')}")
                    logger.info(f"   答案: {answer_data.get('answer', '')[:200]}...")
                    sources = answer_data.get('sources', [])
                    logger.info(f"   参考来源: {len(sources)} 个文档片段")
                else:
                    logger.warning(f"   ⚠️  问答返回错误: {answer_data.get('error', '未知错误')}")
            else:
                logger.error(f"   ❌ 问答请求失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"   ❌ 问答请求异常: {e}")
    
    # 4. 测试统计信息
    logger.info("\n4. 测试统计信息...")
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=10)
        if response.status_code == 200:
            stats_data = response.json()
            if stats_data.get('status') == 'success':
                logger.info("✅ 统计信息获取成功")
                milvus_stats = stats_data.get('milvus_stats', {})
                logger.info(f"   数据库实体数: {milvus_stats.get('num_entities', 0)}")
                logger.info(f"   PDF文件数: {stats_data.get('pdf_files_count', 0)}")
                logger.info(f"   向量维度: {stats_data.get('config', {}).get('dimension', 0)}")
            else:
                logger.warning(f"⚠️  统计信息返回错误: {stats_data.get('error', '未知错误')}")
        else:
            logger.error(f"❌ 统计信息请求失败: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ 统计信息请求异常: {e}")
    
    logger.info("\n=== API功能测试完成 ===")
    return True

if __name__ == "__main__":
    logger.info("启动API测试脚本")
    logger.info("请确保API服务已启动 (python pdf_search_api.py)")
    logger.info("服务地址: http://localhost:5001")
    
    try:
        test_api_endpoints()
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出现异常: {e}")