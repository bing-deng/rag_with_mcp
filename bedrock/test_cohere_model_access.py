#!/usr/bin/env python3
"""
AWS Bedrock Cohere模型访问测试
验证Cohere Embed Multilingual模型的可用性
"""

import boto3
import json
from botocore.exceptions import ClientError

def test_cohere_models():
    """测试Cohere模型访问"""
    
    print("=== AWS Bedrock Cohere模型测试 ===")
    
    # 初始化客户端
    region_name = 'ap-northeast-1'  # 东京区域
    
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        bedrock = boto3.client('bedrock', region_name=region_name)
        print(f"✅ Bedrock客户端初始化成功 - 区域: {region_name}")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return
    
    # 1. 列出Cohere模型
    print("\n1. 检查可用的Cohere模型...")
    try:
        response = bedrock.list_foundation_models(byProvider='Cohere')
        models = response.get('modelSummaries', [])
        
        if models:
            print("✅ 找到Cohere模型:")
            for model in models:
                print(f"   📋 模型ID: {model['modelId']}")
                print(f"      模型名称: {model['modelName']}")
                print(f"      输入模态: {model.get('inputModalities', [])}")
                print(f"      输出模态: {model.get('outputModalities', [])}")
                print("-" * 50)
        else:
            print("❌ 未找到Cohere模型")
            
    except Exception as e:
        print(f"❌ 检查Cohere模型时出错: {e}")
    
    # 2. 测试Cohere Embed模型
    cohere_embed_models = [
        'cohere.embed-multilingual-v3',  # 多语言版本
        'cohere.embed-english-v3',       # 英语版本
    ]
    
    successful_models = []
    
    for model_id in cohere_embed_models:
        print(f"\n2. 测试模型: {model_id}")
        
        try:
            # 测试文本
            test_texts = [
                "这是一个中文测试文本",
                "This is an English test text"
            ]
            
            # 构建请求体
            body = {
                "texts": test_texts,
                "input_type": "search_document",
                "embedding_types": ["float"],
                "truncate": "END"
            }
            
            # 调用模型
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType='application/json',
                accept='*/*',
                body=json.dumps(body)
            )
            
            # 解析响应
            response_body = json.loads(response['body'].read())
            embeddings = response_body['embeddings']
            
            print(f"   ✅ 成功!")
            print(f"   📊 生成了 {len(embeddings)} 个嵌入向量")
            print(f"   📐 向量维度: {len(embeddings[0])}")
            print(f"   📝 处理的文本: {test_texts}")
            
            successful_models.append(model_id)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"   ❌ 失败: {error_code}")
            
            if error_code == 'ValidationException':
                print(f"      错误: 验证失败 - {error_message}")
            elif error_code == 'AccessDeniedException':
                print(f"      错误: 访问被拒绝 - 需要申请模型访问权限")
            elif error_code == 'ResourceNotFoundException':
                print(f"      错误: 模型不存在 - {error_message}")
            else:
                print(f"      详情: {error_message}")
                
        except Exception as e:
            print(f"   ❌ 其他错误: {e}")
    
    # 3. 测试语义搜索功能
    if successful_models:
        print(f"\n3. 测试语义搜索功能（使用 {successful_models[0]}）...")
        
        try:
            # 查询文本
            query = "机器学习"
            
            # 文档库
            documents = [
                "人工智能是一门计算机科学",
                "机器学习是人工智能的一个分支",
                "深度学习使用神经网络",
                "今天天气很好",
                "我喜欢吃苹果"
            ]
            
            # 获取查询嵌入向量
            query_body = {
                "texts": [query],
                "input_type": "search_query",
                "embedding_types": ["float"],
                "truncate": "END"
            }
            
            query_response = bedrock_runtime.invoke_model(
                modelId=successful_models[0],
                contentType='application/json',
                accept='*/*',
                body=json.dumps(query_body)
            )
            
            query_result = json.loads(query_response['body'].read())
            query_embedding = query_result['embeddings'][0]
            
            # 获取文档嵌入向量
            doc_body = {
                "texts": documents,
                "input_type": "search_document",
                "embedding_types": ["float"],
                "truncate": "END"
            }
            
            doc_response = bedrock_runtime.invoke_model(
                modelId=successful_models[0],
                contentType='application/json',
                accept='*/*',
                body=json.dumps(doc_body)
            )
            
            doc_result = json.loads(doc_response['body'].read())
            doc_embeddings = doc_result['embeddings']
            
            # 计算相似度
            import numpy as np
            
            query_vec = np.array(query_embedding)
            similarities = []
            
            for i, doc_embedding in enumerate(doc_embeddings):
                doc_vec = np.array(doc_embedding)
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                )
                similarities.append({
                    'index': i,
                    'document': documents[i],
                    'similarity': float(similarity)
                })
            
            # 排序
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            print("   ✅ 语义搜索成功!")
            print(f"   🔍 查询: {query}")
            print("   📋 最相关的文档:")
            
            for i, result in enumerate(similarities[:3], 1):
                print(f"      {i}. {result['document']}")
                print(f"         相似度: {result['similarity']:.4f}")
            
        except Exception as e:
            print(f"   ❌ 语义搜索测试失败: {e}")
    
    # 总结
    print(f"\n{'='*60}")
    if successful_models:
        print(f"🎉 测试完成！可用的Cohere模型:")
        for model in successful_models:
            print(f"   ✅ {model}")
        
        print(f"\n📋 使用建议:")
        print(f"   - 多语言应用使用: cohere.embed-multilingual-v3")
        print(f"   - 纯英语应用使用: cohere.embed-english-v3")
        print(f"   - 向量维度: 1024")
        print(f"   - 支持语言: 100+ (多语言版本)")
        
    else:
        print("❌ 所有Cohere模型都无法访问")
        print("\n🔍 故障排除建议:")
        print("1. 在AWS Bedrock控制台申请Cohere模型访问权限")
        print("2. 检查IAM权限是否包含 bedrock:InvokeModel")
        print("3. 确认当前区域支持Cohere模型")
        print("4. 检查AWS凭证配置是否正确")

def check_model_access_status():
    """检查模型访问状态"""
    print("=== 检查模型访问状态 ===")
    
    try:
        bedrock = boto3.client('bedrock', region_name='ap-northeast-1')
        
        # 尝试获取模型访问状态（这个API可能需要特殊权限）
        try:
            response = bedrock.get_model_invocation_logging_configuration()
            print("✅ 可以访问模型调用日志配置")
        except:
            print("⚠️  无法访问模型调用日志配置（可能是权限问题）")
        
        # 检查基础模型列表
        response = bedrock.list_foundation_models()
        total_models = len(response.get('modelSummaries', []))
        
        cohere_models = [m for m in response['modelSummaries'] if m['providerName'] == 'Cohere']
        anthropic_models = [m for m in response['modelSummaries'] if m['providerName'] == 'Anthropic']
        amazon_models = [m for m in response['modelSummaries'] if m['providerName'] == 'Amazon']
        
        print(f"✅ 模型统计:")
        print(f"   总模型数: {total_models}")
        print(f"   Cohere模型: {len(cohere_models)}")
        print(f"   Anthropic模型: {len(anthropic_models)}")
        print(f"   Amazon模型: {len(amazon_models)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查模型访问状态失败: {e}")
        return False

if __name__ == "__main__":
    if check_model_access_status():
        print("\n" + "="*60)
        test_cohere_models()
    else:
        print("\n请检查AWS配置和权限设置")