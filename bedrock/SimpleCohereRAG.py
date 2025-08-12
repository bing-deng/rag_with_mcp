#!/usr/bin/env python3
"""
Cohere最小化测试 - 诊断"0"错误
"""

import boto3
import json
import traceback

def minimal_cohere_test():
    """最简单的Cohere测试"""
    
    print("=== Cohere最小化诊断测试 ===")
    
    try:
        # 初始化客户端
        region_name = 'ap-northeast-1'
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        model_id = 'cohere.embed-multilingual-v3'
        
        print(f"✅ 客户端初始化成功")
        print(f"   区域: {region_name}")
        print(f"   模型: {model_id}")
        
        # 最简单的请求
        test_text = "Hello"
        
        body = {
            "texts": [test_text],
            "input_type": "search_document",
            "embedding_types": ["float"]
        }
        
        print(f"\n📤 发送请求:")
        print(f"   文本: {test_text}")
        print(f"   请求体: {json.dumps(body, indent=2)}")
        
        # 发送请求
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='*/*',
            body=json.dumps(body)
        )
        
        print(f"✅ API调用成功")
        print(f"   响应状态码: {response['ResponseMetadata']['HTTPStatusCode']}")
        
        # 读取响应
        response_body = response['body'].read()
        print(f"   响应体长度: {len(response_body)} 字节")
        
        # 解析JSON
        result = json.loads(response_body)
        print(f"✅ JSON解析成功")
        
        # 分析响应结构
        print(f"\n📋 响应结构分析:")
        for key, value in result.items():
            print(f"   {key}: {type(value)}")
            
            if key == 'embeddings':
                if isinstance(value, list):
                    print(f"      列表长度: {len(value)}")
                    if len(value) > 0:
                        first_embedding = value[0]
                        print(f"      第一个元素类型: {type(first_embedding)}")
                        if isinstance(first_embedding, list):
                            print(f"      向量维度: {len(first_embedding)}")
                            print(f"      向量前3个值: {first_embedding[:3]}")
                        else:
                            print(f"      第一个元素内容: {first_embedding}")
                else:
                    print(f"      不是列表，内容: {value}")
            elif isinstance(value, (str, int, float, bool)):
                print(f"      值: {value}")
            elif isinstance(value, list):
                print(f"      列表，长度: {len(value)}")
                if len(value) > 0:
                    print(f"      第一个元素: {value[0]}")
        
        # 尝试提取嵌入向量
        print(f"\n🎯 提取嵌入向量测试:")
        
        if 'embeddings' in result:
            embeddings = result['embeddings']
            print(f"   embeddings存在: ✅")
            print(f"   embeddings类型: {type(embeddings)}")
            
            if isinstance(embeddings, list):
                print(f"   embeddings是列表: ✅")
                print(f"   列表长度: {len(embeddings)}")
                
                if len(embeddings) > 0:
                    print(f"   列表不为空: ✅")
                    first_item = embeddings[0]
                    print(f"   第一个元素类型: {type(first_item)}")
                    
                    if first_item is not None:
                        print(f"   第一个元素不为None: ✅")
                        if isinstance(first_item, list):
                            print(f"   第一个元素是列表: ✅")
                            print(f"   🎉 成功提取嵌入向量！维度: {len(first_item)}")
                            return first_item
                        else:
                            print(f"   ❌ 第一个元素不是列表: {first_item}")
                    else:
                        print(f"   ❌ 第一个元素为None")
                else:
                    print(f"   ❌ embeddings列表为空")
            else:
                print(f"   ❌ embeddings不是列表")
        else:
            print(f"   ❌ 响应中没有embeddings字段")
        
        print(f"\n📄 完整响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return None
        
    except Exception as e:
        print(f"\n❌ 发生异常:")
        print(f"   异常类型: {type(e).__name__}")
        print(f"   异常信息: {str(e)}")
        print(f"\n📋 详细堆栈跟踪:")
        traceback.print_exc()
        return None

def test_different_parameters():
    """测试不同的参数组合"""
    
    print(f"\n{'='*60}")
    print("=== 不同参数组合测试 ===")
    
    region_name = 'ap-northeast-1'
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
    model_id = 'cohere.embed-multilingual-v3'
    
    # 不同的参数组合
    test_cases = [
        {
            "name": "最简参数",
            "body": {
                "texts": ["Hello"],
                "input_type": "search_document"
            }
        },
        {
            "name": "添加embedding_types",
            "body": {
                "texts": ["Hello"],
                "input_type": "search_document",
                "embedding_types": ["float"]
            }
        },
        {
            "name": "添加truncate",
            "body": {
                "texts": ["Hello"],
                "input_type": "search_document",
                "embedding_types": ["float"],
                "truncate": "END"
            }
        },
        {
            "name": "中文文本",
            "body": {
                "texts": ["你好"],
                "input_type": "search_document",
                "embedding_types": ["float"]
            }
        },
        {
            "name": "不同input_type",
            "body": {
                "texts": ["Hello"],
                "input_type": "search_query",
                "embedding_types": ["float"]
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试: {test_case['name']}")
        print(f"   参数: {json.dumps(test_case['body'], ensure_ascii=False)}")
        
        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType='application/json',
                accept='*/*',
                body=json.dumps(test_case['body'])
            )
            
            result = json.loads(response['body'].read())
            
            if 'embeddings' in result and result['embeddings']:
                embeddings = result['embeddings']
                if len(embeddings) > 0 and embeddings[0] is not None:
                    print(f"   ✅ 成功! 向量维度: {len(embeddings[0])}")
                else:
                    print(f"   ❌ embeddings为空或None")
            else:
                print(f"   ❌ 没有embeddings字段或为空")
                print(f"   响应键: {list(result.keys())}")
                
        except Exception as e:
            print(f"   ❌ 失败: {e}")

if __name__ == "__main__":
    # 运行最小化测试
    embedding = minimal_cohere_test()
    
    if embedding:
        print(f"\n🎉 测试成功! 嵌入向量已获取，维度: {len(embedding)}")
    else:
        print(f"\n❌ 测试失败，继续诊断...")
        
    # 测试不同参数
    test_different_parameters()