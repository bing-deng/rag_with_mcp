#!/usr/bin/env python3
"""
AWS Bedrock 东京区域（ap-northeast-1）Claude 测试脚本
使用APAC推理配置文件
"""

import boto3
import json
from botocore.exceptions import ClientError

def test_tokyo_bedrock():
    """测试东京区域的Bedrock Claude访问"""
    
    print("=== AWS Bedrock 东京区域测试 ===")
    
    # 初始化东京区域客户端
    region_name = 'ap-northeast-1'  # 东京区域
    
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        bedrock = boto3.client('bedrock', region_name=region_name)
        print(f"✅ Bedrock客户端初始化成功 - 区域: {region_name}")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return
    
    # 东京区域可用的APAC推理配置文件
    apac_inference_profiles = [
        # APAC Claude Sonnet 4 - 最新模型
        'apac.anthropic.claude-sonnet-4-20250514-v1:0',
        
        # APAC Claude 3.5 Sonnet v2
        'apac.anthropic.claude-3-5-sonnet-20241022-v2:0',
        
        # APAC Claude 3.5 Sonnet v1 
        'apac.anthropic.claude-3-5-sonnet-20240620-v1:0',
        
        # APAC Claude 3.7 Sonnet (如果可用)
        'apac.anthropic.claude-3-7-sonnet-20241125-v1:0',
        
        # APAC Claude 3 Sonnet (备用)
        'apac.anthropic.claude-3-sonnet-20240229-v1:0',
        
        # APAC Claude 3 Haiku (轻量级选项)
        'apac.anthropic.claude-3-haiku-20240307-v1:0',
    ]
    
    # 首先列出可用的推理配置文件
    print("\n1. 检查可用的推理配置文件...")
    try:
        response = bedrock.list_inference_profiles()
        profiles = response.get('inferenceProfileSummaries', [])
        
        claude_profiles = []
        for profile in profiles:
            if 'claude' in profile['inferenceProfileName'].lower():
                claude_profiles.append({
                    'id': profile['inferenceProfileId'],
                    'name': profile['inferenceProfileName'],
                    'status': profile['status'],
                    'type': profile.get('type', 'N/A')
                })
        
        if claude_profiles:
            print("✅ 找到Claude推理配置文件:")
            for profile in claude_profiles:
                print(f"   📋 ID: {profile['id']}")
                print(f"      名称: {profile['name']}")
                print(f"      状态: {profile['status']}")
                print(f"      类型: {profile['type']}")
                print("-" * 50)
        else:
            print("⚠️  未找到Claude推理配置文件，但将继续测试预定义ID")
            
    except Exception as e:
        print(f"❌ 获取推理配置文件时出错: {e}")
        print("   继续使用预定义的推理配置文件ID...")
    
    # 测试不同的推理配置文件
    print(f"\n2. 测试 {len(apac_inference_profiles)} 个APAC推理配置文件...")
    successful_model_id = None
    
    for i, model_id in enumerate(apac_inference_profiles, 1):
        print(f"\n{i}. 测试推理配置文件: {model_id}")
        
        try:
            # 构建测试请求
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 150,
                "messages": [
                    {
                        "role": "user",
                        "content": "你好！请用中文简单介绍一下你自己，并说明你是哪个版本的Claude。"
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # 发送请求
            response = bedrock_runtime.invoke_model(
                modelId=model_id,  # 使用APAC推理配置文件ID
                contentType='application/json',
                accept='application/json',
                body=json.dumps(body)
            )
            
            # 解析响应
            response_body = json.loads(response['body'].read())
            claude_response = response_body['content'][0]['text']
            
            print(f"   ✅ 成功！")
            print(f"   📝 Claude回复: {claude_response}")
            successful_model_id = model_id
            break
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"   ❌ 失败: {error_code}")
            
            if error_code == 'ValidationException':
                if 'model identifier' in error_message.lower():
                    print("      错误: 无效的模型标识符")
                elif 'on-demand throughput' in error_message.lower():
                    print("      错误: 需要使用推理配置文件")
                else:
                    print(f"      详情: {error_message}")
            elif error_code == 'AccessDeniedException':
                print(f"      错误: 访问被拒绝 - 请检查模型访问权限")
            elif error_code == 'ThrottlingException':
                print(f"      错误: 请求被限流 - {error_message}")
            else:
                print(f"      详情: {error_message}")
                
        except Exception as e:
            print(f"   ❌ 其他错误: {e}")
    
    # 测试嵌入向量（Amazon Titan Embed）
    print(f"\n{'='*60}")
    print("3. 测试嵌入向量（Amazon Titan）...")
    try:
        embed_body = {
            "inputText": "これは日本語のテストテキストです。This is a multilingual test.",
            "dimensions": 1024,
            "normalize": True
        }
        
        embed_response = bedrock_runtime.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps(embed_body)
        )
        
        embed_result = json.loads(embed_response['body'].read())
        embedding = embed_result['embedding']
        
        print(f"✅ 嵌入向量生成成功")
        print(f"   📐 向量维度: {len(embedding)}")
        print(f"   📊 前5个值: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ 嵌入向量生成失败: {e}")
    
    # 测试多语言嵌入
    print(f"\n4. 测试多语言嵌入向量...")
    multilingual_texts = [
        "人工智能正在改变世界",  # 中文
        "人工知能が世界を変えている",  # 日文
        "Artificial intelligence is changing the world",  # 英文
    ]
    
    try:
        embeddings = []
        for text in multilingual_texts:
            embed_body = {
                "inputText": text,
                "dimensions": 512,  # 使用较小维度以节省资源
                "normalize": True
            }
            
            response = bedrock_runtime.invoke_model(
                modelId='amazon.titan-embed-text-v2:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps(embed_body)
            )
            
            result = json.loads(response['body'].read())
            embeddings.append(result['embedding'])
        
        print("✅ 多语言嵌入向量生成成功")
        for i, text in enumerate(multilingual_texts):
            print(f"   📝 文本: {text}")
            print(f"   📐 向量维度: {len(embeddings[i])}")
        
        # 计算相似度
        import numpy as np
        
        # 中文和日文的相似度
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"   🤝 中文与日文相似度: {similarity:.4f}")
        
        # 中文和英文的相似度
        vec3 = np.array(embeddings[2])
        similarity2 = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        print(f"   🤝 中文与英文相似度: {similarity2:.4f}")
        
    except Exception as e:
        print(f"❌ 多语言嵌入向量测试失败: {e}")
    
    # 总结
    print(f"\n{'='*60}")
    if successful_model_id:
        print(f"🎉 测试完成！可用的Claude模型ID: {successful_model_id}")
        print(f"\n📋 在代码中使用以下配置:")
        print(f"   区域: {region_name}")
        print(f"   推理配置文件ID: {successful_model_id}")
        print(f"\n💡 代码示例:")
        print(f"   bedrock_runtime = boto3.client('bedrock-runtime', region_name='{region_name}')")
        print(f"   claude_model_id = '{successful_model_id}'")
    else:
        print("❌ 所有Claude推理配置文件都无法访问")
        print("\n🔍 故障排除建议:")
        print("1. 确保在AWS Bedrock控制台中申请了Claude模型的访问权限")
        print("2. 检查IAM权限是否包含 bedrock:InvokeModel")
        print("3. 确认你的AWS账户在ap-northeast-1区域有Bedrock访问权限")
        print("4. 检查AWS凭证配置是否正确")

def check_tokyo_model_access():
    """检查东京区域的模型访问权限"""
    print("=== 检查东京区域模型访问权限 ===")
    
    try:
        bedrock = boto3.client('bedrock', region_name='ap-northeast-1')
        
        # 列出基础模型
        response = bedrock.list_foundation_models(byProvider='Anthropic')
        models = response.get('modelSummaries', [])
        
        print("✅ 可用的Anthropic模型:")
        for model in models:
            print(f"   📋 模型ID: {model['modelId']}")
            print(f"      模型名称: {model['modelName']}")
            print(f"      输入类型: {model.get('inputModalities', [])}")
            print(f"      输出类型: {model.get('outputModalities', [])}")
            print("-" * 40)
        
        return len(models) > 0
        
    except Exception as e:
        print(f"❌ 检查模型访问权限失败: {e}")
        return False

if __name__ == "__main__":
    # 先检查模型访问权限
    if check_tokyo_model_access():
        print("\n" + "="*60)
        test_tokyo_bedrock()
    else:
        print("\n请先在AWS Bedrock控制台申请Anthropic模型访问权限")
        print("控制台地址: https://ap-northeast-1.console.aws.amazon.com/bedrock/home?region=ap-northeast-1#/modelaccess")