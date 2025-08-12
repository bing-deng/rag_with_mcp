import boto3
import json
from botocore.exceptions import ClientError
import numpy as np
from typing import List, Dict, Any

class TokyoBedrockService:
    """专门针对东京地区(ap-northeast-1)的AWS Bedrock服务类"""
    
    def __init__(self):
        """
        初始化东京地区的Bedrock客户端
        """
        self.region_name = 'ap-northeast-1'
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=self.region_name
        )
        
        # 使用Claude 4的推理配置文件ID（APAC区域）
        self.claude_model_id = 'apac.anthropic.claude-sonnet-4-20250514-v1:0'
        self.cohere_embed_model_id = 'cohere.embed-multilingual-v3'
        
        print(f"初始化服务 - 区域: {self.region_name}")
        print(f"Claude 4模型: {self.claude_model_id}")
        print(f"Cohere嵌入模型: {self.cohere_embed_model_id}")
    
    def get_embeddings(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        获取Cohere嵌入向量
        """
        try:
            body = {
                "texts": texts,
                "input_type": input_type,
                "embedding_types": ["float"],
                "truncate": "END"
            }
            
            print(f"请求嵌入 - 文本数量: {len(texts)}, input_type: {input_type}")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.cohere_embed_model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            print(f"API响应结构: {list(response_body.keys())}")
            
            # 调试：打印响应格式
            if 'embeddings' in response_body:
                embeddings_data = response_body['embeddings']
                print(f"embeddings类型: {type(embeddings_data)}")
                print(f"embeddings内容: {list(embeddings_data.keys()) if isinstance(embeddings_data, dict) else 'not a dict'}")
                
                # 根据实际的Cohere API响应格式处理
                embeddings = []
                
                if isinstance(embeddings_data, dict):
                    # Cohere API返回格式: {"embeddings": {"float": [[...], [...]]}}
                    if 'float' in embeddings_data:
                        float_embeddings = embeddings_data['float']
                        print(f"float embeddings类型: {type(float_embeddings)}, 数量: {len(float_embeddings)}")
                        if isinstance(float_embeddings, list):
                            embeddings = float_embeddings
                        else:
                            print(f"意外的float embeddings格式: {type(float_embeddings)}")
                            return []
                    else:
                        print(f"embeddings字典中没有'float'键，可用键: {list(embeddings_data.keys())}")
                        # 尝试直接使用embeddings数据
                        if len(embeddings_data) > 0:
                            first_key = list(embeddings_data.keys())[0]
                            print(f"尝试使用第一个键: {first_key}")
                            potential_embeddings = embeddings_data[first_key]
                            if isinstance(potential_embeddings, list) and len(potential_embeddings) > 0:
                                if isinstance(potential_embeddings[0], list):
                                    embeddings = potential_embeddings
                                else:
                                    embeddings = [potential_embeddings]
                            else:
                                return []
                        else:
                            return []
                elif isinstance(embeddings_data, list):
                    # 如果embeddings本身就是list
                    embeddings = embeddings_data
                else:
                    print(f"未知的embeddings格式: {type(embeddings_data)}")
                    return []
                
                print(f"成功解析 {len(embeddings)} 个嵌入向量")
                if embeddings and len(embeddings) > 0:
                    print(f"第一个嵌入向量维度: {len(embeddings[0])}")
                
                return embeddings
            else:
                print("响应中没有'embeddings'字段")
                return []
                
        except ClientError as e:
            print(f"获取嵌入向量时出错: {e}")
            return []
        except Exception as e:
            print(f"处理响应时出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    def semantic_search(self, query: str, documents: List[str], top_k: int = 3) -> List[Dict]:
        """
        语义搜索
        """
        print(f"\n=== 语义搜索 ===")
        print(f"查询: {query}")
        print(f"文档数量: {len(documents)}")
        
        # 获取查询嵌入
        query_embeddings = self.get_embeddings([query], input_type="search_query")
        if not query_embeddings or len(query_embeddings) == 0:
            print("无法获取查询的嵌入向量")
            return []
        
        query_embedding = query_embeddings[0]
        print(f"查询嵌入维度: {len(query_embedding)}")
        
        # 获取文档嵌入
        doc_embeddings = self.get_embeddings(documents, input_type="search_document")
        if not doc_embeddings or len(doc_embeddings) != len(documents):
            print("无法获取文档的嵌入向量")
            return []
        
        print(f"文档嵌入数量: {len(doc_embeddings)}")
        
        # 计算余弦相似度
        similarities = []
        query_vec = np.array(query_embedding)
        
        for i, doc_embedding in enumerate(doc_embeddings):
            if doc_embedding and len(doc_embedding) > 0:
                doc_vec = np.array(doc_embedding)
                # 余弦相似度计算
                cosine_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                similarities.append({
                    'document': documents[i],
                    'similarity': float(cosine_sim),
                    'index': i
                })
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        result = similarities[:top_k]
        
        print(f"找到 {len(result)} 个相关文档:")
        for i, doc in enumerate(result, 1):
            print(f"{i}. 相似度: {doc['similarity']:.3f} - {doc['document'][:50]}...")
        
        return result
    
    def chat_with_claude(self, message: str, system_prompt: str = "", max_tokens: int = 1000) -> str:
        """
        与Claude对话
        """
        try:
            messages = [{"role": "user", "content": message}]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            print(f"调用Claude - 消息长度: {len(message)}")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.claude_model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                answer = response_body['content'][0]['text']
                print(f"Claude回答长度: {len(answer)}")
                return answer
            else:
                print("Claude响应格式异常")
                return None
                
        except ClientError as e:
            print(f"调用Claude时出错: {e}")
            return None
        except Exception as e:
            print(f"处理Claude响应时出错: {e}")
            return None

def tokyo_rag_demo():
    """东京地区RAG演示"""
    print("=== 东京地区RAG演示 ===\n")
    
    try:
        # 初始化服务
        service = TokyoBedrockService()
        
        # 知识库
        knowledge_base = [
            "AWS Bedrock是亚马逊提供的完全托管的AI服务，支持多种基础模型的API访问，包括Claude、Cohere、Titan等模型。",
            "Claude是Anthropic开发的大型语言模型，具有出色的推理能力和安全性，支持长文本对话。",
            "Cohere提供多语言嵌入模型，能够将文本转换为向量表示，支持语义搜索和相似度计算。",
            "RAG（检索增强生成）技术结合了信息检索和文本生成，先检索相关文档，再生成基于上下文的回答。",
            "向量数据库用于存储和检索高维向量，是RAG系统的核心组件，能快速找到语义相似的文档。",
            "AWS在东京地区(ap-northeast-1)提供Bedrock服务，支持Claude和Cohere等多种AI模型。"
        ]
        
        # 用户问题
        user_query = "AWS Bedrock在东京地区支持哪些AI模型？"
        
        print(f"用户问题: {user_query}\n")
        
        # 步骤1: 语义搜索
        relevant_docs = service.semantic_search(user_query, knowledge_base, top_k=3)
        
        if not relevant_docs:
            print("语义搜索失败，无法继续RAG流程")
            return
        
        # 步骤2: 构建增强提示
        context = "\n".join([f"- {doc['document']}" for doc in relevant_docs])
        
        enhanced_prompt = f"""请基于以下上下文信息回答用户问题：

上下文信息：
{context}

用户问题：{user_query}

请提供详细、准确的回答，并说明AWS Bedrock在东京地区的具体服务能力。"""

        print(f"\n=== 增强提示 ===")
        print(f"上下文长度: {len(context)} 字符")
        
        # 步骤3: Claude生成回答
        print(f"\n=== Claude生成回答 ===")
        rag_response = service.chat_with_claude(
            message=enhanced_prompt,
            system_prompt="你是AWS服务专家，请基于提供的上下文准确回答关于AWS Bedrock的问题。",
            max_tokens=1000
        )
        
        # 展示结果
        print(f"\n{'='*60}")
        print("最终RAG结果")
        print(f"{'='*60}")
        print(f"用户问题: {user_query}")
        print(f"\n检索到的相关文档:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"{i}. [相似度: {doc['similarity']:.3f}] {doc['document']}")
        
        print(f"\nRAG生成的回答:")
        if rag_response:
            print(rag_response)
        else:
            print("Claude回答生成失败")
        
        print(f"\n{'='*60}")
        
    except Exception as e:
        print(f"RAG演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

def test_components_separately():
    """分别测试各个组件"""
    print("=== 分别测试各个组件 ===\n")
    
    try:
        service = TokyoBedrockService()
        
        # 测试1: Cohere嵌入
        print("1. 测试Cohere嵌入...")
        test_texts = ["这是一个测试文本", "AWS Bedrock服务"]
        embeddings = service.get_embeddings(test_texts, input_type="search_query")
        
        if embeddings and len(embeddings) == 2:
            print(f"✓ 嵌入测试成功 - 维度: {len(embeddings[0])}")
        else:
            print("✗ 嵌入测试失败")
            return
        
        # 测试2: Claude对话
        print("\n2. 测试Claude对话...")
        test_message = "请简单介绍一下AWS Bedrock服务。"
        response = service.chat_with_claude(test_message, max_tokens=200)
        
        if response:
            print(f"✓ Claude测试成功")
            print(f"回答: {response[:100]}...")
        else:
            print("✗ Claude测试失败")
            return
        
        # 测试3: 语义搜索
        print("\n3. 测试语义搜索...")
        test_docs = [
            "AWS Bedrock是AI服务平台",
            "Claude是对话AI模型", 
            "机器学习需要大量数据"
        ]
        query = "什么是AWS AI服务？"
        results = service.semantic_search(query, test_docs, top_k=2)
        
        if results:
            print(f"✓ 语义搜索测试成功")
        else:
            print("✗ 语义搜索测试失败")
        
        print("\n所有组件测试完成！可以运行完整RAG演示。")
        
    except Exception as e:
        print(f"组件测试出错: {e}")
        import traceback
        traceback.print_exc()

# 同时添加一个专门的调试函数
def debug_cohere_response():
    """调试Cohere API响应格式"""
    print("=== 调试Cohere API响应 ===\n")
    
    try:
        service = TokyoBedrockService()
        
        # 发送一个简单的请求来查看响应格式
        body = {
            "texts": ["测试文本"],
            "input_type": "search_query",
            "embedding_types": ["float"],
            "truncate": "END"
        }
        
        response = service.bedrock_runtime.invoke_model(
            modelId=service.cohere_embed_model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        
        print("完整API响应:")
        print(json.dumps(response_body, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"调试过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("东京地区AWS Bedrock RAG演示")
    print("请确保已设置AWS凭证环境变量:")
    print("export AWS_ACCESS_KEY_ID=your_key")
    print("export AWS_SECRET_ACCESS_KEY=your_secret")
    print("export AWS_DEFAULT_REGION=ap-northeast-1")
    print("\n" + "="*50 + "\n")
    
    # 选择运行模式
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_components_separately()
        elif sys.argv[1] == "debug":
            debug_cohere_response()
    else:
        tokyo_rag_demo()

# # 查看完整的API响应格式
# python bedrock_usage.py debug

# # 测试各个组件
# python bedrock_usage.py test

# # 运行完整演示
# python bedrock_usage.py

# python bedrock_usage.py 

# 东京地区AWS Bedrock RAG演示
# 请确保已设置AWS凭证环境变量:
# export AWS_ACCESS_KEY_ID=your_key
# export AWS_SECRET_ACCESS_KEY=your_secret
# export AWS_DEFAULT_REGION=ap-northeast-1

# ==================================================

# === 东京地区RAG演示 ===

# 初始化服务 - 区域: ap-northeast-1
# Claude 4模型: apac.anthropic.claude-sonnet-4-20250514-v1:0
# Cohere嵌入模型: cohere.embed-multilingual-v3
# 用户问题: AWS Bedrock在东京地区支持哪些AI模型？


# === 语义搜索 ===
# 查询: AWS Bedrock在东京地区支持哪些AI模型？
# 文档数量: 6
# 请求嵌入 - 文本数量: 1, input_type: search_query
# API响应结构: ['id', 'texts', 'embeddings', 'response_type']
# embeddings类型: <class 'dict'>
# embeddings内容: ['float']
# float embeddings类型: <class 'list'>, 数量: 1
# 成功解析 1 个嵌入向量
# 第一个嵌入向量维度: 1024
# 查询嵌入维度: 1024
# 请求嵌入 - 文本数量: 6, input_type: search_document
# API响应结构: ['id', 'texts', 'embeddings', 'response_type']
# embeddings类型: <class 'dict'>
# embeddings内容: ['float']
# float embeddings类型: <class 'list'>, 数量: 6
# 成功解析 6 个嵌入向量
# 第一个嵌入向量维度: 1024
# 文档嵌入数量: 6
# 找到 3 个相关文档:
# 1. 相似度: 0.848 - AWS在东京地区(ap-northeast-1)提供Bedrock服务，支持Claude和Coher...
# 2. 相似度: 0.761 - AWS Bedrock是亚马逊提供的完全托管的AI服务，支持多种基础模型的API访问，包括Claud...
# 3. 相似度: 0.500 - Claude是Anthropic开发的大型语言模型，具有出色的推理能力和安全性，支持长文本对话。...

# === 增强提示 ===
# 上下文长度: 183 字符

# === Claude生成回答 ===
# 调用Claude - 消息长度: 282
# Claude回答长度: 466

# ============================================================
# 最终RAG结果
# ============================================================
# 用户问题: AWS Bedrock在东京地区支持哪些AI模型？

# 检索到的相关文档:
# 1. [相似度: 0.848] AWS在东京地区(ap-northeast-1)提供Bedrock服务，支持Claude和Cohere等多种AI模型。
# 2. [相似度: 0.761] AWS Bedrock是亚马逊提供的完全托管的AI服务，支持多种基础模型的API访问，包括Claude、Cohere、Titan等模型。
# 3. [相似度: 0.500] Claude是Anthropic开发的大型语言模型，具有出色的推理能力和安全性，支持长文本对话。

# RAG生成的回答:
# 根据提供的上下文信息，AWS Bedrock在东京地区(ap-northeast-1)支持的AI模型包括：

# ## 支持的AI模型

# 1. **Claude模型**
#    - 由Anthropic开发的大型语言模型
#    - 具有出色的推理能力和安全性特性
#    - 支持长文本对话处理

# 2. **Cohere模型**
#    - 专业的自然语言处理模型

# 3. **Titan模型**
#    - 亚马逊自研的基础模型

# ## AWS Bedrock在东京地区的服务能力

# AWS Bedrock作为完全托管的AI服务，在东京地区提供以下核心能力：

# - **多模型支持**：通过统一的API接口访问多种基础模型
# - **完全托管**：无需管理底层基础设施，专注于应用开发
# - **API访问**：提供标准化的API接口，便于集成到应用程序中
# - **区域化部署**：在东京地区本地化部署，确保低延迟和数据合规性

# 这使得开发者可以在东京地区便捷地使用多种先进的AI模型，构建各类智能应用，同时享受AWS云服务的可靠性和安全性保障。        