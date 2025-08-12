"""
结合AWS Bedrock和本地Weaviate的完整RAG系统
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bedrock'))

from bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient
import time

class BedrockWeaviateRAG:
    """结合AWS Bedrock和Weaviate的RAG系统"""
    
    def __init__(self):
        self.bedrock_service = TokyoBedrockService()
        self.weaviate_client = WeaviateRAGClient()
        
    def setup_knowledge_base(self, documents):
        """设置知识库"""
        print("🔧 设置知识库...")
        
        # 等待Weaviate就绪
        if not self.weaviate_client.wait_for_weaviate():
            return False
            
        # 创建schema
        if not self.weaviate_client.create_schema():
            return False
            
        # 添加文档
        if not self.weaviate_client.add_documents(documents):
            return False
            
        # 等待向量化
        print("⏳ 等待向量化完成...")
        time.sleep(5)
        
        return True
    
    def rag_query(self, question, top_k=3):
        """执行RAG查询"""
        print(f"\n🤖 RAG查询: {question}")
        print("="*60)
        
        # 步骤1: 使用Weaviate进行语义搜索
        print("1️⃣ 使用Weaviate检索相关文档...")
        relevant_docs = self.weaviate_client.semantic_search(question, limit=top_k)
        
        if not relevant_docs:
            print("❌ 未找到相关文档")
            return None
        
        print(f"✅ 找到 {len(relevant_docs)} 个相关文档")
        
        # 步骤2: 构建增强提示
        print("2️⃣ 构建增强提示...")
        context = "\n".join([
            f"【{doc['title']}】{doc['content']}" 
            for doc in relevant_docs
        ])
        
        enhanced_prompt = f"""请基于以下上下文信息回答用户问题：

上下文信息：
{context}

用户问题：{question}

请提供详细、准确的回答，并在适当时引用上下文中的信息。"""

        # 步骤3: 使用Claude生成回答
        print("3️⃣ 使用Claude生成回答...")
        rag_response = self.bedrock_service.chat_with_claude(
            message=enhanced_prompt,
            system_prompt="你是一个知识助手，请基于提供的上下文准确回答问题。",
            max_tokens=1000
        )
        
        # 返回完整结果
        return {
            "question": question,
            "retrieved_docs": relevant_docs,
            "context": context,
            "answer": rag_response
        }
    
    def display_result(self, result):
        """显示查询结果"""
        if not result:
            return
            
        print(f"\n{'='*80}")
        print("🎯 RAG查询结果")
        print(f"{'='*80}")
        
        print(f"❓ 问题: {result['question']}")
        
        print(f"\n📚 检索到的文档:")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            print(f"  {i}. 【{doc['title']}】(相似度: {doc['certainty']:.3f})")
            print(f"     {doc['content'][:100]}...")
        
        print(f"\n🤖 AI回答:")
        if result['answer']:
            print(result['answer'])
        else:
            print("回答生成失败")
        
        print(f"{'='*80}")

def main():
    """主函数演示"""
    print("=== AWS Bedrock + 本地Weaviate RAG演示 ===\n")
    
    # 初始化RAG系统
    rag_system = BedrockWeaviateRAG()
    
    # 准备知识库文档
    knowledge_docs = [
        {
            "title": "AWS Bedrock服务概述",
            "content": "AWS Bedrock是亚马逊提供的完全托管的生成式AI服务，支持Claude、Cohere、Titan等多种基础模型。在东京地区(ap-northeast-1)可用，提供API访问和推理配置文件。",
            "source": "aws_bedrock_docs",
            "metadata": {"service": "bedrock", "region": "ap-northeast-1"}
        },
        {
            "title": "Claude 4模型特性",
            "content": "Claude 4是Anthropic最新的大型语言模型，具有更强的推理能力、更好的安全性和更长的上下文处理能力。支持复杂的分析任务和创意写作。",
            "source": "anthropic_docs", 
            "metadata": {"model": "claude-4", "provider": "anthropic"}
        },
        {
            "title": "Weaviate向量数据库",
            "content": "Weaviate是开源的向量数据库，支持语义搜索、自动向量化和GraphQL查询。特别适合RAG应用，提供高性能的相似度搜索功能。",
            "source": "weaviate_docs",
            "metadata": {"database": "weaviate", "type": "vector"}
        },
        {
            "title": "RAG系统架构设计",
            "content": "检索增强生成(RAG)系统包含文档存储、向量化、检索和生成四个核心组件。通过结合外部知识库，能显著提高AI回答的准确性和时效性。",
            "source": "rag_guide",
            "metadata": {"topic": "rag", "type": "architecture"}
        }
    ]
    
    # 设置知识库
    if not rag_system.setup_knowledge_base(knowledge_docs):
        print("❌ 知识库设置失败")
        return
    
    # 测试查询
    test_questions = [
        "AWS Bedrock在东京地区支持哪些模型？",
        "Claude 4有什么新特性？",
        "如何设计一个RAG系统？",
        "Weaviate相比其他向量数据库有什么优势？"
    ]
    
    print(f"✅ 知识库设置完成，开始测试查询...\n")
    
    for question in test_questions:
        result = rag_system.rag_query(question)
        rag_system.display_result(result)
        print("\n" + "="*40 + "\n")
    
    print("🎉 RAG演示完成!")
    print("💡 您可以通过修改 test_questions 来测试其他问题")

if __name__ == "__main__":
    main()
