import weaviate
import json
import numpy as np
from typing import List, Dict, Any
import requests
import time

class WeaviateRAGClient:
    """本地Weaviate RAG客户端"""
    
    def __init__(self, weaviate_url="http://localhost:8180"):
        self.weaviate_url = weaviate_url
        self.client = None
        self._connect()
    
    def _connect(self):
        """连接到Weaviate实例"""
        try:
            self.client = weaviate.connect_to_custom(
                http_host="localhost",
                http_port=8180,
                http_secure=False,
                grpc_host="localhost", 
                grpc_port=50051,
                grpc_secure=False
            )
            
            # 检查连接
            if self.client.is_ready():
                print(f"✅ 成功连接到Weaviate: {self.weaviate_url}")
                return True
            else:
                print("❌ Weaviate未就绪")
                return False
                
        except Exception as e:
            print(f"❌ 连接Weaviate失败: {e}")
            return False
    
    def wait_for_weaviate(self, timeout=120):
        """等待Weaviate启动"""
        print("等待Weaviate启动...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.weaviate_url}/v1/.well-known/ready", timeout=5)
                if response.status_code == 200:
                    print("✅ Weaviate已就绪!")
                    self._connect()
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print("⏳ 等待中...")
            time.sleep(3)
        
        print(f"❌ {timeout}秒后Weaviate仍未就绪")
        return False
    
    def create_schema(self, class_name="Document"):
        """创建文档类的schema - 使用新版本API"""
        try:
            # 删除已存在的collection
            try:
                if self.client.collections.exists(class_name):
                    self.client.collections.delete(class_name)
                    print(f"🗑️  删除已存在的集合: {class_name}")
            except:
                pass
            
            # 使用新版本API创建collection
            from weaviate.classes.config import Configure, Property, DataType
            
            collection = self.client.collections.create(
                name=class_name,
                description="存储文档和文本的集合",
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="文档内容",
                        vectorize_property_name=False
                    ),
                    Property(
                        name="title", 
                        data_type=DataType.TEXT,
                        description="文档标题",
                        skip_vectorization=True
                    ),
                    Property(
                        name="source",
                        data_type=DataType.TEXT,
                        description="文档来源",
                        skip_vectorization=True
                    ),
                    Property(
                        name="category",
                        data_type=DataType.TEXT,
                        description="文档分类",
                        skip_vectorization=True
                    ),
                    Property(
                        name="provider",
                        data_type=DataType.TEXT,
                        description="服务提供商",
                        skip_vectorization=True
                    )
                ]
            )
            
            print(f"✅ 成功创建集合: {class_name}")
            return True
            
        except Exception as e:
            print(f"❌ 创建集合失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], class_name="Document"):
        """批量添加文档 - 使用新版本API"""
        try:
            print(f"📝 开始添加 {len(documents)} 个文档...")
            
            collection = self.client.collections.get(class_name)
            
            # 准备数据对象
            objects_to_insert = []
            for i, doc in enumerate(documents):
                # 从metadata中提取字段
                metadata = doc.get("metadata", {})
                
                data_object = {
                    "content": doc.get("content", ""),
                    "title": doc.get("title", f"Document {i+1}"),
                    "source": doc.get("source", "unknown"),
                    "category": metadata.get("category", "general"),
                    "provider": metadata.get("provider", "unknown")
                }
                
                objects_to_insert.append(data_object)
                print(f"⏳ 准备文档 {i+1}/{len(documents)}: {data_object['title']}")
            
            # 批量插入
            response = collection.data.insert_many(objects_to_insert)
            
            if response.errors:
                print(f"⚠️  插入过程中有一些错误: {len(response.errors)} 个")
                for error in response.errors[:3]:  # 只显示前3个错误
                    print(f"   - {error}")
            else:
                print(f"✅ 成功添加所有 {len(documents)} 个文档")
            
            return True
            
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def semantic_search(self, query: str, class_name="Document", limit=5) -> List[Dict]:
        """语义搜索 - 使用新版本API"""
        try:
            print(f"🔍 搜索查询: {query}")
            
            collection = self.client.collections.get(class_name)
            
            response = collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=["certainty", "distance"]
            )
            
            documents = []
            for item in response.objects:
                documents.append({
                    "content": item.properties["content"],
                    "title": item.properties["title"],
                    "source": item.properties["source"],
                    "category": item.properties.get("category", ""),
                    "provider": item.properties.get("provider", ""),
                    "certainty": item.metadata.certainty,
                    "distance": item.metadata.distance
                })
            
            print(f"✅ 找到 {len(documents)} 个相关文档")
            return documents
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_stats(self, class_name="Document"):
        """获取统计信息 - 使用新版本API"""
        try:
            collection = self.client.collections.get(class_name)
            
            # 获取文档数量
            response = collection.aggregate.over_all(total_count=True)
            count = response.total_count
            
            print(f"📊 {class_name} 集合包含 {count} 个文档")
            return count
            
        except Exception as e:
            print(f"❌ 获取统计失败: {e}")
            return 0
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()

def demo_weaviate_rag():
    """演示Weaviate RAG功能"""
    print("=== Weaviate本地RAG演示 (v4.x API) ===\n")
    
    # 初始化客户端
    client = WeaviateRAGClient()
    
    try:
        # 等待Weaviate启动
        if not client.wait_for_weaviate():
            print("❌ 无法连接到Weaviate，请确保Docker Compose已启动")
            return
        
        # 创建schema
        if not client.create_schema():
            print("❌ Schema创建失败")
            return
        
        # 准备示例文档
        sample_docs = [
            {
                "title": "AWS Bedrock概述",
                "content": "AWS Bedrock是亚马逊提供的完全托管的AI服务，支持多种基础模型的API访问，包括Claude、Cohere、Titan等模型。它在东京地区提供服务。",
                "source": "aws_docs",
                "metadata": {"category": "aws", "provider": "amazon"}
            },
            {
                "title": "Claude模型介绍", 
                "content": "Claude是Anthropic开发的大型语言模型，以安全、有用和诚实著称。Claude具有出色的推理能力和安全性，支持长文本对话。",
                "source": "anthropic_docs",
                "metadata": {"category": "llm", "provider": "anthropic"}
            },
            {
                "title": "Cohere嵌入模型",
                "content": "Cohere提供多语言嵌入模型，能够将文本转换为向量表示，支持语义搜索和相似度计算。适用于RAG系统的文档检索。",
                "source": "cohere_docs", 
                "metadata": {"category": "embedding", "provider": "cohere"}
            },
            {
                "title": "RAG技术原理",
                "content": "RAG（检索增强生成）技术结合了信息检索和文本生成，先检索相关文档，再基于这些信息生成回答。是提高AI准确性的重要技术。",
                "source": "tech_blog",
                "metadata": {"category": "technology", "provider": "general"}
            },
            {
                "title": "Weaviate向量数据库",
                "content": "Weaviate是开源的向量数据库，支持语义搜索、自动向量化和GraphQL查询。特别适合RAG应用，提供高性能的相似度搜索功能。",
                "source": "weaviate_docs",
                "metadata": {"category": "database", "provider": "weaviate"}
            }
        ]
        
        # 添加文档
        if not client.add_documents(sample_docs):
            print("❌ 文档添加失败")
            return
        
        # 等待向量化完成
        print("⏳ 等待向量化完成...")
        time.sleep(10)
        
        # 获取统计信息
        client.get_stats()
        
        # 执行搜索测试
        test_queries = [
            "AWS在东京地区提供什么AI服务？",
            "什么是RAG技术？",
            "向量数据库有什么用？"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"查询: {query}")
            print('='*50)
            
            results = client.semantic_search(query, limit=3)
            
            if results:
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. 【{doc['title']}】")
                    print(f"   内容: {doc['content']}")
                    print(f"   相似度: {doc['certainty']:.3f}")
                    print(f"   来源: {doc['source']}")
                    print(f"   分类: {doc['category']}")
            else:
                print("未找到相关文档")
        
        print(f"\n{'='*60}")
        print("本地Weaviate RAG演示完成!")
        print("您可以通过 http://localhost:8180 访问Weaviate控制台")
        print(f"{'='*60}")
    
    finally:
        # 确保关闭连接
        client.close()

if __name__ == "__main__":
    demo_weaviate_rag() 