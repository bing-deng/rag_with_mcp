import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.config import Configure, Property, DataType
import json
import numpy as np
from typing import List, Dict, Any
import requests
import time

class WeaviateRAGClient:
    """最终修复版Weaviate RAG客户端 - 基于成功的REST API格式"""
    
    def __init__(self, weaviate_url="http://localhost:8180"):
        self.weaviate_url = weaviate_url
        self.client = None
        self._connect()
    
    def _connect(self):
        """连接Weaviate"""
        try:
            self.client = weaviate.connect_to_custom(
                http_host="localhost",
                http_port=8180,
                http_secure=False,
                grpc_host="localhost", 
                grpc_port=50051,
                grpc_secure=False
            )
            print(f"✅ 成功连接Weaviate: {self.weaviate_url}")
        except Exception as e:
            print(f"❌ Weaviate连接失败: {e}")
            self.client = None
    
    def wait_for_weaviate(self, timeout=30):
        """等待Weaviate启动"""
        import time
        for i in range(timeout):
            try:
                if self.client and self.client.is_ready():
                    print("✅ Weaviate已就绪!")
                    return True
                time.sleep(1)
                if i == 0:
                    print("等待Weaviate启动...")
            except:
                time.sleep(1)
        print("❌ Weaviate启动超时")
        return False
    
    def create_schema(self, class_name="Document"):
        """🎯 创建collection - 基于成功的REST API格式"""
        try:
            # 删除已存在的collection
            try:
                if self.client.collections.exists(class_name):
                    self.client.collections.delete(class_name)
                    print(f"🗑️ 删除已存在collection: {class_name}")
            except Exception as e:
                print(f"⚠️ 删除collection时出错: {e}")
            
            # 🎯 基于成功REST API的配置
            collection = self.client.collections.create(
                name=class_name,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="provider", data_type=DataType.TEXT),
                ],
                # 🎯 关键：模仿成功的REST API配置
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw(
                    distance_metric=weaviate.classes.config.VectorDistances.COSINE
                )
            )
            
            print(f"✅ 成功创建最终版collection: {class_name}")
            print("   - 基于成功的REST API配置")
            return True
            
        except Exception as e:
            print(f"❌ Schema创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_documents_with_external_vectors(self, documents: List[Dict[str, Any]], 
                                          embeddings: List[List[float]], 
                                          class_name="Document"):
        """🎯 使用外部向量添加文档 - 基于成功的REST API格式"""
        try:
            print(f"📝 添加文档与外部向量: {len(documents)} 个")
            
            if len(documents) != len(embeddings):
                raise ValueError(f"文档数量({len(documents)})与嵌入数量({len(embeddings)})不匹配")
            
            collection = self.client.collections.get(class_name)
            
            # 🎯 关键修复：基于成功REST API，直接使用向量数组而不是字典
            data_objects = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                metadata = doc.get("metadata", {})
                
                # 🎯 最关键修复：直接使用向量数组，不用字典包装
                data_obj = DataObject(
                    properties={
                        "content": doc.get("content", ""),
                        "title": doc.get("title", f"文档{i+1}"),
                        "source": doc.get("source", "未知"),
                        "category": metadata.get("category", "通用"),
                        "provider": metadata.get("provider", "未知")
                    },
                    vector=embedding  # 🎯 直接使用向量数组，就像REST API一样
                )
                
                data_objects.append(data_obj)
                
                if (i + 1) % 10 == 0:
                    print(f"⏳ 已准备 {i+1}/{len(documents)} 个文档")
            
            # 批量插入
            response = collection.data.insert_many(data_objects)
            
            if response.errors:
                print(f"⚠️ 插入过程中有 {len(response.errors)} 个错误")
                for error in response.errors[:3]:
                    print(f"   - {error}")
                return len(data_objects) - len(response.errors) > 0
            else:
                print(f"✅ 成功添加所有 {len(documents)} 个文档")
                return True
            
        except Exception as e:
            print(f"❌ 文档添加失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def semantic_search_with_external_vector(self, query_vector: List[float], 
                                            class_name="Document", limit=5) -> List[Dict]:
        """🎯 关键方法：使用外部查询向量进行语义搜索 - 基于成功的GraphQL格式"""
        try:
            print(f"🔍 执行外部向量语义搜索，向量维度: {len(query_vector)}")
            
            collection = self.client.collections.get(class_name)
            
            # 🎯 基于成功的GraphQL，直接使用向量数组
            response = collection.query.near_vector(
                near_vector=query_vector,  # 🎯 直接使用向量，不用字典包装
                limit=limit,
                certainty=0.1,  # 🎯 设置较低阈值，基于成功测试的经验
                return_metadata=["certainty", "distance"]
            )
            
            results = []
            for obj in response.objects:
                result = {
                    'content': obj.properties.get('content', ''),
                    'title': obj.properties.get('title', ''),
                    'source': obj.properties.get('source', ''),
                    'category': obj.properties.get('category', ''),
                    'provider': obj.properties.get('provider', ''),
                    'certainty': obj.metadata.certainty if obj.metadata else 0,
                    'distance': obj.metadata.distance if obj.metadata else 1.0
                }
                results.append(result)
            
            print(f"✅ 找到 {len(results)} 个相关文档")
            if results:
                print(f"   最佳匹配相似度: {results[0]['certainty']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ 外部向量搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def delete_collection(self, class_name="Document"):
        """删除collection"""
        try:
            if self.client.collections.exists(class_name):
                self.client.collections.delete(class_name)
                print(f"✅ 成功删除collection: {class_name}")
                return True
            else:
                print(f"⚠️ Collection不存在: {class_name}")
                return False
        except Exception as e:
            print(f"❌ 删除collection失败: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], class_name="Document"):
        """传统文档添加方法（使用内置vectorizer）- 保留向后兼容"""
        print("⚠️ 使用传统添加方法，需要内置vectorizer")
        return False  # 在外部向量模式下不支持
    
    def semantic_search(self, query: str, class_name="Document", limit=5):
        """传统语义搜索（使用内置vectorizer）- 保留向后兼容"""
        print("⚠️ 使用传统搜索方法，需要内置vectorizer")
        return []  # 在外部向量模式下不支持
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close() 