import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.config import Configure, Property, DataType
import json
import numpy as np
from typing import List, Dict, Any
import time

class WeaviateDebugClient:
    """调试版Weaviate客户端 - 修复向量访问API"""
    
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
    
    def debug_collection_status(self, class_name="Document"):
        """🔍 调试：检查collection状态 - 修复向量访问"""
        try:
            collection = self.client.collections.get(class_name)
            
            # 获取collection信息
            config = collection.config.get()
            print(f"📊 Collection配置:")
            print(f"   - 名称: {config.name}")
            print(f"   - 向量配置: {config.vector_config}")
            
            # 🎯 修复：正确获取带向量的对象
            try:
                response = collection.query.fetch_objects(
                    limit=10,
                    include_vector=True  # 明确请求包含向量
                )
                print(f"📦 存储的对象数量: {len(response.objects)}")
                
                for i, obj in enumerate(response.objects):
                    print(f"   对象 {i+1}:")
                    print(f"     - 标题: {obj.properties.get('title', 'N/A')}")
                    print(f"     - 内容长度: {len(str(obj.properties.get('content', '')))}")
                    
                    # 🎯 修复：正确访问向量数据
                    try:
                        if hasattr(obj, 'vector') and obj.vector:
                            if isinstance(obj.vector, dict):
                                # Named vectors格式
                                if 'default' in obj.vector:
                                    vector_data = obj.vector['default']
                                    print(f"     - 向量维度: {len(vector_data)} (named)")
                                    print(f"     - 向量前3个值: {vector_data[:3]}")
                                else:
                                    print(f"     - 向量键: {list(obj.vector.keys())}")
                            elif isinstance(obj.vector, list):
                                # 直接向量格式
                                print(f"     - 向量维度: {len(obj.vector)} (direct)")
                                print(f"     - 向量前3个值: {obj.vector[:3]}")
                            else:
                                print(f"     - 向量类型: {type(obj.vector)}")
                        else:
                            print(f"     - 向量: 无向量数据")
                    except Exception as ve:
                        print(f"     - 向量访问错误: {ve}")
                        
            except Exception as e:
                print(f"❌ 获取对象时出错: {e}")
                
            return True
            
        except Exception as e:
            print(f"❌ 调试collection状态失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def debug_search_with_vector(self, query_vector: List[float], 
                                class_name="Document", limit=5):
        """🔍 调试：详细的向量搜索 - 修复搜索API"""
        try:
            print(f"🔍 调试搜索开始")
            print(f"   - 查询向量维度: {len(query_vector)}")
            print(f"   - 查询向量前3个值: {query_vector[:3]}")
            
            collection = self.client.collections.get(class_name)
            
            # 首先验证collection存在且有数据
            all_objects = collection.query.fetch_objects(limit=1)
            if not all_objects.objects:
                print("⚠️ Collection中没有对象!")
                return []
            else:
                print(f"✅ Collection中有 {len(all_objects.objects)} 个对象")
            
            # 🎯 尝试修复后的搜索格式
            search_formats = [
                ("format_v4_correct", query_vector),     # Weaviate 4.x默认格式
                ("format_named_dict", {"default": query_vector}),  # Named vector尝试
                ("format_list_wrap", [query_vector]),    # 列表包装
            ]
            
            for format_name, vector_param in search_formats:
                try:
                    print(f"🔍 尝试搜索格式 {format_name}...")
                    
                    # 🎯 根据格式调整API调用
                    if format_name == "format_v4_correct":
                        response = collection.query.near_vector(
                            near_vector=vector_param,
                            limit=limit,
                            return_metadata=["certainty", "distance"]
                        )
                    elif format_name == "format_named_dict":
                        response = collection.query.near_vector(
                            near_vector=vector_param,
                            limit=limit,
                            return_metadata=["certainty", "distance"]
                        )
                    elif format_name == "format_list_wrap":
                        response = collection.query.near_vector(
                            near_vector=vector_param[0],  # 取出实际向量
                            limit=limit,
                            return_metadata=["certainty", "distance"]
                        )
                    
                    print(f"   - {format_name} 返回结果数: {len(response.objects)}")
                    
                    if response.objects:
                        print(f"✅ {format_name} 搜索成功!")
                        results = []
                        for i, obj in enumerate(response.objects):
                            print(f"     结果 {i+1}:")
                            if obj.metadata:
                                print(f"       - 相似度: {obj.metadata.certainty}")
                                print(f"       - 距离: {obj.metadata.distance}")
                            print(f"       - 标题: {obj.properties.get('title', 'N/A')}")
                            
                            result = {
                                'content': obj.properties.get('content', ''),
                                'title': obj.properties.get('title', ''),
                                'source': obj.properties.get('source', ''),
                                'certainty': obj.metadata.certainty if obj.metadata else 0,
                                'distance': obj.metadata.distance if obj.metadata else 1.0
                            }
                            results.append(result)
                        
                        return results  # 返回第一个成功的格式结果
                        
                except Exception as e:
                    print(f"   - {format_name} 失败: {str(e)[:100]}...")
                    continue
            
            # 🎯 如果常规搜索都失败，尝试更底层的方法
            print("🔍 尝试使用低阈值搜索...")
            try:
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=["certainty", "distance"],
                    certainty=0.1  # 设置很低的阈值
                )
                if response.objects:
                    print(f"✅ 低阈值搜索成功! 找到 {len(response.objects)} 个结果")
                    # 处理结果...
                    return [{"content": "测试成功", "title": "低阈值搜索", "certainty": 0.1}]
            except Exception as e:
                print(f"   - 低阈值搜索也失败: {e}")
            
            return []
            
        except Exception as e:
            print(f"❌ 调试搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            similarity = dot_product / (norm1 * norm2)
            return similarity
        except:
            return 0.0
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close() 