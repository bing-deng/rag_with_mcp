#!/usr/bin/env python3
"""
文本相似性搜索系统示例
使用 Milvus 构建一个实际的文本搜索引擎，支持语义搜索功能
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, 
    DataType, Collection, MilvusException
)

class TextSearchEngine:
    """基于 Milvus 的文本相似性搜索引擎"""
    
    def __init__(self, host='localhost', port='19530', collection_name='text_search'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.dimension = 384  # 使用 sentence-transformers 的默认维度
        
    def connect(self):
        """连接到 Milvus 服务器"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ 成功连接到 Milvus: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def create_collection(self):
        """创建文本搜索集合"""
        # 定义集合模式
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(fields, "文本相似性搜索集合")
        
        # 如果集合已存在，删除它
        if utility.has_collection(self.collection_name):
            print(f"📝 集合 {self.collection_name} 已存在，正在删除...")
            utility.drop_collection(self.collection_name)
        
        # 创建新集合
        self.collection = Collection(self.collection_name, schema)
        print(f"✅ 集合 {self.collection_name} 创建成功")
        
        return self.collection
    
    def create_index(self):
        """创建向量索引"""
        if not self.collection:
            raise ValueError("集合尚未创建")
        
        # 为不同规模的数据选择合适的索引
        index_params = {
            "metric_type": "COSINE",  # 使用余弦相似度，适合文本搜索
            "index_type": "HNSW",     # HNSW 索引，高精度
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }
        
        print("🔧 正在创建索引...")
        self.collection.create_index("embedding", index_params)
        print("✅ 索引创建完成")
    
    def load_collection(self):
        """加载集合到内存"""
        if not self.collection:
            raise ValueError("集合尚未创建")
        
        print("📥 正在加载集合到内存...")
        self.collection.load()
        print("✅ 集合加载完成")
    
    def simple_text_to_vector(self, text: str) -> List[float]:
        """
        简单的文本向量化方法（演示用）
        在实际应用中，应该使用 sentence-transformers 或其他预训练模型
        """
        # 这是一个非常简单的向量化方法，仅用于演示
        # 实际应用中应该使用 sentence-transformers 等库
        words = text.lower().split()
        vector = np.random.normal(0, 1, self.dimension).tolist()
        
        # 添加一些基于文本内容的特征
        if words:
            # 基于文本长度调整向量
            length_factor = min(len(words) / 10.0, 1.0)
            vector = [v * length_factor for v in vector]
        
        # 归一化向量（余弦相似度需要）
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def insert_documents(self, documents: List[Dict]) -> bool:
        """
        插入文档数据
        documents 格式: [{"title": "标题", "content": "内容", "category": "分类", "url": "链接"}]
        """
        if not self.collection:
            raise ValueError("集合尚未创建")
        
        print(f"📝 正在处理 {len(documents)} 个文档...")
        
        # 准备数据
        titles = []
        contents = []
        categories = []
        urls = []
        timestamps = []
        embeddings = []
        
        for doc in documents:
            # 组合标题和内容用于向量化
            full_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            
            titles.append(doc.get('title', ''))
            contents.append(doc.get('content', ''))
            categories.append(doc.get('category', 'general'))
            urls.append(doc.get('url', ''))
            timestamps.append(int(time.time()))
            embeddings.append(self.simple_text_to_vector(full_text))
        
        # 插入数据
        data = [titles, contents, categories, urls, timestamps, embeddings]
        
        try:
            insert_result = self.collection.insert(data)
            print(f"✅ 成功插入 {insert_result.insert_count} 条文档")
            
            # 强制刷新到磁盘
            self.collection.flush()
            return True
            
        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, category_filter: str = None) -> List[Dict]:
        """
        搜索相似文档
        """
        if not self.collection:
            raise ValueError("集合尚未创建")
        
        # 将查询转换为向量
        query_vector = self.simple_text_to_vector(query)
        
        # 设置搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}  # HNSW 搜索参数
        }
        
        # 构建过滤表达式
        expr = None
        if category_filter:
            expr = f'category == "{category_filter}"'
        
        try:
            # 执行搜索
            results = self.collection.search(
                [query_vector],
                "embedding",
                search_params,
                limit=top_k,
                expr=expr,
                output_fields=["title", "content", "category", "url", "timestamp"]
            )
            
            # 处理搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": float(hit.score),  # 相似度分数
                        "title": hit.entity.get("title"),
                        "content": hit.entity.get("content"),
                        "category": hit.entity.get("category"),
                        "url": hit.entity.get("url"),
                        "timestamp": hit.entity.get("timestamp")
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """获取集合统计信息"""
        if not self.collection:
            return {}
        
        stats = {
            "collection_name": self.collection.name,
            "total_documents": self.collection.num_entities,
            "description": self.collection.description,
            "dimension": self.dimension
        }
        
        return stats
    
    def delete_documents(self, expr: str) -> bool:
        """根据表达式删除文档"""
        try:
            self.collection.delete(expr)
            print(f"✅ 删除操作完成: {expr}")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        print("🔌 已断开连接")

def create_sample_documents() -> List[Dict]:
    """创建示例文档数据"""
    documents = [
        {
            "title": "Python 机器学习入门",
            "content": "Python 是一种强大的编程语言，特别适合机器学习和数据科学。本文介绍了如何使用 scikit-learn 进行基础的机器学习任务。",
            "category": "technology",
            "url": "https://example.com/python-ml-intro"
        },
        {
            "title": "深度学习与神经网络",
            "content": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂模式。TensorFlow 和 PyTorch 是最流行的深度学习框架。",
            "category": "technology",
            "url": "https://example.com/deep-learning"
        },
        {
            "title": "向量数据库的应用",
            "content": "向量数据库如 Milvus 专门用于存储和检索高维向量数据，在推荐系统、相似性搜索和 AI 应用中发挥重要作用。",
            "category": "technology",
            "url": "https://example.com/vector-database"
        },
        {
            "title": "自然语言处理技术",
            "content": "自然语言处理（NLP）结合了计算机科学和人工智能，帮助计算机理解和生成人类语言。BERT 和 GPT 是重要的 NLP 模型。",
            "category": "technology",
            "url": "https://example.com/nlp-tech"
        },
        {
            "title": "健康饮食指南",
            "content": "健康的饮食习惯对身体健康至关重要。建议多吃蔬菜水果，减少加工食品的摄入，保持营养均衡。",
            "category": "health",
            "url": "https://example.com/healthy-diet"
        },
        {
            "title": "运动与健身",
            "content": "规律的运动可以提高身体素质，增强免疫力。推荐每周至少进行 150 分钟的中等强度有氧运动。",
            "category": "health",
            "url": "https://example.com/exercise-fitness"
        },
        {
            "title": "旅行摄影技巧",
            "content": "旅行摄影需要掌握光线、构图和时机。黄金时段的光线最适合拍摄，同时要注意前景和背景的搭配。",
            "category": "photography",
            "url": "https://example.com/travel-photography"
        },
        {
            "title": "美食制作心得",
            "content": "烹饪是一门艺术，需要掌握火候、调味和食材搭配。新鲜的食材和适当的调料是制作美食的关键。",
            "category": "cooking",
            "url": "https://example.com/cooking-tips"
        }
    ]
    
    return documents

def main():
    """主函数 - 演示文本搜索引擎的完整功能"""
    print("🚀 启动文本相似性搜索引擎演示")
    print("=" * 50)
    
    # 创建搜索引擎实例
    search_engine = TextSearchEngine()
    
    try:
        # 1. 连接到 Milvus
        if not search_engine.connect():
            return
        
        # 2. 创建集合
        search_engine.create_collection()
        
        # 3. 创建索引
        search_engine.create_index()
        
        # 4. 加载集合
        search_engine.load_collection()
        
        # 5. 插入示例文档
        documents = create_sample_documents()
        search_engine.insert_documents(documents)
        
        # 6. 显示统计信息
        stats = search_engine.get_statistics()
        print(f"\n📊 集合统计信息:")
        print(f"   集合名称: {stats['collection_name']}")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   向量维度: {stats['dimension']}")
        
        # 7. 演示搜索功能
        print(f"\n🔍 搜索演示:")
        print("-" * 30)
        
        queries = [
            "机器学习算法",
            "健康生活方式",
            "摄影技术",
            "Python 编程"
        ]
        
        for query in queries:
            print(f"\n查询: '{query}'")
            results = search_engine.search(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. 【{result['category']}】{result['title']}")
                    print(f"      相似度: {result['score']:.4f}")
                    print(f"      内容: {result['content'][:60]}...")
            else:
                print("  未找到相关结果")
        
        # 8. 演示分类过滤搜索
        print(f"\n🏷️  分类过滤搜索演示:")
        print("-" * 30)
        
        query = "技术"
        category = "technology"
        print(f"在分类 '{category}' 中搜索: '{query}'")
        
        filtered_results = search_engine.search(query, top_k=5, category_filter=category)
        for i, result in enumerate(filtered_results, 1):
            print(f"  {i}. {result['title']} (分数: {result['score']:.4f})")
        
        # 9. 演示删除功能
        print(f"\n🗑️  删除演示:")
        print("-" * 30)
        
        # 删除特定分类的文档
        delete_expr = 'category == "cooking"'
        print(f"删除条件: {delete_expr}")
        search_engine.delete_documents(delete_expr)
        
        # 显示更新后的统计信息
        updated_stats = search_engine.get_statistics()
        print(f"删除后文档总数: {updated_stats['total_documents']}")
        
        print(f"\n✅ 演示完成!")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        
    finally:
        # 清理资源
        search_engine.disconnect()

if __name__ == "__main__":
    main() 