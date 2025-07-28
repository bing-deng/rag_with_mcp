#!/usr/bin/env python3
"""
Milvus 向量数据库使用示例
演示了 Milvus 的基本操作，包括连接、集合管理、数据插入和向量搜索
"""

import random
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# 配置参数
HOST = 'localhost'
PORT = '19530'
COLLECTION_NAME = 'demo_collection'
DIMENSION = 128  # 向量维度

def connect_to_milvus():
    """连接到 Milvus 服务器"""
    print("正在连接到 Milvus...")
    connections.connect("default", host=HOST, port=PORT)
    print(f"成功连接到 Milvus 服务器: {HOST}:{PORT}")

def create_collection():
    """创建集合"""
    print(f"创建集合: {COLLECTION_NAME}")
    
    # 定义字段模式
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    
    # 创建集合模式
    schema = CollectionSchema(fields, "这是一个演示集合")
    
    # 检查集合是否已存在
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 已存在，正在删除...")
        utility.drop_collection(COLLECTION_NAME)
    
    # 创建集合
    collection = Collection(COLLECTION_NAME, schema)
    print(f"集合 {COLLECTION_NAME} 创建成功")
    return collection

def generate_sample_data(num_entities=1000):
    """生成示例数据"""
    print(f"生成 {num_entities} 条示例数据...")
    
    # 生成随机向量数据
    embeddings = [[random.random() for _ in range(DIMENSION)] for _ in range(num_entities)]
    
    # 生成文本数据
    texts = [f"这是第 {i} 条示例文本数据" for i in range(num_entities)]
    
    # 生成 ID
    ids = list(range(num_entities))
    
    data = [ids, texts, embeddings]
    print("示例数据生成完成")
    return data

def insert_data(collection, data):
    """插入数据到集合"""
    print("正在插入数据...")
    insert_result = collection.insert(data)
    print(f"插入数据完成，插入了 {insert_result.insert_count} 条记录")
    return insert_result

def create_index(collection):
    """创建索引"""
    print("正在创建索引...")
    
    index_params = {
        "metric_type": "L2",  # 距离度量类型
        "index_type": "IVF_FLAT",  # 索引类型
        "params": {"nlist": 128}  # 索引参数
    }
    
    collection.create_index("embedding", index_params)
    print("索引创建完成")

def load_collection(collection):
    """加载集合到内存"""
    print("正在加载集合到内存...")
    collection.load()
    print("集合加载完成")

def search_vectors(collection, num_queries=5, top_k=10):
    """搜索相似向量"""
    print(f"执行向量搜索，查询 {num_queries} 个向量，返回前 {top_k} 个最相似的结果...")
    
    # 生成查询向量
    query_vectors = [[random.random() for _ in range(DIMENSION)] for _ in range(num_queries)]
    
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    # 执行搜索
    results = collection.search(
        query_vectors,
        "embedding",
        search_params,
        limit=top_k,
        output_fields=["id", "text"]
    )
    
    # 打印搜索结果
    for i, result in enumerate(results):
        print(f"\n查询向量 {i+1} 的搜索结果:")
        for j, hit in enumerate(result):
            print(f"  排名 {j+1}: ID={hit.id}, 距离={hit.distance:.4f}, 文本='{hit.entity.get('text')[:50]}...'")
    
    return results

def get_collection_stats(collection):
    """获取集合统计信息"""
    print("\n集合统计信息:")
    print(f"  集合名称: {collection.name}")
    print(f"  记录数量: {collection.num_entities}")
    print(f"  集合描述: {collection.description}")
    
    # 获取集合模式信息
    print("  字段信息:")
    for field in collection.schema.fields:
        print(f"    - {field.name}: {field.dtype}")

def query_data(collection, filter_expr="id in [0, 1, 2, 3, 4]"):
    """查询特定数据"""
    print(f"\n执行条件查询: {filter_expr}")
    
    results = collection.query(
        expr=filter_expr,
        output_fields=["id", "text", "embedding"]
    )
    
    print(f"查询到 {len(results)} 条记录:")
    for result in results[:3]:  # 只显示前3条
        print(f"  ID: {result['id']}, 文本: {result['text']}")
    
    return results

def delete_data(collection, delete_expr="id in [0, 1, 2]"):
    """删除数据"""
    print(f"\n删除数据: {delete_expr}")
    
    collection.delete(delete_expr)
    print("数据删除完成")

def main():
    """主函数 - 演示 Milvus 的完整使用流程"""
    try:
        # 1. 连接到 Milvus
        connect_to_milvus()
        
        # 2. 创建集合
        collection = create_collection()
        
        # 3. 生成并插入示例数据
        data = generate_sample_data(1000)
        insert_result = insert_data(collection, data)
        
        # 4. 创建索引
        create_index(collection)
        
        # 5. 加载集合
        load_collection(collection)
        
        # 6. 获取统计信息
        get_collection_stats(collection)
        
        # 7. 执行向量搜索
        search_results = search_vectors(collection, num_queries=3, top_k=5)
        
        # 8. 条件查询
        query_results = query_data(collection)
        
        # 9. 删除部分数据
        delete_data(collection)
        
        # 10. 再次查看统计信息
        print("\n删除数据后的统计信息:")
        get_collection_stats(collection)
        
        print("\n=== Milvus 演示完成 ===")
        
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        # 清理资源
        if 'collection' in locals():
            collection.release()
        connections.disconnect("default")
        print("连接已断开")

if __name__ == "__main__":
    main()
