#!/usr/bin/env python3
"""
深度调试RAG搜索问题
"""
import sys
import os

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_debug_client import WeaviateDebugClient
from weaviate.classes.data import DataObject
from weaviate.classes.config import Property, DataType  # 🎯 正确的导入
import time

def main():
    print("🔧 深度調試開始...")
    
    try:
        # 1. 初始化服务
        print("\n🧪 1. サービス初期化")
        bedrock_service = TokyoBedrockService()
        debug_client = WeaviateDebugClient()
        
        if not debug_client.client or not debug_client.client.is_ready():
            print("❌ Weaviate未就绪")
            return
        
        print("✅ サービス初期化完成")
        
        # 2. 准备测试数据
        print("\n🧪 2. テストデータ準備")
        test_texts = ["これはテストです", "日本語の文書です"] 
        test_query = "テスト検索"
        
        # 生成嵌入
        embeddings = bedrock_service.get_embeddings(test_texts, input_type="search_document")
        query_embedding = bedrock_service.get_embeddings([test_query], input_type="search_query")[0]
        
        if not embeddings or not query_embedding:
            print("❌ 嵌入生成失敗")
            return
            
        print(f"✅ 嵌入生成成功: 文档{len(embeddings)}個, 查询1個")
        
        # 3. 创建并调试collection
        test_collection = "DebugTest"
        
        print(f"\n🧪 3. Collection作成: {test_collection}")
        if not debug_client.client.collections.exists(test_collection):
            # 🎯 使用正确的API语法
            result = debug_client.client.collections.create(
                name=test_collection,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),  # 正确的语法
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                ]
            )
            print("✅ Collection作成成功")
        else:
            print("✅ Collection既存")
        
        # 4. 添加文档
        print(f"\n🧪 4. 文書追加")
        collection = debug_client.client.collections.get(test_collection)
        
        test_docs = [
            {"content": text, "title": f"テスト{i+1}", "source": "調試"}
            for i, text in enumerate(test_texts)
        ]
        
        # 使用DataObject添加
        data_objects = []
        for doc, embedding in zip(test_docs, embeddings):
            data_obj = DataObject(
                properties={
                    "content": doc["content"],
                    "title": doc["title"], 
                    "source": doc["source"]
                },
                vector=embedding
            )
            data_objects.append(data_obj)
        
        response = collection.data.insert_many(data_objects)
        
        if response.errors:
            print(f"⚠️ 插入中有錯誤: {len(response.errors)}")
            for error in response.errors[:3]:
                print(f"   - {error}")
        else:
            print("✅ 文書追加成功")
        
        # 5. 等待索引
        print("\n🧪 5. インデックス待機")
        time.sleep(3)  # 等待3秒让索引完成
        
        # 6. 调试collection状态
        print("\n🧪 6. Collection状態確認")
        debug_client.debug_collection_status(test_collection)
        
        # 7. 调试搜索
        print("\n🧪 7. 検索デバッグ")
        results = debug_client.debug_search_with_vector(
            query_embedding, 
            test_collection, 
            limit=5
        )
        
        if results:
            print(f"🎉 検索成功! {len(results)}件の結果")
            for i, result in enumerate(results):
                print(f"   結果{i+1}: {result['title']} (相似度: {result['certainty']:.3f})")
        else:
            print("❌ 検索結果なし")
            
        # 清理
        try:
            debug_client.client.collections.delete(test_collection)
            print(f"✅ テストCollection削除: {test_collection}")
        except:
            pass
            
        debug_client.close()
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 