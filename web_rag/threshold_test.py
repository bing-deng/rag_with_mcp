#!/usr/bin/env python3
"""
相似度阈值测试 - 使用修复后的Client
"""
import sys
import os
import time

# パス設定  
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient  # 使用修复后的客户端
from weaviate.classes.config import Property, DataType
from weaviate.classes.data import DataObject

def test_final_fix():
    print("🔧 最终修复测试开始...")
    
    try:
        # 初始化
        bedrock_service = TokyoBedrockService()
        client = WeaviateRAGClient()
        
        if not client.wait_for_weaviate():
            print("❌ Weaviate未就绪")
            return
        
        # 准备测试数据
        test_texts = ["これはテストです", "日本語の文書です"]
        query_text = "テスト検索"
        
        # 生成嵌入
        embeddings = bedrock_service.get_embeddings(test_texts, input_type="search_document")
        query_embedding = bedrock_service.get_embeddings([query_text], input_type="search_query")[0]
        
        # 创建collection
        test_collection = "FinalTest"
        if not client.create_schema(test_collection):
            print("❌ Collection创建失败")
            return
        
        # 添加文档
        test_docs = [
            {"content": text, "title": f"テスト{i+1}", "source": "最終テスト"}
            for i, text in enumerate(test_texts)
        ]
        
        if not client.add_documents_with_external_vectors(test_docs, embeddings, test_collection):
            print("❌ 文档添加失败")
            return
        
        time.sleep(3)  # 等待索引
        
        # 搜索测试
        print("🔍 最終検索テスト...")
        results = client.semantic_search_with_external_vector(
            query_embedding, 
            test_collection,
            limit=5
        )
        
        if results:
            print(f"🎉 最終修復成功！找到 {len(results)} 个结果")
            for i, result in enumerate(results):
                print(f"   結果{i+1}:")
                print(f"     - タイトル: {result['title']}")
                print(f"     - 相似度: {result['certainty']:.4f}")
                print(f"     - 距離: {result['distance']:.4f}")
        else:
            print("❌ 最終修復でも検索結果なし")
        
        # 清理
        client.client.collections.delete(test_collection)
        client.close()
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_fix()
    if success:
        print("\n✅ 最終修復完了！RAGシステム準備完了。")
    else:
        print("\n❌ まだ問題が残っています。") 