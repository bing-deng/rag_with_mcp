#!/usr/bin/env python3
"""
简化版RAG健康检查 - 强制重新导入所有模块
"""
import sys
import os
import importlib

# 清除可能的模块缓存
modules_to_reload = ['weaviate_client', 'bedrock.bedrock_usage']
for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("🔧 シンプル診断開始...")
print(f"プロジェクトルート: {project_root}")

try:
    # 1. Bedrockサービステスト
    print("\n🧪 1. Bedrockサービステスト")
    from bedrock.bedrock_usage import TokyoBedrockService
    bedrock_service = TokyoBedrockService()
    print("✅ Bedrockサービス正常")
    
    # 2. Weaviateクライアントテスト
    print("\n🧪 2. Weaviateクライアントテスト")
    from weaviate_client import WeaviateRAGClient
    weaviate_client = WeaviateRAGClient()
    
    # 接続テスト
    if weaviate_client.wait_for_weaviate(timeout=5):
        print("✅ Weaviate接続成功")
    else:
        print("❌ Weaviate接続失敗")
        sys.exit(1)
    
    # 3. メソッド存在確認
    print("\n🧪 3. 必要メソッド確認")
    
    required_methods = [
        'add_documents_with_external_vectors',
        'semantic_search_with_external_vector'
    ]
    
    all_methods_exist = True
    for method_name in required_methods:
        if hasattr(weaviate_client, method_name):
            print(f"✅ {method_name}: 存在")
            # メソッドの詳細情報を表示
            method = getattr(weaviate_client, method_name)
            if callable(method):
                print(f"   └─ 実行可能: Yes")
            else:
                print(f"   └─ 実行可能: No")
        else:
            print(f"❌ {method_name}: 不存在")
            all_methods_exist = False
    
    # 4. 簡単な機能テスト
    if all_methods_exist:
        print("\n🧪 4. 基本機能テスト")
        
        # テスト用嵌入生成
        test_texts = ["これはテストです", "日本語の文書です"]
        embeddings = bedrock_service.get_embeddings(test_texts, input_type="search_document")
        
        if embeddings and len(embeddings) > 0:
            print(f"✅ 嵌入生成成功: {len(embeddings)}個, 次元: {len(embeddings[0])}")
            
            # Schema作成テスト
            test_collection = "SimpleHealthTest"
            if weaviate_client.create_schema(test_collection):
                print(f"✅ Schema作成成功: {test_collection}")
                
                # 外部ベクトルでの文書追加テスト
                test_docs = [
                    {"content": text, "title": f"テスト{i+1}"}
                    for i, text in enumerate(test_texts)
                ]
                
                try:
                    result = weaviate_client.add_documents_with_external_vectors(
                        documents=test_docs,
                        embeddings=embeddings,
                        class_name=test_collection
                    )
                    
                    if result:
                        print("✅ 外部ベクトル文書追加成功")
                        
                        # 検索テスト
                        query_embedding = bedrock_service.get_embeddings(
                            ["テスト検索"], input_type="search_query"
                        )[0]
                        
                        search_results = weaviate_client.semantic_search_with_external_vector(
                            query_vector=query_embedding,
                            class_name=test_collection,
                            limit=2
                        )
                        
                        if search_results:
                            print(f"✅ 外部ベクトル検索成功: {len(search_results)}件")
                            print("🎉 全システム正常動作確認!")
                        else:
                            print("❌ 外部ベクトル検索失敗")
                    else:
                        print("❌ 外部ベクトル文書追加失敗")
                except Exception as e:
                    print(f"❌ 機能テスト中エラー: {e}")
            else:
                print("❌ Schema作成失敗")
        else:
            print("❌ 嵌入生成失敗")
    else:
        print("❌ 必要メソッドが不足、機能テスト中止")

except Exception as e:
    print(f"❌ エラー発生: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("🏁 シンプル診断完了") 