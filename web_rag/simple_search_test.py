#!/usr/bin/env python3
"""
簡単な検索テスト - メソッド調整版
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient

def test_search_methods():
    print("🔧 検索メソッドテスト開始...")
    
    try:
        # サービス初期化
        bedrock_service = TokyoBedrockService()
        weaviate_client = WeaviateRAGClient()
        
        test_query = "電圧調査について"
        
        # 1. 利用可能メソッドの確認
        print("📋 利用可能メソッド:")
        methods = [m for m in dir(weaviate_client) if 'search' in m and not m.startswith('_')]
        for method in methods:
            print(f"   - {method}")
        
        # 2. 外部ベクトル検索テスト
        if hasattr(weaviate_client, 'semantic_search_with_external_vector'):
            print(f"\n🧪 外部ベクトル検索テスト: '{test_query}'")
            try:
                results = weaviate_client.semantic_search_with_external_vector(
                    test_query, bedrock_service.get_embeddings, top_k=5
                )
                print(f"✅ 外部ベクトル検索成功: {len(results)}件")
                
                for i, result in enumerate(results[:3], 1):
                    similarity = result.get('similarity', result.get('certainty', 0))
                    content_preview = result.get('content', '')[:100] + "..."
                    print(f"   結果{i}: 相似度={similarity:.3f}")
                    print(f"           内容: {content_preview}")
                
            except Exception as e:
                print(f"❌ 外部ベクトル検索エラー: {str(e)}")
        
        # 3. 基本検索テスト
        if hasattr(weaviate_client, 'semantic_search'):
            print(f"\n🧪 基本検索テスト: '{test_query}'")
            try:
                results = weaviate_client.semantic_search(test_query, limit=5)
                print(f"✅ 基本検索成功: {len(results)}件")
                
                for i, result in enumerate(results[:3], 1):
                    similarity = result.get('similarity', result.get('certainty', 0))
                    content_preview = result.get('content', '')[:100] + "..."
                    print(f"   結果{i}: 相似度={similarity:.3f}")
                    print(f"           内容: {content_preview}")
                
            except Exception as e:
                print(f"❌ 基本検索エラー: {str(e)}")
        
        # 4. 直接オブジェクト検索（デバッグ用）
        print(f"\n🧪 保存オブジェクト確認...")
        try:
            if hasattr(weaviate_client.client, 'collections'):
                collection = weaviate_client.client.collections.get("Document")
                objects = collection.query.fetch_objects(limit=3)
                print(f"✅ 保存オブジェクト: {len(objects.objects)}個")
                
                for i, obj in enumerate(objects.objects[:2], 1):
                    content = obj.properties.get('content', '')[:100] + "..."
                    print(f"   オブジェクト{i}: {content}")
            
        except Exception as e:
            print(f"❌ オブジェクト確認エラー: {str(e)}")
        
    except Exception as e:
        print(f"❌ テスト初期化エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search_methods() 