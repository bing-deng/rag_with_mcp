#!/usr/bin/env python3
"""
直接Weaviate GraphQL测试 - 绕过Python Client问题
"""
import requests
import json
import sys
import os

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bedrock.bedrock_usage import TokyoBedrockService

def test_direct_weaviate():
    print("🔧 直接Weaviate测试开始...")
    
    weaviate_url = "http://localhost:8180"
    
    try:
        # 1. 初始化Bedrock服务
        print("\n🧪 1. Bedrock初期化")
        bedrock_service = TokyoBedrockService()
        
        # 2. 生成测试向量
        print("\n🧪 2. テストベクトル生成")
        test_texts = ["これはテストです", "日本語の文書です"]
        test_query = "テスト検索"
        
        doc_embeddings = bedrock_service.get_embeddings(test_texts, input_type="search_document")
        query_embedding = bedrock_service.get_embeddings([test_query], input_type="search_query")[0]
        
        if not doc_embeddings or not query_embedding:
            print("❌ 嵌入生成失敗")
            return
        
        print(f"✅ 嵌入生成成功: 文档{len(doc_embeddings)}個, 查询1個")
        
        # 3. 使用REST API直接创建collection
        print("\n🧪 3. 直接REST API創建Collection")
        
        collection_name = "DirectTest"
        
        # 删除已存在的collection
        delete_url = f"{weaviate_url}/v1/schema/{collection_name}"
        requests.delete(delete_url)
        
        # 创建新collection
        create_schema = {
            "class": collection_name,
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "title", "dataType": ["text"]},
                {"name": "source", "dataType": ["text"]}
            ],
            "vectorizer": "none",  # 关键：禁用向量化器
            "vectorIndexConfig": {
                "distance": "cosine"
            }
        }
        
        create_response = requests.post(
            f"{weaviate_url}/v1/schema",
            json=create_schema,
            headers={"Content-Type": "application/json"}
        )
        
        if create_response.status_code == 200:
            print("✅ Collection創建成功")
        else:
            print(f"❌ Collection創建失敗: {create_response.status_code} - {create_response.text}")
            return
        
        # 4. 使用REST API直接添加对象
        print("\n🧪 4. 直接REST API追加オブジェクト")
        
        for i, (text, embedding) in enumerate(zip(test_texts, doc_embeddings)):
            obj_data = {
                "class": collection_name,
                "properties": {
                    "content": text,
                    "title": f"テスト{i+1}",
                    "source": "直接テスト"
                },
                "vector": embedding  # 直接向量
            }
            
            obj_response = requests.post(
                f"{weaviate_url}/v1/objects",
                json=obj_data,
                headers={"Content-Type": "application/json"}
            )
            
            if obj_response.status_code == 200:
                print(f"✅ オブジェクト{i+1}追加成功")
            else:
                print(f"❌ オブジェクト{i+1}追加失敗: {obj_response.status_code} - {obj_response.text}")
        
        # 5. 等待索引
        print("\n🧪 5. インデックス待機...")
        import time
        time.sleep(3)
        
        # 6. 使用GraphQL进行搜索
        print("\n🧪 6. GraphQL検索テスト")
        
        # 构建GraphQL查询
        graphql_query = {
            "query": f"""
            {{
              Get {{
                {collection_name}(
                  nearVector: {{
                    vector: {json.dumps(query_embedding)}
                    certainty: 0.1
                  }}
                  limit: 5
                ) {{
                  content
                  title
                  source
                  _additional {{
                    certainty
                    distance
                    vector
                  }}
                }}
              }}
            }}
            """
        }
        
        graphql_response = requests.post(
            f"{weaviate_url}/v1/graphql",
            json=graphql_query,
            headers={"Content-Type": "application/json"}
        )
        
        if graphql_response.status_code == 200:
            result = graphql_response.json()
            print("✅ GraphQL検索成功")
            
            if 'data' in result and 'Get' in result['data']:
                objects = result['data']['Get'][collection_name]
                print(f"📊 検索結果: {len(objects)}件")
                
                for i, obj in enumerate(objects):
                    print(f"   結果{i+1}:")
                    print(f"     - タイトル: {obj.get('title', 'N/A')}")
                    print(f"     - コンテンツ: {obj.get('content', 'N/A')}")
                    if '_additional' in obj:
                        additional = obj['_additional']
                        print(f"     - 確実性: {additional.get('certainty', 'N/A')}")
                        print(f"     - 距離: {additional.get('distance', 'N/A')}")
                        
                if objects:
                    print("🎉 GraphQL検索で結果が見つかりました！")
                else:
                    print("⚠️ GraphQL検索でも結果が見つかりませんでした")
                    
                    # 尝试无向量条件的查询
                    print("🔍 条件なし検索テスト...")
                    simple_query = {
                        "query": f"""
                        {{
                          Get {{
                            {collection_name}(limit: 5) {{
                              content
                              title
                              _additional {{
                                vector
                              }}
                            }}
                          }}
                        }}
                        """
                    }
                    
                    simple_response = requests.post(
                        f"{weaviate_url}/v1/graphql",
                        json=simple_query,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if simple_response.status_code == 200:
                        simple_result = simple_response.json()
                        if 'data' in simple_result and 'Get' in simple_result['data']:
                            simple_objects = simple_result['data']['Get'][collection_name]
                            print(f"📦 全オブジェクト数: {len(simple_objects)}")
                            
                            for obj in simple_objects:
                                print(f"   - {obj.get('title', 'N/A')}: {obj.get('content', 'N/A')}")
                                if '_additional' in obj and 'vector' in obj['_additional']:
                                    vec = obj['_additional']['vector']
                                    if vec:
                                        print(f"     ベクトル次元: {len(vec)}")
                                    else:
                                        print(f"     ベクトル: なし")
            else:
                print(f"❌ GraphQL結果形式エラー: {result}")
        else:
            print(f"❌ GraphQL検索失敗: {graphql_response.status_code} - {graphql_response.text}")
        
        # 清理
        print("\n🧪 7. クリーンアップ")
        requests.delete(f"{weaviate_url}/v1/schema/{collection_name}")
        print("✅ テストCollection削除")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_weaviate() 