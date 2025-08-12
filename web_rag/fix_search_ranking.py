#!/usr/bin/env python3
"""
修复搜索排序问题 - 确保正确答案被找到
"""
import sys
import os

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient
from pdf_processor import PDFProcessor

def fix_search_ranking():
    print("🔧 検索ランキング修正テスト開始...")
    
    try:
        # 1. 重新处理PDF，只存储電圧調査相关的块
        print("\n🔍 1. 電圧調査関連チャンクのみを保存...")
        
        pdf_path = "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
        processor = PDFProcessor()
        chunks = processor.process_pdf(pdf_path)
        
        # 只选择電圧調査相关块
        voltage_chunks = []
        search_terms = ['電圧調査', '電柱番号', '不具合状況', '発生時間帯', '発生範囲']
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            found_terms = []
            for term in search_terms:
                if term in content:
                    found_terms.append(term)
            
            if found_terms:
                chunk['found_terms'] = found_terms
                chunk['is_complete'] = len(found_terms) >= 4  # 是否包含4个或更多关键信息
                voltage_chunks.append((i, chunk))
        
        print(f"📊 電圧調査関連チャンク: {len(voltage_chunks)}個")
        
        # 找到包含完整信息的块
        complete_chunks = [chunk for _, chunk in voltage_chunks if chunk.get('is_complete', False)]
        print(f"📊 完整情報包含チャンク: {len(complete_chunks)}個")
        
        if not complete_chunks:
            print("❌ 完整情報を含むチャンクが見つかりません")
            return
        
        # 2. 创建专门的collection，只存储這些块
        bedrock_service = TokyoBedrockService()
        weaviate_client = WeaviateRAGClient()
        
        if not weaviate_client.wait_for_weaviate():
            print("❌ Weaviate接続失敗")
            return
        
        test_collection = "VoltageInvestigationOnly"
        
        # 删除已存在的collection
        try:
            if weaviate_client.client.collections.exists(test_collection):
                weaviate_client.client.collections.delete(test_collection)
        except:
            pass
        
        if not weaviate_client.create_schema(test_collection):
            print("❌ Collection作成失敗")
            return
        
        print("✅ 専用Collection作成成功")
        
        # 3. 只存储電圧調査相关的块
        voltage_chunks_only = [chunk for _, chunk in voltage_chunks]
        contents = [chunk.get('content', '')[:1000] for chunk in voltage_chunks_only]
        embeddings = bedrock_service.get_embeddings(contents, input_type="search_document")
        
        if embeddings and len(embeddings) == len(voltage_chunks_only):
            success = weaviate_client.add_documents_with_external_vectors(
                voltage_chunks_only, embeddings, test_collection
            )
            
            if success:
                print("✅ 電圧調査関連チャンクのみ保存成功")
                
                # 4. 测试不同的搜索查询
                test_queries = [
                    "電圧調査では、どの4つの情報を優先的に収集すべきですか？",
                    "電圧調査で収集すべき情報は何ですか？", 
                    "電圧異常調査での記入ポイントは何ですか？",
                    "電圧調査について教えてください",
                    "電柱番号 不具合状況 発生時間帯 発生範囲"  # 直接关键词
                ]
                
                for query in test_queries:
                    print(f"\n🔍 クエリテスト: '{query}'")
                    
                    query_embedding = bedrock_service.get_embeddings([query], input_type="search_query")[0]
                    results = weaviate_client.semantic_search_with_external_vector(
                        query_embedding, test_collection, limit=8  # 增加limit
                    )
                    
                    if results:
                        print(f"   📊 結果数: {len(results)}")
                        
                        # 显示每个结果
                        for i, result in enumerate(results):
                            content = result['content']
                            
                            # 检查包含的关键信息
                            contained_terms = []
                            for term in search_terms:
                                if term in content:
                                    contained_terms.append(term)
                            
                            is_complete = len(contained_terms) >= 4
                            
                            print(f"     結果{i+1}: 相似度 {result['certainty']:.4f}")
                            print(f"       含有キーワード: {', '.join(contained_terms)}")
                            print(f"       完整回答: {'✅ はい' if is_complete else '❌ いいえ'}")
                            
                            if is_complete:
                                print(f"       🎯 これが正解です！")
                                print(f"       内容: {content[:200]}...")
                                break
                    else:
                        print("   ❌ 結果なし")
                
                # 5. 如果还是找不到，尝试直接匹配
                print(f"\n🔍 直接内容マッチングテスト...")
                
                collection = weaviate_client.client.collections.get(test_collection)
                all_results = collection.query.fetch_objects(limit=20)
                
                print(f"📦 保存された全オブジェクト: {len(all_results.objects)}")
                
                for i, obj in enumerate(all_results.objects):
                    content = obj.properties.get('content', '')
                    
                    # 检查是否包含完整的4个信息
                    contained_terms = []
                    for term in search_terms:
                        if term in content:
                            contained_terms.append(term)
                    
                    if len(contained_terms) >= 4:
                        print(f"\n🎯 完整回答発見 (オブジェクト{i}):")
                        print(f"   含有情報: {', '.join(contained_terms)}")
                        print(f"   内容プレビュー: {content[:300]}...")
                        
                        # 测试这个特定内容的相似度
                        if embeddings and i < len(embeddings):
                            print(f"   このオブジェクトのベクトルで直接検索...")
                            direct_results = weaviate_client.semantic_search_with_external_vector(
                                embeddings[i], test_collection, limit=3
                            )
                            if direct_results:
                                print(f"     直接検索結果: {len(direct_results)}件")
                                print(f"     最高相似度: {direct_results[0]['certainty']:.4f}")
            else:
                print("❌ チャンク保存失敗")
        else:
            print("❌ 嵌入生成失敗")
        
        # 清理
        try:
            weaviate_client.client.collections.delete(test_collection)
            print("✅ テストCollection削除")
        except:
            pass
            
        weaviate_client.close()
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_search_ranking() 