#!/usr/bin/env python3
"""
完整PDF处理调试 - 从提取到存储的全流程
"""
import sys
import os

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pdf_processor import PDFProcessor
from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient

def debug_complete_pdf_processing():
    print("🔧 完整PDF処理調査開始...")
    
    pdf_path = "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF文件不存在: {pdf_path}")
        return
    
    try:
        # 1. PDF文本提取和分块
        print("\n🔍 1. PDF文本提取と分塊調査...")
        processor = PDFProcessor()
        
        # 🎯 使用正确的方法名
        chunks = processor.process_pdf(pdf_path)
        print(f"📊 分塊結果: {len(chunks)}個のチャンク")
        
        # 查找包含電圧調査的块
        voltage_chunks = []
        search_terms = ['電圧調査', '電柱番号', '不具合状況', '発生時間帯', '発生範囲']
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            title = chunk.get('title', '')
            
            found_terms = []
            for term in search_terms:
                if term in content or term in title:
                    found_terms.append(term)
            
            if found_terms:
                voltage_chunks.append((i, chunk, found_terms))
                print(f"✅ チャンク{i}: 電圧調査関連内容発見")
                print(f"   発見キーワード: {', '.join(found_terms)}")
                print(f"   内容長: {len(content)}文字")
                print(f"   プレビュー: {content[:100]}...")
        
        print(f"📊 電圧調査関連チャンク: {len(voltage_chunks)}個")
        
        if not voltage_chunks:
            print("❌ 電圧調査関連チャンクが見つかりません！")
            print("🔍 全チャンクをチェック...")
            
            # 显示所有块的简要信息
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '')
                title = chunk.get('title', '')
                print(f"\nチャンク{i}:")
                print(f"  タイトル: {title}")
                print(f"  長さ: {len(content)}")
                print(f"  プレビュー: {content[:150]}...")
                
                # 检查是否包含17页相关内容
                if any(keyword in content.lower() for keyword in ['page 17', 'ページ17', '電圧調査について']):
                    print("  ⚠️ このチャンクにpage 17または電圧調査の可能性")
                    print(f"  完整内容: {content}")
            
            print("\n🔍 原因分析:")
            print("1. PDF分塊時にキーワードが分離された可能性")
            print("2. 文字エンコーディング問題")
            print("3. PDF構造が複雑でテキスト抽出に問題")
            return
        
        # 2. 显示找到的電圧調査相关块的完整内容
        print(f"\n🔍 2. 電圧調査関連チャンク詳細表示...")
        for i, (chunk_index, chunk, found_terms) in enumerate(voltage_chunks):
            print(f"\n=== 関連チャンク{i+1} (元インデックス: {chunk_index}) ===")
            print(f"発見キーワード: {', '.join(found_terms)}")
            print(f"タイトル: {chunk.get('title', 'N/A')}")
            print(f"ソース: {chunk.get('source', 'N/A')}")
            print(f"内容長: {len(chunk.get('content', ''))}")
            print(f"内容:")
            print("-" * 50)
            print(chunk.get('content', ''))
            print("-" * 50)
            
            # 检查是否包含完整的4つの情報
            content = chunk.get('content', '')
            complete_info = all(term in content for term in search_terms)
            print(f"完整4情報包含: {'✅ はい' if complete_info else '❌ いいえ'}")
        
        # 3. 测试嵌入生成
        print(f"\n🔍 3. 嵌入生成テスト...")
        bedrock_service = TokyoBedrockService()
        
        if voltage_chunks:
            test_chunk = voltage_chunks[0][1]  # 第一个電圧調査相关块
        else:
            test_chunk = chunks[0] if chunks else None
        
        if test_chunk:
            test_content = test_chunk.get('content', '')
            
            if len(test_content) > 10:
                try:
                    # 限制内容长度避免token过多
                    content_for_embedding = test_content[:1000]
                    embedding = bedrock_service.get_embeddings([content_for_embedding], input_type="search_document")
                    if embedding and len(embedding) > 0:
                        print(f"✅ 嵌入生成成功: {len(embedding[0])}次元")
                    else:
                        print("❌ 嵌入生成失敗 - 空の結果")
                except Exception as e:
                    print(f"❌ 嵌入生成エラー: {e}")
            else:
                print("⚠️ テストコンテンツが短すぎます")
        else:
            print("❌ テスト用チャンクがありません")
        
        # 4. 测试存储到Weaviate
        if voltage_chunks:
            print(f"\n🔍 4. Weaviate保存テスト...")
            weaviate_client = WeaviateRAGClient()
            
            if not weaviate_client.wait_for_weaviate():
                print("❌ Weaviate接続失敗")
                return
            
            # 创建测试collection
            test_collection = "PDFDebugTest"
            try:
                if weaviate_client.client.collections.exists(test_collection):
                    weaviate_client.client.collections.delete(test_collection)
            except:
                pass
            
            if weaviate_client.create_schema(test_collection):
                print("✅ テストコレクション作成成功")
                
                # 只测试電圧調査相关的块
                test_chunks = [chunk for _, chunk, _ in voltage_chunks[:3]]  # 最多测试3个块
                
                # 生成嵌入
                contents = [chunk.get('content', '')[:1000] for chunk in test_chunks]  # 限制长度
                embeddings = bedrock_service.get_embeddings(contents, input_type="search_document")
                
                if embeddings and len(embeddings) == len(test_chunks):
                    # 尝试存储
                    success = weaviate_client.add_documents_with_external_vectors(
                        test_chunks, embeddings, test_collection
                    )
                    
                    if success:
                        print("✅ 電圧調査チャンク保存成功！")
                        
                        # 测试搜索
                        print("🔍 検索テスト実行中...")
                        query = "電圧調査では、どの4つの情報を優先的に収集すべきですか？"
                        query_embedding = bedrock_service.get_embeddings([query], input_type="search_query")[0]
                        
                        results = weaviate_client.semantic_search_with_external_vector(
                            query_embedding, test_collection, limit=5
                        )
                        
                        if results:
                            print(f"🎉 検索テスト成功！{len(results)}件の結果")
                            for i, result in enumerate(results):
                                print(f"\n結果{i+1}:")
                                print(f"  相似度: {result['certainty']:.4f}")
                                print(f"  タイトル: {result['title']}")
                                print(f"  内容プレビュー: {result['content'][:200]}...")
                                
                                # 检查是否包含4つの情報
                                content = result['content']
                                contains_info = []
                                for term in search_terms:
                                    if term in content:
                                        contains_info.append(term)
                                        
                                if contains_info:
                                    print(f"  含まれる情報: {', '.join(contains_info)}")
                                else:
                                    print(f"  含まれる情報: なし")
                        else:
                            print("❌ 検索テスト失敗 - 結果なし")
                    else:
                        print("❌ 電圧調査チャンク保存失敗")
                else:
                    print(f"❌ 嵌入生成とチャンク数が不一致: 嵌入{len(embeddings) if embeddings else 0}, チャンク{len(test_chunks)}")
                
                # 清理
                try:
                    weaviate_client.client.collections.delete(test_collection)
                    print("✅ テストコレクション削除")
                except:
                    pass
            else:
                print("❌ テストコレクション作成失敗")
            
            weaviate_client.close()
        else:
            print("❌ 電圧調査チャンクが見つからないため、Weaviateテストをスキップ")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_complete_pdf_processing() 