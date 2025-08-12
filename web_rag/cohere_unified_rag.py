#!/usr/bin/env python3
"""
统一Cohere RAG服务 - 完全基于AWS Bedrock Cohere嵌入
"""
import sys
import os
import importlib

# 强制清除缓存
if 'weaviate_client' in sys.modules:
    importlib.reload(sys.modules['weaviate_client'])

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient
from pdf_processor import PDFProcessor
from typing import List, Dict, Any, Optional
import time
import re

class CohereUnifiedRAGService:
    """統一Cohere RAGサービス - すべてAWS Bedrock Cohereを使用"""
    
    def __init__(self):
        print("🎯 統一Cohere RAGサービス初期化中...")
        self.bedrock_service = TokyoBedrockService()
        self.weaviate_client = WeaviateRAGClient()
        self.pdf_processor = PDFProcessor()
        self.knowledge_loaded = False
        
    def load_pdf_knowledge(self, pdf_path: str, force_reload: bool = False):
        """PDFナレッジベースの読み込み - 向量维度冲突修复版"""
        try:
            # 1. 🔧 重要修正：清除旧collection，避免维度冲突
            print("🧹 旧データクリア中...")
            try:
                # 旧collectionを削除
                if hasattr(self.weaviate_client.client, 'collections'):
                    collections = self.weaviate_client.client.collections
                    if collections.exists("Document"):
                        collections.delete("Document")
                        print("✅ 旧DocumentCollection削除完了")
                    
                    # 新collectionを作成（Cohere用：1024次元）
                    success = self.weaviate_client.create_schema()
                    if not success:
                        print("❌ 新schema作成失敗")
                        return False
                    print("✅ 新schema作成完了（Cohere 1024次元用）")
                        
            except Exception as e:
                print(f"⚠️ Collection削除警告: {str(e)}")
                # 続行する
            
            # 2. PDF処理
            print("📄 PDF処理開始...")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            # 3. 電圧調査関連チャンクのフィルタリング
            voltage_keywords = [
                '電圧調査', '電圧異常', '電柱番号', '不具合状況', 
                '発生時間帯', '発生範囲', '記入のポイント',
                '電圧', '調査', '問合せ情報'
            ]
            
            relevant_chunks = []
            complete_answer_chunks = []
            
            for i, chunk in enumerate(chunks):
                content = chunk['content']
                content_lower = content.lower()
                
                # 関連性チェック
                if any(keyword.lower() in content_lower for keyword in voltage_keywords):
                    chunk['chunk_id'] = f"chunk_{i}"
                    relevant_chunks.append(chunk)
                    
                    # 完整回答チェック
                    required_all = ['電柱番号', '不具合状況', '発生時間帯', '発生範囲']
                    has_all = all(req in content for req in required_all)
                    
                    if has_all and ('記入のポイント' in content or '電圧調査について' in content):
                        complete_answer_chunks.append(chunk)
                        print(f"🎯 完整回答発見: chunk_{i}")
            
            print(f"📊 関連チャンク: {len(relevant_chunks)}個")
            print(f"📊 完整回答チャンク: {len(complete_answer_chunks)}個")
            
            if not relevant_chunks:
                print("❌ 関連チャンクが見つかりません")
                return False
            
            # 4. 完整回答を最優先で配置
            priority_chunks = complete_answer_chunks + [
                chunk for chunk in relevant_chunks 
                if chunk not in complete_answer_chunks
            ]
            
            # 5. Cohere嵌入向量生成
            print("🔧 Cohere嵌入向量生成中...")
            
            # 文書内容を抽出
            texts = [chunk['content'] for chunk in priority_chunks]
            
            # 🎯 重要：search_document用に嵌入生成
            embeddings = self.bedrock_service.get_embeddings(
                texts, 
                input_type="search_document"  # 文書保存用
            )
            
            if not embeddings:
                print("❌ 嵌入向量生成失敗")
                return False
                
            if len(embeddings) != len(priority_chunks):
                print(f"❌ 嵌入数量不一致: 文書{len(priority_chunks)}個 vs 嵌入{len(embeddings)}個")
                return False
                
            print(f"✅ 嵌入向量生成成功: {len(embeddings)}個の{len(embeddings[0])}次元ベクトル")
            
            # 6. 🎯 清理后的数据库添加
            print("💾 クリーン環境でWeaviate保存開始...")
            success = self.weaviate_client.add_documents_with_external_vectors(
                priority_chunks,  # 文書リスト
                embeddings       # 生成済み1024次元嵌入向量リスト
            )
            
            if success:
                print(f"✅ ナレッジベース構築完了: {len(priority_chunks)}個のチャンク")
                self.knowledge_loaded = True
                return True
            else:
                print("❌ ナレッジベース構築失敗")
                return False
                
        except Exception as e:
            print(f"❌ PDFナレッジ読み込みエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def cohere_search(self, query: str, limit: int = 8) -> List[Dict]:
        """純Cohere検索 - 正しいAPI使用"""
        try:
            print(f"🔍 Cohere検索実行: '{query}' (limit={limit})")
            
            # 1. 🎯 重要：まずqueryをCohereで嵌入に変換
            query_embeddings = self.bedrock_service.get_embeddings(
                [query], 
                input_type="search_query"  # 🎯 検索用
            )
            
            if not query_embeddings:
                print("❌ 查询向量生成失败")
                return []
            
            query_vector = query_embeddings[0]  # 最初のベクトルを取得
            print(f"✅ 查询向量生成成功: {len(query_vector)}维")
            
            # 2. 🎯 重要：正しいAPIで外部ベクトル検索
            search_results = self.weaviate_client.semantic_search_with_external_vector(
                query_vector=query_vector,  # 🎯 向量を渡す
                class_name="Document",
                limit=limit  # 🎯 limit パラメータを使用
            )
            
            print(f"✅ 検索完了: {len(search_results)}件発見")
            return search_results
            
        except Exception as e:
            print(f"❌ Cohere検索エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def enhanced_search_and_rank(self, query: str, limit: int = 8) -> List[Dict]:
        """強化された検索とランキング"""
        try:
            # 基本検索
            basic_results = self.cohere_search(query, limit)
            
            if not basic_results:
                return []
            
            # 完整回答優先ランキング
            complete_keywords = ['電柱番号', '不具合状況', '発生時間帯', '発生範囲']
            bonus_phrases = ['記入のポイント', '電圧調査について', '電圧異常の場合']
            
            scored_results = []
            for result in basic_results:
                content = result.get('content', '')
                
                # 完整性スコア
                completeness_score = sum(1 for kw in complete_keywords if kw in content)
                
                # ボーナススコア
                bonus_score = sum(2 for phrase in bonus_phrases if phrase in content)
                
                # 基本相似度
                similarity = result.get('similarity', result.get('certainty', 0))
                
                # 総合スコア = 相似度 + 完整性ボーナス + フレーズボーナス
                total_score = similarity + (completeness_score * 0.05) + (bonus_score * 0.03)
                
                result['total_score'] = total_score
                result['completeness_score'] = completeness_score
                result['similarity'] = similarity
                
                scored_results.append(result)
                
                print(f"   📊 スコア: 相似度={similarity:.3f}, 完整性={completeness_score}/4, 総合={total_score:.3f}")
            
            # 総合スコア順でソート
            ranked_results = sorted(scored_results, key=lambda x: x['total_score'], reverse=True)
            
            return ranked_results[:3]  # トップ3を返す
            
        except Exception as e:
            print(f"❌ 強化検索エラー: {str(e)}")
            return []
    
    def generate_answer_claude(self, query: str, context_docs: List[Dict]) -> str:
        """Claude4による回答生成 - 修正版"""
        
        if not context_docs:
            return "申し訳ありませんが、関連する文書が見つかりませんでした。"
        
        # コンテキスト構築
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            completeness = doc.get('completeness_score', 0)
            total_score = doc.get('total_score', 0)
            
            context_parts.append(f"""[検索結果{i}] (相似度: {similarity:.3f}, 完整性: {completeness}/4, 総合: {total_score:.3f})
{content}""")
        
        context = "\n\n".join(context_parts)
        
        # 電圧調査専用プロンプト
        voltage_prompt = f"""あなたは関西電力の電力設備申込み業務に精通したカスタマーサポート担当者です。
次に示すのは、社内の「高圧託送業務WEBマニュアル」から検索した関連文書です。

--- 検索結果開始 ---
{context}
--- 検索結果終了 ---

質問: {query}

回答要件:
1. 回答は必ず検索結果に基づいて作成してください
2. 検索結果に記載がない情報は、推測せず「資料に記載がありません」と答えてください  
3. 電圧調査に関する4つの情報が質問されている場合は、以下の形式で回答してください：
   
   電圧調査では、以下の4つの情報を優先的に収集します：
   1. [項目名]: [説明・例]
   2. [項目名]: [説明・例]  
   3. [項目名]: [説明・例]
   4. [項目名]: [説明・例]
   
4. 回答は1〜3文以内で、日本語で簡潔に記述してください
5. 必要に応じて検索結果の根拠（ページ番号または文章抜粋）を最後に示してください

回答:"""

        try:
            print("🤖 Claude4で回答生成中...")
            # 🔧 修正：正しいメソッド名を使用
            response = self.bedrock_service.chat_with_claude(
                message=voltage_prompt,
                system_prompt="",
                max_tokens=2000
            )
            return response if response else "申し訳ありませんが、回答の生成に失敗しました。"
            
        except Exception as e:
            print(f"❌ 回答生成エラー: {str(e)}")
            return f"申し訳ありませんが、回答生成中にエラーが発生しました: {str(e)}"
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """質問応答メイン処理 - 統一Cohere版"""
        start_time = time.time()
        
        try:
            if not self.knowledge_loaded:
                return {
                    'answer': "ナレッジベースが読み込まれていません。",
                    'search_results': [],
                    'processing_time': 0,
                    'confidence': 0
                }
            
            # 1. Cohere検索実行
            search_results = self.enhanced_search_and_rank(question, limit=8)
            
            print(f"📊 最終検索結果: {len(search_results)}件")
            
            # 2. Claude4回答生成
            answer = self.generate_answer_claude(question, search_results)
            
            # 3. 信頼度計算
            confidence = 0
            if search_results:
                best_result = search_results[0]
                confidence = min(best_result.get('total_score', 0), 1.0)
            
            processing_time = time.time() - start_time
            
            return {
                'answer': answer,
                'search_results': search_results[:3],
                'processing_time': processing_time,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"❌ 質問応答エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f"申し訳ありませんが、エラーが発生しました: {str(e)}",
                'search_results': [],
                'processing_time': time.time() - start_time,
                'confidence': 0
            }
    
    def close(self):
        """リソースクリーンアップ"""
        try:
            if hasattr(self.weaviate_client, 'close'):
                self.weaviate_client.close()
        except Exception as e:
            print(f"クリーンアップ警告: {str(e)}")

# テスト実行
if __name__ == "__main__":
    print("🎯 統一Cohere RAGシステムテスト")
    
    service = CohereUnifiedRAGService()
    
    try:
        # PDFナレッジ読み込み
        pdf_path = "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
        print(f"📖 PDFファイル読み込み: {pdf_path}")
        
        if service.load_pdf_knowledge(pdf_path):
            print("✅ ナレッジベース構築完了")
            
            # テスト質問
            test_questions = [
                "電圧調査では、どの4つの情報を優先的に収集すべきですか？",
                "電圧調査について教えてください",
                "電圧異常調査での記入ポイントは何ですか？"
            ]
            
            for i, question in enumerate(test_questions, 1):
                print(f"\n{'='*60}")
                print(f"🧪 テスト {i}/3: {question}")
                print('='*60)
                
                result = service.ask_question(question)
                
                print(f"\n📝 回答:")
                print(f"{result['answer']}")
                print(f"\n📊 メタデータ:")
                print(f"   信頼度: {result['confidence']:.3f}")
                print(f"   処理時間: {result['processing_time']:.2f}秒")
                print(f"   検索結果数: {len(result['search_results'])}件")
                
        else:
            print("❌ ナレッジベースの構築に失敗しました")
            
    finally:
        # リソースクリーンアップ  
        service.close()
        print("🧹 リソースクリーンアップ完了") 