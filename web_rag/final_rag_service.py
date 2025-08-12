"""
最终版RAG服务 - 修复导入问题
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
weaviate_path = os.path.join(project_root, 'weaviate')
sys.path.insert(0, weaviate_path)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient  # 直接从当前目录导入
from pdf_processor import PDFProcessor
from typing import List, Dict, Any, Optional
import time
import re

class FinalWebRAGService:
    """最终版WebRAGサービス - 修复版"""
    
    def __init__(self):
        print("🎯 最終版RAGサービス初期化中...")
        self.bedrock_service = TokyoBedrockService()
        self.weaviate_client = WeaviateRAGClient()
        self.pdf_processor = PDFProcessor()
        self.knowledge_loaded = False
        
        # デバッグ: メソッド存在確認
        print(f"🔧 WeaviateRAGClient メソッド確認:")
        methods = [method for method in dir(self.weaviate_client) if not method.startswith('_')]
        print(f"   利用可能メソッド: {', '.join(methods)}")
        
        # 具体的にチェック
        if hasattr(self.weaviate_client, 'add_documents_with_external_vectors'):
            print("✅ add_documents_with_external_vectors メソッド存在")
        else:
            print("❌ add_documents_with_external_vectors メソッド存在しません")
    
    def load_pdf_knowledge(self, pdf_path: str, force_reload: bool = False):
        """PDFナレッジベースの読み込み - 修复版"""
        try:
            # 1. PDF処理
            print("📄 PDF処理開始...")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            # 2. 電圧調査関連チャンクのフィルタリング（改良版）
            voltage_keywords = [
                '電圧調査', '電圧異常', '電柱番号', '不具合状況', 
                '発生時間帯', '発生範囲', '記入のポイント',
                '電圧', '調査', '問合せ情報'
            ]
            
            relevant_chunks = []
            for i, chunk in enumerate(chunks):
                content = chunk['content'].lower()
                # より寛容な一致条件
                if any(keyword.lower() in content for keyword in voltage_keywords):
                    chunk['chunk_id'] = f"chunk_{i}"
                    relevant_chunks.append(chunk)
            
            # 3. 完整回答包含检测（改良版）
            complete_answer_chunks = []
            for chunk in relevant_chunks:
                content = chunk['content']
                # より厳密な完整回答検出
                required_all = ['電柱番号', '不具合状況', '発生時間帯', '発生範囲']
                has_all = all(req in content for req in required_all)
                
                if has_all and ('記入のポイント' in content or '電圧調査について' in content):
                    complete_answer_chunks.append(chunk)
                    print(f"🎯 完整回答発見: chunk_{relevant_chunks.index(chunk)}")
            
            print(f"📊 関連チャンク: {len(relevant_chunks)}個")
            print(f"📊 完整回答チャンク: {len(complete_answer_chunks)}個")
            
            if not complete_answer_chunks:
                print("❌ 完整回答が見つかりません")
                return False
            
            # 4. Weaviateへの保存 - 完整回答を优先
            print("💾 Weaviate保存開始...")
            
            # まず完整回答を最初に保存（優先度向上）
            priority_chunks = complete_answer_chunks + [
                chunk for chunk in relevant_chunks 
                if chunk not in complete_answer_chunks
            ]
            
            # メソッド存在チェック
            if not hasattr(self.weaviate_client, 'add_documents_with_external_vectors'):
                print("❌ メソッドが存在しないため、代替メソッドを使用...")
                # 代替手段: 直接 add_documents を使用
                if hasattr(self.weaviate_client, 'add_documents'):
                    print("🔄 add_documents メソッドを使用...")
                    # 先获取嵌入
                    texts = [chunk['content'] for chunk in priority_chunks]
                    embeddings = self.bedrock_service.get_embeddings(texts, input_type="search_document")
                    
                    # 添加向量到文档
                    for i, chunk in enumerate(priority_chunks):
                        if i < len(embeddings):
                            chunk['vector'] = embeddings[i]
                    
                    success = self.weaviate_client.add_documents(priority_chunks)
                else:
                    print("❌ どのドキュメント追加メソッドも見つかりません")
                    return False
            else:
                success = self.weaviate_client.add_documents_with_external_vectors(
                    priority_chunks, self.bedrock_service.get_embeddings
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
    
    def enhanced_search(self, query: str, top_k: int = 8) -> List[Dict]:
        """強化された検索 - 修復版"""
        try:
            # 正しい検索メソッドを使用
            if hasattr(self.weaviate_client, 'semantic_search_with_external_vector'):
                print(f"🔍 外部ベクトル検索を使用: top_k={top_k}")
                basic_results = self.weaviate_client.semantic_search_with_external_vector(
                    query, self.bedrock_service.get_embeddings, top_k=top_k
                )
            elif hasattr(self.weaviate_client, 'semantic_search'):
                print(f"🔍 基本検索を使用: limit={top_k}")
                # semantic_searchはlimitパラメータを使用
                basic_results = self.weaviate_client.semantic_search(query, limit=top_k)
            else:
                print("❌ 検索メソッドが見つかりません")
                return []
            
            print(f"🔍 基本検索結果: {len(basic_results)}件")
            
            # 完整回答优先排序
            def prioritize_complete_answers(results):
                complete_keywords = ['電柱番号', '不具合状況', '発生時間帯', '発生範囲']
                
                scored_results = []
                for result in results:
                    content = result.get('content', '')
                    
                    # 完整性评分
                    completeness_score = sum(1 for kw in complete_keywords if kw in content)
                    
                    # 关键短语加分
                    bonus_phrases = ['記入のポイント', '電圧調査について', '電圧異常の場合']
                    bonus_score = sum(2 for phrase in bonus_phrases if phrase in content)
                    
                    # 获取相似度（处理不同的字段名）
                    similarity = result.get('similarity', result.get('certainty', result.get('_additional', {}).get('certainty', 0)))
                    
                    # 总得分 = 相似度 + 完整性加分 + 关键短语加分
                    total_score = similarity + (completeness_score * 0.05) + (bonus_score * 0.03)
                    
                    result['total_score'] = total_score
                    result['completeness_score'] = completeness_score
                    result['similarity'] = similarity
                    scored_results.append(result)
                    
                    print(f"   - 文書得点: 相似度={similarity:.3f}, 完整性={completeness_score}/4, 総合={total_score:.3f}")
                
                return sorted(scored_results, key=lambda x: x['total_score'], reverse=True)
            
            prioritized_results = prioritize_complete_answers(basic_results)
            return prioritized_results[:3]
            
        except Exception as e:
            print(f"❌ 検索エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """回答生成 - 優化されたプロンプト"""
        
        if not context_docs:
            return "申し訳ありませんが、関連する文書が見つかりませんでした。"
        
        # コンテキスト構築
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            completeness = doc.get('completeness_score', 0)
            
            context_parts.append(f"""[検索結果{i}] (相似度: {similarity:.3f}, 完整性: {completeness}/4)
{content}""")
        
        context = "\n\n".join(context_parts)
        
        # 最適化されたプロンプト
        enhanced_prompt = f"""あなたは電力設備の申込み業務に精通したカスタマーサポート担当者です。
次に示すのは、社内マニュアルから検索した関連文書です。

--- 検索結果開始 ---
{context}
--- 検索結果終了 ---

質問: {query}

回答要件:
1. 回答は必ず検索結果に基づくこと
2. 検索結果に明確な記載がない場合は「マニュアルに記載がありません」と回答
3. 回答は日本語で、簡潔かつ正確に記述すること  
4. 電圧調査に関する具体的な4つの情報が質問されている場合は、番号付きリストで回答
5. 必要に応じて検索結果からの引用（抜粋）を根拠として示すこと

回答:"""

        try:
            response = self.bedrock_service.generate_text_claude(enhanced_prompt)
            return response if response else "申し訳ありませんが、回答の生成に失敗しました。"
            
        except Exception as e:
            print(f"❌ 回答生成エラー: {str(e)}")
            return f"申し訳ありませんが、回答生成中にエラーが発生しました: {str(e)}"
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """質問応答メイン処理"""
        start_time = time.time()
        
        try:
            if not self.knowledge_loaded:
                return {
                    'answer': "ナレッジベースが読み込まれていません。",
                    'search_results': [],
                    'processing_time': 0,
                    'confidence': 0
                }
            
            # 検索実行
            print(f"🔍 検索実行: '{question}'")
            search_results = self.enhanced_search(question, top_k=8)
            
            print(f"📊 検索結果: {len(search_results)}件")
            for i, result in enumerate(search_results[:3], 1):
                completeness = result.get('completeness_score', 0)
                total_score = result.get('total_score', 0)
                print(f"   結果{i}: 相似度={result.get('similarity', 0):.3f}, 完整性={completeness}/4, 総合={total_score:.3f}")
            
            # 回答生成
            print("🤖 回答生成中...")
            answer = self.generate_answer(question, search_results)
            
            # 信頼度計算
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

# 使用例
if __name__ == "__main__":
    print("🎯 最終版RAGシステムテスト（修正版）")
    
    service = FinalWebRAGService()
    
    try:
        # PDFを読み込み
        pdf_path = "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
        if service.load_pdf_knowledge(pdf_path):
            
            # テスト質問
            test_questions = [
                "電圧調査では、どの4つの情報を優先的に収集すべきですか？",
                "電圧調査について教えてください",
                "電圧異常調査での記入ポイントは何ですか？"
            ]
            
            for question in test_questions:
                print(f"\n{'='*50}")
                print(f"質問: {question}")
                print('='*50)
                
                result = service.ask_question(question)
                print(f"\n回答:\n{result['answer']}")
                print(f"\n信頼度: {result['confidence']:.3f}")
                print(f"処理時間: {result['processing_time']:.2f}秒")
        else:
            print("❌ PDFナレッジの読み込みに失敗しました")
            
    finally:
        # リソースクリーンアップ
        service.close() 