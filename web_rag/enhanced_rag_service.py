"""
完全修复版强化RAG服务 - 统一向量空间
"""
import sys
import os

# 项目路径设置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
weaviate_path = os.path.join(project_root, 'weaviate')
sys.path.insert(0, weaviate_path)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient
from pdf_processor import PDFProcessor
from typing import List, Dict, Any, Optional
import time
import re

class EnhancedWebRAGService:
    """完全修复版强化RAG服务 - 统一向量空间处理"""
    
    def __init__(self):
        print("🔧 初始化完全修复版强化RAG服务...")
        self.bedrock_service = TokyoBedrockService()
        self.weaviate_client = WeaviateRAGClient()
        self.pdf_processor = PDFProcessor()
        self.collection_name = "FixedDocumentCollection"
        
        # 查询扩展词典
        self.query_expansions = {
            '電圧調査': ['電圧異常', '電圧測定', '電圧確認', '電圧チェック'],
            '計器番号': ['計量器番号', 'メーター番号', '計量器'],
            '供給地点': ['供給地点特定番号', '需要場所', '供給箇所'],
            '電柱番号': ['電柱', '柱番号', 'ポール番号'],
            '情報収集': ['聞き取り', '確認項目', '調査内容', 'データ収集'],
            '優先的': ['重要', '必須', '優先', 'まず'],
            '4つの情報': ['4項目', '4つの項目', '重要事項', '調査項目'],
        }
        
        print("✅ 完全修复版RAG服务初始化完成")
    
    def initialize_knowledge_base(self, pdf_path: str) -> bool:
        """统一向量空间的知识库初始化"""
        try:
            print("📚 开始统一向量空间知识库初始化...")
            
            # 1. Weaviate连接
            if not self.weaviate_client.wait_for_weaviate(timeout=60):
                print("❌ Weaviate连接失败")
                return False
            
            # 2. 处理PDF文档
            print("📄 处理PDF文档...")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            if not chunks:
                print("❌ PDF处理失败")
                return False
            
            print(f"✅ PDF处理完成，生成 {len(chunks)} 个文档块")
            
            # 🎯 关键步骤：统一生成所有文档的Cohere嵌入
            print("🧠 统一生成文档嵌入向量（Cohere multilingual v3）...")
            doc_texts = [chunk['content'] for chunk in chunks]
            doc_embeddings = self.bedrock_service.get_embeddings(
                doc_texts, 
                input_type="search_document"
            )
            
            if not doc_embeddings or len(doc_embeddings) != len(chunks):
                print(f"❌ 文档嵌入生成失败: 需要{len(chunks)}个，获得{len(doc_embeddings) if doc_embeddings else 0}个")
                return False
            
            print(f"✅ 成功生成文档嵌入: {len(doc_embeddings)} 个，维度: {len(doc_embeddings[0])}")
            
            # 3. 创建无vectorizer的schema
            print("🗃️ 创建外部嵌入专用向量数据库...")
            if not self.weaviate_client.create_schema(self.collection_name):
                print("❌ Schema创建失败")
                return False
            
            # 4. 🎯 关键步骤：使用外部嵌入添加文档
            print("📝 使用外部嵌入添加文档到向量数据库...")
            if not self.weaviate_client.add_documents_with_external_vectors(
                chunks, doc_embeddings, self.collection_name
            ):
                print("❌ 外部嵌入文档添加失败")
                return False
            
            # 5. 验证数据
            time.sleep(5)  # 稍等片刻让数据索引完成
            count = self.weaviate_client.get_stats(self.collection_name)
            if count > 0:
                print(f"✅ 统一向量空间知识库初始化完成！包含 {count} 个文档块")
                return True
            else:
                print("❌ 知识库验证失败，文档数量为0")
                return False
                
        except Exception as e:
            print(f"❌ 知识库初始化错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unified_semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """统一向量空间的语义搜索"""
        try:
            print(f"🔍 执行统一向量空间语义搜索: {query}")
            
            # 🎯 关键步骤：使用相同Cohere模型生成查询嵌入
            query_embeddings = self.bedrock_service.get_embeddings(
                [query], 
                input_type="search_query"
            )
            
            if not query_embeddings or len(query_embeddings) == 0:
                print("❌ 查询嵌入生成失败")
                return []
            
            query_embedding = query_embeddings[0]
            print(f"✅ 查询嵌入生成成功，维度: {len(query_embedding)}")
            
            # 🎯 关键步骤：使用外部嵌入向量进行搜索
            results = self.weaviate_client.semantic_search_with_external_vector(
                query_embedding, 
                class_name=self.collection_name, 
                limit=limit
            )
            
            if results:
                print(f"✅ 统一空间搜索成功，找到 {len(results)} 个文档")
                for i, doc in enumerate(results[:3], 1):
                    print(f"  {i}. 相似度: {doc.get('certainty', 0):.3f}")
                    print(f"     内容预览: {doc['content'][:80]}...")
            
            return results
            
        except Exception as e:
            print(f"❌ 统一语义搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def multi_query_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """多查询策略搜索"""
        try:
            all_results = []
            seen_contents = set()
            
            # 原查询
            results1 = self.unified_semantic_search(query, limit)
            self.add_unique_results(results1, all_results, seen_contents, "原查询")
            
            # 扩展查询
            expanded_query = self.expand_query(query)
            if expanded_query != query:
                results2 = self.unified_semantic_search(expanded_query, limit)
                self.add_unique_results(results2, all_results, seen_contents, "扩展查询")
            
            # 按相似度排序
            all_results.sort(key=lambda x: x.get('certainty', 0), reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            print(f"❌ 多查询搜索失败: {e}")
            return []
    
    def add_unique_results(self, results: List[Dict], all_results: List[Dict], 
                          seen_contents: set, search_method: str):
        """去重添加结果"""
        for result in results:
            content_key = result['content'][:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                result['search_method'] = search_method
                all_results.append(result)
    
    def expand_query(self, query: str) -> str:
        """查询扩展"""
        expanded = query
        for key, synonyms in self.query_expansions.items():
            if key in query:
                expanded += f" {' '.join(synonyms)}"
        
        if '4つ' in query:
            expanded += " 四つ 4項目 4個"
        
        return expanded
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> Optional[str]:
        """生成精准回答"""
        try:
            if not context_docs:
                return "申し訳ございませんが、お尋ねの内容に関する情報が資料から見つかりませんでした。"
            
            print(f"\n📋 回答生成用文書 ({len(context_docs)}件):")
            for i, doc in enumerate(context_docs, 1):
                print(f"  {i}. 相似度: {doc.get('certainty', 0):.3f}")
                print(f"     内容: {doc['content'][:100]}...")
            
            # 构建上下文
            formatted_docs = []
            for i, doc in enumerate(context_docs, 1):
                certainty = doc.get('certainty', 0)
                search_method = doc.get('search_method', '統一検索')
                
                formatted_doc = f"[文書{i}] (相似度: {certainty:.3f}, 検索方法: {search_method})\n{doc['content']}"
                formatted_docs.append(formatted_doc)
            
            search_results = "\n\n".join(formatted_docs)
            
            if len(search_results) > 4500:
                search_results = search_results[:4500] + "\n\n[注: 内容が長いため一部省略]"
            
            # 针对项目列举问题的特化提示
            if any(keyword in query for keyword in ['4つ', '4項目', '優先的', '収集']):
                enhanced_prompt = f"""あなたは電力設備の専門スタッフです。以下の資料に基づいて質問に正確に回答してください。

--- 資料検索結果 ---
{search_results}
--- 検索結果終了 ---

質問: {query}

重要事項:
1. 資料の内容のみに基づいて回答してください
2. 具体的な項目がある場合は、必ず番号付きで明確に列挙してください
3. 例示がある場合は具体的に記載してください
4. 情報がない場合は「資料に記載がありません」と明記してください

回答形式:
**回答:**
[資料に基づく具体的な項目や内容]

**詳細:**
[必要に応じて各項目の説明]

**参考:**
[資料からの具体的な引用]"""
            else:
                enhanced_prompt = f"""以下の資料に基づいて質問に回答してください。

--- 資料検索結果 ---
{search_results}
--- 検索結果終了 ---

質問: {query}

資料の内容のみに基づいて、正確で実用的な回答をしてください。"""
            
            print("🤖 精准回答生成中...")
            
            answer = self.bedrock_service.chat_with_claude(
                message=enhanced_prompt,
                system_prompt="資料に基づく正確な回答を提供してください。推測や想像は禁止です。具体的な項目は明確に列挙してください。",
                max_tokens=2000
            )
            
            if answer:
                print("✅ 精准回答生成成功")
                return answer
            else:
                return "申し訳ございませんが、回答の生成に失敗いたしました。"
                
        except Exception as e:
            print(f"❌ 回答生成错误: {e}")
            return "システムエラーが発生いたしました。"
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """完整的统一向量空间查询"""
        print(f"\n{'='*80}")
        print(f"🔍 統一ベクトル空間クエリ実行: {question}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # 多查询策略搜索
        relevant_docs = self.multi_query_search(question, limit=top_k)
        
        # 生成回答
        answer = self.generate_answer(question, relevant_docs)
        
        processing_time = time.time() - start_time
        
        result = {
            "question": question,
            "answer": answer,
            "sources": relevant_docs,
            "source_count": len(relevant_docs),
            "processing_time": round(processing_time, 2),
            "vector_space": "unified_cohere_external"
        }
        
        print(f"\n✅ 統一空間クエリ完了，処理時間: {processing_time:.2f}秒")
        print(f"{'='*80}")
        
        return result
    
    def close(self):
        """关闭连接"""
        try:
            if hasattr(self.weaviate_client, 'close'):
                self.weaviate_client.close()
            print("✅ RAG服务连接已关闭")
        except Exception as e:
            print(f"⚠️ 关闭连接时警告: {e}")

def test_unified_rag():
    """测试统一向量空间RAG"""
    pdf_path = os.path.join(os.path.dirname(__file__), "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDFファイルが見つかりません: {pdf_path}")
        return
    
    rag_service = None
    try:
        rag_service = EnhancedWebRAGService()
        
        # 统一向量空间知识库初始化
        print("🚀 統一ベクトル空間知識ベース初期化開始...")
        if not rag_service.initialize_knowledge_base(pdf_path):
            print("❌ 統一知識ベース初期化失敗")
            return
        
        # 关键测试问题
        critical_question = "電圧調査では、どの4つの情報を優先的に収集すべきですか？"
        
        print(f"\n{'='*100}")
        print("🎯 关键问题测试 - 统一向量空间版本")
        result = rag_service.query(critical_question)
        
        print(f"\n📊 测试结果:")
        print(f"问题: {result['question']}")
        print(f"检索文档数: {result['source_count']}")
        print(f"处理时间: {result['processing_time']}秒")
        print(f"向量空间: {result['vector_space']}")
        print(f"\n🤖 回答:")
        print(result['answer'])
        
        # 检查是否包含期待的关键词
        answer_text = result['answer']
        expected_keywords = ['電柱番号', '不具合状況', '発生時間帯', '発生範囲']
        found_keywords = [kw for kw in expected_keywords if kw in answer_text]
        
        print(f"\n🔍 关键词检查:")
        print(f"期待关键词: {expected_keywords}")
        print(f"发现关键词: {found_keywords}")
        
        if len(found_keywords) >= 3:
            print("🎉 SUCCESS: 关键信息检索成功！")
        else:
            print("⚠️ WARNING: 部分关键信息可能缺失")
        
        print(f"{'='*100}")
        
    except Exception as e:
        print(f"❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag_service:
            rag_service.close()

if __name__ == "__main__":
    test_unified_rag() 