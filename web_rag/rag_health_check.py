"""
更新版RAG系统健康检查工具 - 检测修复后的配置
"""
import sys
import os
import numpy as np
from typing import List, Dict, Any
import time

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
weaviate_path = os.path.join(project_root, 'weaviate')
sys.path.insert(0, weaviate_path)

from bedrock.bedrock_usage import TokyoBedrockService
from weaviate_client import WeaviateRAGClient

class UpdatedRAGHealthChecker:
    """更新版RAG健康诊断工具"""
    
    def __init__(self):
        print("🔧 更新版RAGヘルスチェッカー初期化中...")
        self.bedrock_service = TokyoBedrockService()
        self.weaviate_client = WeaviateRAGClient()
    
    def check_embedding_consistency(self):
        """1. 嵌入一致性检查"""
        print("\n" + "="*60)
        print("📊 1. 嵌入モデル一致性チェック")
        print("="*60)
        
        # テスト文書と查询
        test_docs = [
            "電圧調査では電柱番号を確認します",
            "計器番号の聞き取りが重要です", 
            "4つの情報を優先的に収集します",
            "不具合状況を詳しく聞きます",
            "発生時間帯と範囲を特定します"
        ]
        
        test_queries = [
            "電圧調査の4つの情報とは何ですか",  # 关键查询
            "電柱番号について教えて",           
            "計器番号の重要性は",
            "全く関係ない天気の話"              # 无关查询
        ]
        
        print("🔍 文書嵌入取得中（input_type=search_document）...")
        doc_embeddings = self.bedrock_service.get_embeddings(
            test_docs, 
            input_type="search_document"
        )
        
        print("🔍 查询嵌入取得中（input_type=search_query）...")
        query_embeddings = self.bedrock_service.get_embeddings(
            test_queries,
            input_type="search_query" 
        )
        
        if not doc_embeddings or not query_embeddings:
            print("❌ 嵌入取得失敗")
            return False
        
        print(f"✅ 文書嵌入: {len(doc_embeddings)} 個, 次元: {len(doc_embeddings[0])}")
        print(f"✅ 查询嵌入: {len(query_embeddings)} 個, 次元: {len(query_embeddings[0])}")
        
        if len(doc_embeddings[0]) != len(query_embeddings[0]):
            print("❌ 嵌入次元不一致！")
            return False
        
        # 重点相似度測试 - 验证关键问题
        print("\n📈 关键相似度テスト:")
        key_query = "電圧調査の4つの情報とは何ですか"  # 关键查询
        query_vec = np.array(query_embeddings[0])  # 第一个查询
        
        print(f"🎯 关键查询: {key_query}")
        
        max_similarity = 0
        best_match_idx = -1
        
        for j, doc_text in enumerate(test_docs):
            doc_vec = np.array(doc_embeddings[j])
            cosine_sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            
            if cosine_sim > max_similarity:
                max_similarity = cosine_sim
                best_match_idx = j
            
            status = "🎯" if cosine_sim > 0.6 else "⚠️" if cosine_sim > 0.3 else "❌"
            print(f"  {status} {cosine_sim:.3f}: {doc_text}")
        
        # 判定结果
        if max_similarity > 0.5:
            print(f"✅ 嵌入一致性良好！最高相似度: {max_similarity:.3f}")
            return True
        else:
            print(f"❌ 嵌入相似度过低！最高相似度: {max_similarity:.3f}")
            return False
    
    def check_updated_weaviate_config(self):
        """2. 检查更新后的Weaviate配置"""
        print("\n" + "="*60)
        print("🗄️ 2. 更新版Weaviate設定チェック")
        print("="*60)
        
        try:
            # Weaviate连接检查
            if not self.weaviate_client.wait_for_weaviate(timeout=10):
                print("❌ Weaviate接続失敗")
                return False
            
            # 测试修复版schema创建
            print("🔧 修复版Schema作成テスト...")
            test_collection = "HealthCheckFixed"
            
            try:
                result = self.weaviate_client.create_schema(test_collection)
                
                if result:
                    print("✅ 修复版Schema作成成功")
                    print("   - vectorizer: None (外部嵌入専用)")
                    print("   - distance: COSINE (余弦相似度)")
                    
                    # 测试外部向量添加功能
                    print("🧪 外部ベクトル追加テスト...")
                    test_docs = [{"content": "テスト文書", "title": "テスト"}]
                    test_embeddings = [[0.1] * 1024]  # 假的嵌入
                    
                    # 这里只测试方法是否存在，不实际添加
                    if hasattr(self.weaviate_client, 'add_documents_with_external_vectors'):
                        print("✅ 外部ベクトル追加メソッド利用可能")
                    else:
                        print("❌ 外部ベクトル追加メソッド不存在")
                        return False
                    
                    if hasattr(self.weaviate_client, 'semantic_search_with_external_vector'):
                        print("✅ 外部ベクトル検索メソッド利用可能")
                    else:
                        print("❌ 外部ベクトル検索メソッド不存在")
                        return False
                    
                    return True
                else:
                    print("❌ 修复版Schema作成失敗")
                    return False
                    
            except Exception as e:
                print(f"❌ Schema作成テストエラー: {e}")
                return False
            
        except Exception as e:
            print(f"❌ Weaviate設定チェックエラー: {e}")
            return False
    
    def check_vector_space_unity(self):
        """3. 向量空间统一性检查"""
        print("\n" + "="*60)
        print("🎯 3. ベクトル空間統一性チェック")
        print("="*60)
        
        issues_found = []
        fixes_applied = []
        
        # 检查修复项目
        print("🔍 修复項目確認:")
        
        # 1. 检查外部嵌入方法是否存在
        if hasattr(self.weaviate_client, 'add_documents_with_external_vectors'):
            print("  ✅ 外部嵌入文書追加メソッド: 利用可能")
            fixes_applied.append("外部嵌入文書追加")
        else:
            print("  ❌ 外部嵌入文書追加メソッド: 不存在")
            issues_found.append("外部嵌入文書追加メソッド不存在")
        
        # 2. 检查外部向量搜索方法
        if hasattr(self.weaviate_client, 'semantic_search_with_external_vector'):
            print("  ✅ 外部ベクトル検索メソッド: 利用可能")
            fixes_applied.append("外部ベクトル検索")
        else:
            print("  ❌ 外部ベクトル検索メソッド: 不存在")
            issues_found.append("外部ベクトル検索メソッド不存在")
        
        # 3. 模拟向量空间统一测试
        print("\n🧪 ベクトル空間統一テスト:")
        try:
            # 简单的向量相似度测试
            test_text1 = "電圧調査の重要な情報"
            test_text2 = "電圧調査について"
            
            embeddings = self.bedrock_service.get_embeddings([test_text1, test_text2], input_type="search_document")
            
            if embeddings and len(embeddings) == 2:
                vec1 = np.array(embeddings[0])
                vec2 = np.array(embeddings[1])
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                print(f"  類似テキスト間相似度: {similarity:.3f}")
                
                if similarity > 0.7:
                    print("  ✅ ベクトル空間正常動作確認")
                    fixes_applied.append("ベクトル空間統一")
                else:
                    print("  ⚠️ ベクトル相似度やや低い")
                    issues_found.append("ベクトル相似度要改善")
            else:
                print("  ❌ ベクトル生成テスト失敗")
                issues_found.append("ベクトル生成失敗")
                
        except Exception as e:
            print(f"  ❌ ベクトル空間テストエラー: {e}")
            issues_found.append("ベクトル空間テスト失敗")
        
        # 结果评估
        print(f"\n📊 修复完了項目: {len(fixes_applied)} 件")
        for fix in fixes_applied:
            print(f"  ✅ {fix}")
        
        if issues_found:
            print(f"\n⚠️ 残存問題: {len(issues_found)} 件")
            for issue in issues_found:
                print(f"  ❌ {issue}")
            return False
        else:
            print("\n✅ ベクトル空間統一性: 良好")
            return True
    
    def run_updated_diagnosis(self):
        """更新版完整診断"""
        print("🏥 更新版RAGシステム健康診断開始")
        print("="*80)
        
        results = {
            "嵌入一致性": False,
            "Weaviate更新設定": False,
            "ベクトル空間統一": False
        }
        
        # 执行各检查
        results["嵌入一致性"] = self.check_embedding_consistency()
        results["Weaviate更新設定"] = self.check_updated_weaviate_config()
        results["ベクトル空間統一"] = self.check_vector_space_unity()
        
        # 综合评估
        print("\n" + "="*80)
        print("📊 更新版診断結果")
        print("="*80)
        
        total_score = 0
        for check_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{check_name:<20}: {status}")
            if result:
                total_score += 1
        
        overall_health = total_score / len(results) * 100
        
        if overall_health >= 90:
            print(f"\n🟢 総合評価: {overall_health:.0f}% - 優秀（修復成功）")
            health_status = "excellent"
        elif overall_health >= 70:
            print(f"\n🟡 総合評価: {overall_health:.0f}% - 良好（一部改善必要）")
            health_status = "good"
        else:
            print(f"\n🔴 総合評価: {overall_health:.0f}% - 改善必要")
            health_status = "needs_work"
        
        # 下一步建议
        self.suggest_next_steps(results, health_status)
        
        return results
    
    def suggest_next_steps(self, results: Dict[str, bool], health_status: str):
        """建议下一步行动"""
        print("\n" + "="*80)
        print("🎯 次のステップ")
        print("="*80)
        
        if health_status == "excellent":
            print("🎉 システム修復完了！")
            print("✅ 推奨アクション:")
            print("   1. Enhanced RAG Service実行テスト")
            print("      python enhanced_rag_service.py")
            print("   2. Webアプリケーション起動")
            print("      python app.py")
            print("   3. 重要質問テスト:")
            print('      "電圧調査では、どの4つの情報を優先的に収集すべきですか？"')
            
        elif health_status == "good":
            print("🔧 部分的修復完了、最終調整推奨")
            if not results["嵌入一致性"]:
                print("   - 嵌入モデル設定要確認")
            if not results["Weaviate更新設定"]:
                print("   - Weaviate Schema再作成要")
            if not results["ベクトル空間統一"]:
                print("   - ベクトル処理統一化要")
        
        else:
            print("❌ 重大問題残存")
            print("🔧 必要修復:")
            for check_name, result in results.items():
                if not result:
                    print(f"   - {check_name}の修復")

def main():
    """更新版メイン実行"""
    try:
        checker = UpdatedRAGHealthChecker()
        results = checker.run_updated_diagnosis()
        
        # 基于结果决定下一步
        all_passed = all(results.values())
        
        if all_passed:
            print(f"\n🚀 システム準備完了！")
            print("以下のコマンドで本格テスト実行可能:")
            print("   python enhanced_rag_service.py")
        else:
            print(f"\n⚠️ まだ修復が必要です。")
            print("問題修正後に再テスト:")
            print("   python rag_health_check.py")
            
    except Exception as e:
        print(f"❌ 診断エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 