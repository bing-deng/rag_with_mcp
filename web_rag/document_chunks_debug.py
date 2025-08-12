#!/usr/bin/env python3
"""
检查文档分块是否包含電圧調査信息
"""
import sys
import os

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from enhanced_rag_service import EnhancedWebRAGService

def debug_document_chunks():
    print("🔍 文書チャンク調査開始...")
    
    try:
        service = EnhancedWebRAGService()
        
        # 检查向量数据库中的文档
        print("\n🔍 Weaviateデータベース内容調査...")
        
        collection_name = "Document"  # 假设使用默认collection名
        collection = service.weaviate_client.client.collections.get(collection_name)
        
        # 获取所有文档
        response = collection.query.fetch_objects(limit=100)
        
        print(f"📦 データベース内文書数: {len(response.objects)}")
        
        # 搜索包含電圧調査的文档块
        search_terms = ["電圧調査", "電柱番号", "不具合状況", "発生時間帯", "発生範囲"]
        
        found_chunks = []
        
        for i, obj in enumerate(response.objects):
            content = obj.properties.get('content', '')
            title = obj.properties.get('title', '')
            
            for term in search_terms:
                if term in content or term in title:
                    found_chunks.append({
                        'index': i,
                        'title': title,
                        'content': content,
                        'matched_term': term
                    })
                    break
        
        print(f"✅ 関連チャンク発見: {len(found_chunks)}件")
        
        for chunk in found_chunks:
            print(f"\n📄 チャンク{chunk['index']}: {chunk['title']}")
            print(f"🔍 マッチ語: {chunk['matched_term']}")
            print(f"📝 内容プレビュー:")
            print("-" * 40)
            print(chunk['content'][:300])
            print("-" * 40)
        
        # 特别搜索"電圧調査について"
        print(f"\n🎯 '電圧調査について'専用検索...")
        for i, obj in enumerate(response.objects):
            content = obj.properties.get('content', '')
            if '電圧調査について' in content or '電圧調査では' in content:
                print(f"\n🎯 発見！チャンク{i}:")
                print(f"📝 完全内容:")
                print("=" * 50)
                print(content)
                print("=" * 50)
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_document_chunks() 