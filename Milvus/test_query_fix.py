#!/usr/bin/env python3
"""
测试查询修复
验证 LLaMA 查询引擎是否正确使用指定的集合
"""

from llama_query import LLaMAQueryEngine

def test_collection_fix():
    """测试集合名称修复"""
    print("🧪 测试集合名称修复")
    print("=" * 50)
    
    # 测试 kandenko_website 集合
    print("🎯 测试集合: kandenko_website")
    engine = LLaMAQueryEngine(
        model_type='ollama', 
        model_name='llama3.2:3b',
        collection_name='kandenko_website'
    )
    
    print(f"✅ LLaMAQueryEngine 集合名称: {engine.collection_name}")
    print(f"✅ MilvusQueryEngine 集合名称: {engine.milvus_engine.collection_name}")
    
    # 连接测试
    if engine.connect_to_milvus():
        print("✅ 成功连接到 Milvus")
        
        # 测试一个简单的日文查询
        test_queries = [
            "関電工",  # Kandenko的日文名称
            "株式会社関電工",  # 完整的公司名称
            "電力工事",  # 电力工程
            "サービス"   # 服务
        ]
        
        print(f"\n🔍 测试查询:")
        for query in test_queries:
            try:
                # 只做向量搜索，不调用 LLaMA（避免输出太长）
                results = engine.milvus_engine.basic_search(query, top_k=3)
                
                print(f"\n📝 查询: '{query}'")
                if results:
                    print(f"   找到 {len(results)} 个相关结果")
                    for i, result in enumerate(results[:2]):  # 只显示前2个
                        print(f"   {i+1}. [{result['content_type']}] {result['title'][:30]}... (相似度: {result['similarity']:.3f})")
                else:
                    print("   ❌ 没有找到相关结果")
                    
            except Exception as e:
                print(f"   ❌ 查询失败: {e}")
        
        engine.milvus_engine.disconnect()
    else:
        print("❌ 连接 Milvus 失败")

if __name__ == "__main__":
    test_collection_fix() 