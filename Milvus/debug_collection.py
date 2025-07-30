#!/usr/bin/env python3
"""
调试集合内容
检查 kandenko_website 集合中的实际数据
"""

from pymilvus import connections, Collection, utility
import pandas as pd

def debug_collection(collection_name="kandenko_website"):
    """调试集合内容"""
    print(f"🔍 调试集合: {collection_name}")
    print("=" * 50)
    
    try:
        # 连接到 Milvus
        connections.connect("default", host="localhost", port="19530")
        print("✅ 已连接到 Milvus")
        
        # 检查集合是否存在
        if not utility.has_collection(collection_name):
            print(f"❌ 集合 {collection_name} 不存在")
            print("📋 可用集合:")
            collections = utility.list_collections()
            for i, col in enumerate(collections, 1):
                print(f"  {i}. {col}")
            return
        
        # 加载集合
        collection = Collection(collection_name)
        collection.load()
        
        # 获取基本信息
        num_entities = collection.num_entities
        print(f"📊 集合统计:")
        print(f"   记录总数: {num_entities}")
        
        if num_entities == 0:
            print("❌ 集合为空！")
            return
        
        # 查询前10条记录
        print(f"\n🔍 前10条记录预览:")
        results = collection.query(
            expr="id >= 0",
            output_fields=["id", "url", "title", "content", "content_type"],
            limit=10
        )
        
        for i, record in enumerate(results):
            print(f"\n📝 记录 {i+1}:")
            print(f"   ID: {record['id']}")
            print(f"   URL: {record['url'][:80]}...")
            print(f"   标题: {record['title'][:50]}...")
            print(f"   类型: {record['content_type']}")
            print(f"   内容: {record['content'][:100]}...")
        
        # 统计URL分布
        print(f"\n🌐 URL 分布统计:")
        url_results = collection.query(
            expr="id >= 0",
            output_fields=["url"],
            limit=100
        )
        
        url_domains = {}
        for record in url_results:
            url = record['url']
            if 'kandenko.co.jp' in url:
                domain = 'kandenko.co.jp'
            elif 'fsa.go.jp' in url:
                domain = 'fsa.go.jp (金融厅)'
            else:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                except:
                    domain = 'unknown'
            
            url_domains[domain] = url_domains.get(domain, 0) + 1
        
        for domain, count in url_domains.items():
            print(f"   {domain}: {count} 条记录")
        
        # 检查是否有 Kandenko 内容
        kandenko_count = sum(1 for r in url_results if 'kandenko.co.jp' in r['url'])
        print(f"\n🎯 Kandenko 相关记录: {kandenko_count} / {len(url_results)}")
        
        if kandenko_count == 0:
            print("❌ 没有找到 Kandenko 网站的内容！")
            print("🔧 需要重新爬取数据")
        else:
            print("✅ 找到 Kandenko 内容")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

def clean_and_recreate():
    """清理并重新创建集合"""
    print(f"\n🧹 清理操作")
    print("-" * 30)
    
    try:
        connections.connect("default", host="localhost", port="19530")
        
        collection_name = "kandenko_website"
        if utility.has_collection(collection_name):
            print(f"🗑️  删除现有集合: {collection_name}")
            utility.drop_collection(collection_name)
            print("✅ 集合已删除")
        else:
            print(f"ℹ️  集合 {collection_name} 不存在，无需删除")
        
        print("🚀 准备重新爬取 Kandenko 网站")
        return True
        
    except Exception as e:
        print(f"❌ 清理失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 Milvus 集合调试工具")
    print("=" * 60)
    
    # 调试当前集合
    debug_collection("kandenko_website")
    
    # 询问是否要清理重建
    print(f"\n" + "=" * 60)
    choice = input("是否要清理并重新爬取？(y/N): ").strip().lower()
    
    if choice == 'y':
        if clean_and_recreate():
            print(f"\n🎯 现在运行以下命令重新爬取:")
            print(f"python fix_and_crawl.py")
            print(f"\n或者直接运行:")
            print(f"python website_crawler.py")
    else:
        print("取消操作")

if __name__ == "__main__":
    main() 