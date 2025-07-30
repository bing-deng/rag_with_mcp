#!/usr/bin/env python3
"""
检查特定页面内容
验证 outline.html 页面是否被正确爬取和存储
"""

from pymilvus import connections, Collection
from query_milvus import MilvusQueryEngine
from html_to_milvus import HTMLToMilvusProcessor
import requests

def check_page_in_milvus(url_pattern="outline.html"):
    """检查特定页面是否在 Milvus 中"""
    print(f"🔍 检查页面: {url_pattern}")
    print("=" * 50)
    
    try:
        engine = MilvusQueryEngine(collection_name='kandenko_website')
        if not engine.connect():
            return False
        
        # 查询包含 outline 的 URL
        results = engine.collection.query(
            expr="id >= 0",
            output_fields=["id", "url", "title", "content", "content_type"],
            limit=1000
        )
        
        outline_pages = [r for r in results if url_pattern in r['url']]
        
        print(f"📊 总记录数: {len(results)}")
        print(f"🎯 包含 '{url_pattern}' 的页面: {len(outline_pages)}")
        
        if outline_pages:
            print(f"\n📄 找到的 outline 页面:")
            for i, page in enumerate(outline_pages):
                print(f"\n{i+1}. URL: {page['url']}")
                print(f"   标题: {page['title'][:100]}...")
                print(f"   类型: {page['content_type']}")
                print(f"   内容: {page['content'][:200]}...")
                
                # 检查是否包含设立时间信息
                content = page['content']
                if '1944' in content or '設立' in content or '会社設立' in content:
                    print(f"   ✅ 包含设立信息: 发现关键词")
                else:
                    print(f"   ❌ 未发现设立信息")
        else:
            print(f"❌ 没有找到包含 '{url_pattern}' 的页面")
            
            # 显示一些示例页面
            print(f"\n📋 现有页面示例:")
            for i, page in enumerate(results[:5]):
                print(f"  {i+1}. {page['url']}")
        
        engine.disconnect()
        return len(outline_pages) > 0
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def fetch_and_test_page(url="https://www.kandenko.co.jp/company/outline.html"):
    """直接获取并测试页面解析"""
    print(f"\n🌐 直接测试页面: {url}")
    print("-" * 50)
    
    try:
        processor = HTMLToMilvusProcessor()
        
        # 获取页面
        html_content = processor.fetch_html(url)
        if not html_content:
            print("❌ 无法获取页面内容")
            return False
        
        print(f"✅ 成功获取页面 ({len(html_content)} 字符)")
        
        # 解析页面
        content_blocks = processor.parse_html(html_content, base_url=url)
        print(f"✅ 解析出 {len(content_blocks)} 个内容块")
        
        # 查找包含设立信息的块
        setup_blocks = []
        for block in content_blocks:
            content = block['content']
            if any(keyword in content for keyword in ['1944', '設立', '会社設立', '創立']):
                setup_blocks.append(block)
        
        print(f"\n🎯 包含设立信息的内容块: {len(setup_blocks)}")
        
        for i, block in enumerate(setup_blocks):
            print(f"\n📝 内容块 {i+1}:")
            print(f"   类型: {block['content_type']}")
            print(f"   标题: {block['title'][:50]}...")
            print(f"   内容: {block['content']}")
        
        if setup_blocks:
            print(f"\n✅ 页面包含设立信息，应该能被正确查询")
        else:
            print(f"\n❌ 页面不包含设立信息关键词")
            
            # 显示所有内容块供调试
            print(f"\n🔍 所有内容块预览:")
            for i, block in enumerate(content_blocks[:10]):
                print(f"  {i+1}. [{block['content_type']}] {block['content'][:100]}...")
        
        return len(setup_blocks) > 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_for_setup_info():
    """测试搜索设立信息"""
    print(f"\n🔍 测试搜索设立相关信息")
    print("-" * 50)
    
    engine = MilvusQueryEngine(collection_name='kandenko_website')
    if not engine.connect():
        return
    
    search_terms = [
        "1944",
        "設立", 
        "会社設立",
        "創立",
        "設立年",
        "会社概要",
        "outline"
    ]
    
    for term in search_terms:
        print(f"\n📝 搜索: '{term}'")
        try:
            results = engine.basic_search(term, top_k=3)
            if results:
                print(f"   找到 {len(results)} 个结果")
                for i, result in enumerate(results):
                    print(f"   {i+1}. [{result['content_type']}] {result['title'][:40]}... (相似度: {result['similarity']:.3f})")
            else:
                print(f"   ❌ 没有找到结果")
        except Exception as e:
            print(f"   ❌ 搜索失败: {e}")
    
    engine.disconnect()

def main():
    """主函数"""
    print("🕵️ Kandenko 设立信息调试工具")
    print("=" * 60)
    
    # 1. 检查 Milvus 中是否有 outline 页面
    has_outline = check_page_in_milvus("outline.html")
    
    # 2. 直接测试页面获取和解析
    page_has_info = fetch_and_test_page()
    
    # 3. 测试搜索设立相关信息
    test_search_for_setup_info()
    
    # 总结
    print(f"\n📋 调试总结:")
    print(f"=" * 50)
    if has_outline:
        print(f"✅ outline.html 页面已在 Milvus 中")
    else:
        print(f"❌ outline.html 页面未在 Milvus 中")
    
    if page_has_info:
        print(f"✅ 页面包含设立信息")
    else:
        print(f"❌ 页面不包含设立信息")
    
    if not has_outline:
        print(f"\n🔧 建议解决方案:")
        print(f"1. 重新运行爬虫，确保覆盖 outline.html 页面")
        print(f"2. 检查爬虫的深度和页面限制设置")
        print(f"3. 运行: python3 fix_encoding_crawl.py")
    elif not page_has_info:
        print(f"\n🔧 建议解决方案:")
        print(f"1. 检查页面解析逻辑")
        print(f"2. 可能需要改进内容提取规则")

if __name__ == "__main__":
    main() 