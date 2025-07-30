#!/usr/bin/env python3
"""
测试修复后的HTML解析器
验证能否正确提取设立信息
"""

from html_to_milvus import HTMLToMilvusProcessor

def test_fixed_parsing():
    """测试修复后的解析器"""
    print("🧪 测试修复后的HTML解析器")
    print("=" * 60)
    
    processor = HTMLToMilvusProcessor()
    url = "https://www.kandenko.co.jp/company/outline.html"
    
    try:
        # 获取页面
        print("🌐 获取页面...")
        html_content = processor.fetch_html(url)
        if not html_content:
            print("❌ 获取页面失败")
            return False
        
        print(f"✅ 获取成功 ({len(html_content)} 字符)")
        
        # 解析页面
        print("\n🔍 解析页面（修复版）...")
        content_blocks = processor.parse_html(html_content, base_url=url)
        
        print(f"✅ 解析完成，提取了 {len(content_blocks)} 个内容块")
        
        # 查找设立信息
        setup_blocks = []
        company_info_blocks = []
        definition_list_blocks = []
        
        for i, block in enumerate(content_blocks):
            content = block['content']
            content_type = block['content_type']
            
            # 分类内容块
            if content_type == 'definition_list':
                definition_list_blocks.append((i, block))
            elif content_type == 'company_info':
                company_info_blocks.append((i, block))
            
            # 查找包含设立信息的块
            if any(keyword in content for keyword in ['1944', '設立', '会社設立']):
                setup_blocks.append((i, block))
        
        # 显示结果
        print(f"\n📊 内容块统计:")
        print(f"   总内容块: {len(content_blocks)}")
        print(f"   定义列表块: {len(definition_list_blocks)}")
        print(f"   公司信息块: {len(company_info_blocks)}")
        print(f"   包含设立信息的块: {len(setup_blocks)}")
        
        # 显示定义列表内容
        if definition_list_blocks:
            print(f"\n📝 定义列表内容:")
            for i, (idx, block) in enumerate(definition_list_blocks):
                print(f"   块 {idx+1} [{block['content_type']}]:")
                print(f"   {block['content']}")
        
        # 显示公司信息内容
        if company_info_blocks:
            print(f"\n🏢 公司信息内容:")
            for i, (idx, block) in enumerate(company_info_blocks):
                print(f"   块 {idx+1} [{block['content_type']}]:")
                print(f"   {block['content']}")
        
        # 显示包含设立信息的内容
        if setup_blocks:
            print(f"\n🎯 包含设立信息的内容:")
            for i, (idx, block) in enumerate(setup_blocks):
                print(f"   ✅ 块 {idx+1} [{block['content_type']}]:")
                print(f"   {block['content']}")
        else:
            print(f"\n❌ 仍然没有找到设立信息")
        
        return len(setup_blocks) > 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_after_fix():
    """测试修复后能否搜索到设立信息"""
    print(f"\n🔍 测试搜索功能")
    print("-" * 40)
    
    from query_milvus import MilvusQueryEngine
    
    # 注意：这个测试需要重新爬取数据后才能生效
    print("⚠️  注意：需要重新爬取数据后才能测试搜索功能")
    print("运行: python3 fix_encoding_crawl.py")

if __name__ == "__main__":
    success = test_fixed_parsing()
    
    if success:
        print(f"\n🎉 修复成功！")
        print(f"现在解析器能正确提取设立信息了")
        print(f"\n🔧 下一步：重新爬取数据")
        print(f"运行: python3 fix_encoding_crawl.py")
    else:
        print(f"\n❌ 修复不完整，需要进一步调试")
    
    test_search_after_fix() 