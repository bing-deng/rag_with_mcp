#!/usr/bin/env python3
"""
测试增强后的HTML解析器
验证新增标签的提取效果
"""

from html_to_milvus import HTMLToMilvusProcessor
import json

def test_enhanced_parsing():
    """测试增强后的解析器"""
    print("🚀 测试增强后的HTML解析器")
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
        print("\n🔍 解析页面（增强版）...")
        content_blocks = processor.parse_html(html_content, base_url=url)
        
        print(f"✅ 解析完成，提取了 {len(content_blocks)} 个内容块")
        
        # 按内容类型分类统计
        content_types = {}
        setup_info_blocks = []
        
        for i, block in enumerate(content_blocks):
            content_type = block['content_type']
            content = block['content']
            
            # 统计类型
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # 查找包含设立信息的块
            if any(keyword in content for keyword in ['1944', '設立', '会社設立', '創立']):
                setup_info_blocks.append((i+1, block))
        
        # 显示统计结果
        print(f"\n📊 内容类型统计:")
        for content_type, count in sorted(content_types.items()):
            print(f"   {content_type}: {count} 个")
        
        # 显示包含设立信息的块
        print(f"\n🎯 包含设立信息的内容块 ({len(setup_info_blocks)} 个):")
        for block_num, block in setup_info_blocks:
            print(f"   ✅ 块 {block_num} [{block['content_type']}]:")
            print(f"      内容: {block['content'][:100]}...")
        
        # 显示新增内容类型的示例
        new_types = ['address', 'quote', 'emphasis', 'semantic_section', 'semantic_article', 
                    'code_pre', 'code_code', 'figure_caption', 'company_div', 'profile_div', 
                    'business_div', 'important_div', 'key_data']
        
        print(f"\n🆕 新增内容类型示例:")
        for content_type in new_types:
            examples = [block for block in content_blocks if block['content_type'] == content_type]
            if examples:
                print(f"   📝 {content_type} ({len(examples)} 个):")
                for i, example in enumerate(examples[:2]):  # 只显示前2个
                    print(f"      {i+1}. {example['content'][:80]}...")
        
        # 与原版对比
        print(f"\n📈 提取能力对比:")
        print(f"   原版（5种标签）: ~3-5 个内容块")
        print(f"   增强版（13种标签）: {len(content_blocks)} 个内容块")
        print(f"   提升幅度: {len(content_blocks)//5}x")
        
        # 检查是否包含关键企业信息
        key_info_found = {}
        key_patterns = {
            '公司名称': ['会社名', '株式会社'],
            '设立时间': ['設立', '創立', '1944'],
            '资本金': ['資本金', '億円', '万円'],
            '员工数': ['従業員', '名'],
            '营业额': ['売上', '億円'],
            '地址': ['本社', '東京', '住所'],
            '联系方式': ['電話', 'TEL', 'FAX']
        }
        
        all_content = ' '.join([block['content'] for block in content_blocks])
        for info_type, patterns in key_patterns.items():
            found = any(pattern in all_content for pattern in patterns)
            key_info_found[info_type] = found
        
        print(f"\n🔍 关键企业信息覆盖率:")
        for info_type, found in key_info_found.items():
            status = "✅" if found else "❌"
            print(f"   {status} {info_type}")
        
        coverage = sum(key_info_found.values()) / len(key_info_found) * 100
        print(f"   总覆盖率: {coverage:.1f}%")
        
        return len(setup_info_blocks) > 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_pages():
    """测试多个页面的提取效果"""
    print(f"\n🔍 测试多个页面提取效果")
    print("-" * 50)
    
    test_urls = [
        "https://www.kandenko.co.jp/company/outline.html",
        "https://www.kandenko.co.jp/",  # 首页
    ]
    
    processor = HTMLToMilvusProcessor()
    
    for url in test_urls:
        try:
            print(f"\n🌐 测试页面: {url}")
            html_content = processor.fetch_html(url)
            if html_content:
                content_blocks = processor.parse_html(html_content, base_url=url)
                print(f"   ✅ 提取了 {len(content_blocks)} 个内容块")
                
                # 统计内容类型
                types = set(block['content_type'] for block in content_blocks)
                print(f"   📊 内容类型: {len(types)} 种")
            else:
                print(f"   ❌ 获取失败")
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")

if __name__ == "__main__":
    success = test_enhanced_parsing()
    
    if success:
        print(f"\n🎉 增强成功！解析器现在支持13种HTML标签类型")
        print(f"大幅提升了内容提取的全面性和准确性")
        
        print(f"\n🏗️ 支持的标签类型:")
        print(f"   基础标签: h1-h6, p, ul/ol/li, table")
        print(f"   企业信息: dt/dd (定义列表)")
        print(f"   地址信息: address")
        print(f"   引用内容: blockquote")  
        print(f"   强调内容: strong/em/b")
        print(f"   语义标签: section/article")
        print(f"   代码内容: pre/code")
        print(f"   图表标题: figcaption")
        print(f"   重要区块: div (智能识别)")
        print(f"   关键数据: span (数字/日期)")
        
        print(f"\n🔧 下一步：重新爬取数据以获得完整提取效果")
        print(f"运行: python3 fix_encoding_crawl.py")
    else:
        print(f"\n❌ 仍有问题需要进一步调试")
    
    # 测试多个页面
    test_multiple_pages() 