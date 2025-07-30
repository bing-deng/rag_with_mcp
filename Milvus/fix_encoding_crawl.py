#!/usr/bin/env python3
"""
修复日文编码问题并重新爬取 Kandenko 网站
集成智能分块处理，提升内容质量
"""

import os
import sys
from pymilvus import connections, utility
from enhanced_website_crawler import EnhancedWebsiteCrawler, create_kandenko_smart_config

def clean_existing_collection():
    """清理现有的集合"""
    print("🧹 清理现有数据...")
    
    try:
        connections.connect("default", host="localhost", port="19530")
        
        collection_name = "kandenko_website_smart"
        if utility.has_collection(collection_name):
            print(f"🗑️  删除现有集合: {collection_name}")
            utility.drop_collection(collection_name)
            print("✅ 旧集合已删除")
        else:
            print("ℹ️  集合不存在，无需删除")
        
        return True
        
    except Exception as e:
        print(f"❌ 清理失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 日文编码修复 + 智能分块 + Kandenko 网站重新爬取")
    print("=" * 70)
    print("🆕 新功能:")
    print("✅ 智能内容分块处理 - 解决内容粒度过细问题")
    print("✅ 内容质量评估 - 自动过滤低质量片段") 
    print("✅ 多语言语义分块 - 按语义边界分割，保持完整性")
    print("✅ 重叠策略 - 防止信息丢失")
    print()
    print("🔧 编码修复:")
    print("✅ 改进 BeautifulSoup 编码处理")
    print("✅ 增强日文字符支持（平假名、片假名）")
    print("✅ 优化请求头（Accept-Language: ja）")
    print("✅ 智能编码检测和fallback")
    print("-" * 70)
    
    # 清理现有数据
    if not clean_existing_collection():
        print("❌ 清理失败，无法继续")
        return 1
    
    # 确认重新爬取
    print(f"\n🚀 准备使用智能分块重新爬取 Kandenko 网站")
    print("🎯 新功能预览:")
    print("   - 智能内容分块（150-800字符）")
    print("   - 自动质量评估过滤")
    print("   - 多语言语义分块")
    print("   - 内容重叠防信息丢失")
    print("\n🚀 自动开始智能爬取...")
    confirm = 'y'
    
    # 创建增强版爬虫配置
    config = create_kandenko_smart_config()
    print(f"\n🎯 目标: {config.base_url}")
    print(f"📊 集合: {config.collection_name}")
    print(f"⚙️  最大页面: {config.max_pages}")
    print(f"🧩 质量阈值: {config.quality_threshold}")
    
    # 开始智能爬取
    crawler = EnhancedWebsiteCrawler(config)
    
    try:
        print(f"\n🕷️  开始智能爬取（编码修复 + 智能分块）...")
        report = crawler.crawl()
        
        # 显示结果
        print(f"\n📋 智能爬取完成:")
        print(f"   耗时: {report['stats']['duration_seconds']:.1f} 秒")
        print(f"   成功页面: {report['stats']['pages_crawled']}")
        print(f"   失败页面: {report['stats']['pages_failed']}")
        print(f"   智能块数: {report['stats']['smart_chunks_created']}")
        print(f"   高质量块: {report['stats']['high_quality_chunks']}")
        print(f"   成功率: {report['stats']['success_rate']:.1%}")
        
        # 保存报告
        report_file = crawler.save_report()
        
        print(f"🎉 智能分块爬取成功！数据已保存到 Milvus")
        print(f"📋 详细报告: {report_file}")
        
        # 智能分块效果说明
        print(f"\n✨ 智能分块效果:")
        print(f"   🧩 内容块更大更完整（150-800字符）")
        print(f"   ⭐ 自动质量评估和过滤")
        print(f"   🌐 多语言语义理解")
        print(f"   🔗 信息重叠防丢失")
        
        # 测试建议
        print(f"\n🎯 测试新系统:")
        print(f"1. 🌐 Web UI测试:")
        print(f"   访问: http://localhost:5001")
        print(f"   集合名: kandenko_website_smart")
        
        print(f"\n2. 📝 推荐的日文查询:")
        print(f"   - '関電工の会社概要'")
        print(f"   - '電力工事サービス'")
        print(f"   - '採用情報'")
        print(f"   - '技術・サービス'")
        
        print(f"\n3. 🤖 中文查询:")
        print(f"   - '关电工公司介绍'")
        print(f"   - '电力工程服务'")
        print(f"   - '公司业务范围'")
        
        print(f"\n4. 📊 质量对比:")
        print(f"   旧版本: 细粒度片段，信息不完整")
        print(f"   新版本: 智能分块，信息完整连贯")
    
    except Exception as e:
        print(f"❌ 爬取失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 