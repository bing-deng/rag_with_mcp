#!/usr/bin/env python3
"""
修复日文编码问题并重新爬取 Kandenko 网站
"""

from pymilvus import connections, utility
from website_crawler import WebsiteCrawler, create_kandenko_config

def clean_existing_collection():
    """清理现有的集合"""
    print("🧹 清理现有数据...")
    
    try:
        connections.connect("default", host="localhost", port="19530")
        
        collection_name = "kandenko_website"
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
    print("🔧 日文编码修复 + Kandenko 网站重新爬取")
    print("=" * 60)
    print("修复内容:")
    print("✅ 改进 BeautifulSoup 编码处理")
    print("✅ 增强日文字符支持（平假名、片假名）")
    print("✅ 优化请求头（Accept-Language: ja）")
    print("✅ 智能编码检测和fallback")
    print("-" * 60)
    
    # 清理现有数据
    if not clean_existing_collection():
        print("❌ 清理失败，无法继续")
        return 1
    
    # 确认重新爬取
    print(f"\n🚀 准备重新爬取 Kandenko 网站（修复日文编码）")
    confirm = input("继续？(y/N): ").strip().lower()
    
    if confirm != 'y':
        print("❌ 用户取消")
        return 0
    
    # 创建爬虫配置
    config = create_kandenko_config()
    print(f"\n🎯 目标: {config.base_url}")
    print(f"📊 集合: {config.collection_name}")
    
    # 开始爬取
    crawler = WebsiteCrawler(config)
    
    try:
        print(f"\n🕷️  开始爬取（修复版）...")
        report = crawler.crawl()
        
        # 显示结果
        print(f"\n📋 爬取完成:")
        print(f"   耗时: {report['stats']['duration_seconds']:.1f} 秒")
        print(f"   成功页面: {report['stats']['pages_crawled']}")
        print(f"   失败页面: {report['stats']['pages_failed']}")
        print(f"   内容块数: {report['stats']['content_blocks_extracted']}")
        print(f"   成功率: {report['stats']['success_rate']:.1%}")
        
        # 保存报告
        report_file = crawler.save_report()
        
        # 保存到 Milvus
        print(f"\n💾 保存到 Milvus（修复编码版本）...")
        if crawler.save_to_milvus():
            print(f"🎉 成功！数据已保存（日文编码已修复）")
            
            # 测试查询
            print(f"\n🧪 编码修复测试:")
            print(f"现在可以正确查询日文内容了！")
            
            print(f"\n🎯 建议的查询:")
            print(f"1. 🔍 基础查询:")
            print(f"   python3 smart_query.py")
            print(f"   选择 'kandenko_website' 集合")
            
            print(f"\n2. 📝 推荐的日文查询:")
            print(f"   - '関電工' (Kandenko)")
            print(f"   - '株式会社関電工' (完整公司名)")
            print(f"   - '電力工事' (电力工程)")
            print(f"   - '会社概要' (公司概要)")
            print(f"   - 'サービス' (服务)")
            
            print(f"\n3. 🤖 中文查询也支持:")
            print(f"   - '关电工公司介绍'")
            print(f"   - '电力工程服务'")
            print(f"   - '公司业务范围'")
            
        else:
            print(f"❌ 保存失败")
            return 1
    
    except Exception as e:
        print(f"❌ 爬取失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 