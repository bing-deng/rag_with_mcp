#!/usr/bin/env python3
"""
修复并重新爬取 Kandenko 网站
解决字段长度限制问题
"""

import os
import sys
from website_crawler import WebsiteCrawler, create_kandenko_config

def main():
    """修复并重新爬取"""
    print("🔧 Kandenko 网站爬取修复脚本")
    print("=" * 50)
    print("修复内容:")
    print("✅ 增加 content 字段长度限制: 5000 → 15000 字符")
    print("✅ 添加内容截断保护机制")
    print("✅ 处理 URL 和标题长度限制")
    print("-" * 50)
    
    # 创建专用配置
    config = create_kandenko_config()
    
    print(f"🎯 目标: {config.base_url}")
    print(f"📊 配置:")
    print(f"   集合名称: {config.collection_name}")
    print(f"   最大页面: {config.max_pages}")
    print(f"   爬取深度: {config.max_depth}")
    print(f"   并发数: {config.concurrent_workers}")
    print(f"   请求间隔: {config.delay_between_requests}秒")
    
    # 确认开始
    print(f"\n🚀 准备重新爬取 Kandenko 网站")
    confirm = input("继续？(y/N): ").strip().lower()
    
    if confirm != 'y':
        print("❌ 用户取消")
        return
    
    # 创建爬虫并开始
    crawler = WebsiteCrawler(config)
    
    try:
        print(f"\n🕷️  开始爬取...")
        report = crawler.crawl()
        
        # 显示爬取结果
        print(f"\n📋 爬取完成:")
        print(f"   耗时: {report['stats']['duration_seconds']:.1f} 秒")
        print(f"   成功页面: {report['stats']['pages_crawled']}")
        print(f"   失败页面: {report['stats']['pages_failed']}")
        print(f"   内容块数: {report['stats']['content_blocks_extracted']}")
        print(f"   成功率: {report['stats']['success_rate']:.1%}")
        
        # 保存爬取报告
        report_file = crawler.save_report()
        
        # 保存到 Milvus（已修复长度问题）
        print(f"\n💾 保存到 Milvus（修复后版本）...")
        if crawler.save_to_milvus():
            print(f"🎉 成功！数据已保存到集合: {config.collection_name}")
            
            # 提供后续操作建议
            print(f"\n🎯 接下来可以:")
            print(f"1. 🔍 Python 查询:")
            print(f"   python query_milvus.py")
            print(f"   # 然后选择集合: {config.collection_name}")
            
            print(f"\n2. 🤖 LLaMA 智能问答:")
            print(f"   python llama_query.py")
            print(f"   # 自动使用集合: {config.collection_name}")
            
            print(f"\n3. 🧠 智能查询系统 (推荐):")
            print(f"   python smart_query.py")
            print(f"   # 自动识别并选择集合")
            
            print(f"\n📝 示例查询:")
            print(f"   - 'Kandenko的主要业务领域是什么？'")
            print(f"   - '公司的技术实力如何？'")
            print(f"   - '电力工程相关的服务有哪些？'")
            
        else:
            print(f"❌ 保存失败，请检查错误信息")
            return 1
    
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断")
        return 1
    except Exception as e:
        print(f"❌ 爬取失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 




# 🔌 已断开连接
# 🎉 成功！数据已保存到集合: kandenko_website

# 🎯 接下来可以:
# 1. 🔍 Python 查询:
#    python query_milvus.py
#    # 然后选择集合: kandenko_website

# 2. 🤖 LLaMA 智能问答:
#    python llama_query.py
#    # 自动使用集合: kandenko_website

# 3. 🧠 智能查询系统 (推荐):
#    python smart_query.py
#    # 自动识别并选择集合

# 📝 示例查询:
#    - 'Kandenko的主要业务领域是什么？'
#    - '公司的技术实力如何？'
#    - '电力工程相关的服务有哪些？    