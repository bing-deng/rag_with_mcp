#!/usr/bin/env python3
"""
测试 outline.html 页面解析
找出为什么设立信息没有被提取
"""

import requests
from bs4 import BeautifulSoup
import re

def analyze_outline_page():
    """分析 outline.html 页面结构"""
    url = "https://www.kandenko.co.jp/company/outline.html"
    
    print("🔍 分析 outline.html 页面结构")
    print("=" * 60)
    
    try:
        # 获取页面
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'ja,en-US,en;q=0.5',
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if response.encoding is None or response.encoding.lower() in ['iso-8859-1', 'ascii']:
            response.encoding = response.apparent_encoding or 'utf-8'
        
        print(f"✅ 获取页面成功 ({len(response.text)} 字符)")
        
        soup = BeautifulSoup(response.text, 'html.parser', from_encoding='utf-8')
        
        # 1. 查找包含 "1944" 的内容
        print(f"\n🎯 查找包含 '1944' 的内容:")
        text_content = soup.get_text()
        if '1944' in text_content:
            print("✅ 页面包含 '1944'")
            
            # 找到包含1944的具体位置
            lines = text_content.split('\n')
            for i, line in enumerate(lines):
                if '1944' in line:
                    print(f"   第{i+1}行: {line.strip()}")
        else:
            print("❌ 页面不包含 '1944'")
        
        # 2. 查找包含 "設立" 的内容
        print(f"\n🎯 查找包含 '設立' 的内容:")
        if '設立' in text_content:
            print("✅ 页面包含 '設立'")
            
            lines = text_content.split('\n')
            for i, line in enumerate(lines):
                if '設立' in line:
                    print(f"   第{i+1}行: {line.strip()}")
        else:
            print("❌ 页面不包含 '設立'")
        
        # 3. 分析HTML结构
        print(f"\n🏗️ HTML结构分析:")
        print(f"   标题数量: h1({len(soup.find_all('h1'))}), h2({len(soup.find_all('h2'))}), h3({len(soup.find_all('h3'))})")
        print(f"   段落数量: {len(soup.find_all('p'))}")
        print(f"   表格数量: {len(soup.find_all('table'))}")
        print(f"   列表数量: ul({len(soup.find_all('ul'))}), ol({len(soup.find_all('ol'))})")
        print(f"   div数量: {len(soup.find_all('div'))}")
        print(f"   dt/dd数量: dt({len(soup.find_all('dt'))}), dd({len(soup.find_all('dd'))})") 
        
        # 4. 检查表格内容
        tables = soup.find_all('table')
        if tables:
            print(f"\n📊 表格内容分析:")
            for i, table in enumerate(tables):
                table_text = table.get_text(strip=True)
                print(f"   表格 {i+1}: {table_text[:100]}...")
                if '1944' in table_text or '設立' in table_text:
                    print(f"   ✅ 表格 {i+1} 包含设立信息！")
                    print(f"   完整内容: {table_text}")
        
        # 5. 检查定义列表 (dt/dd)
        dts = soup.find_all('dt')
        dds = soup.find_all('dd')
        if dts or dds:
            print(f"\n📝 定义列表分析:")
            for i, dt in enumerate(dts):
                dt_text = dt.get_text(strip=True)
                dd = dds[i] if i < len(dds) else None
                dd_text = dd.get_text(strip=True) if dd else ""
                
                print(f"   {i+1}. {dt_text}: {dd_text}")
                
                if '1944' in f"{dt_text} {dd_text}" or '設立' in f"{dt_text} {dd_text}":
                    print(f"   ✅ 定义列表 {i+1} 包含设立信息！")
        
        # 6. 查找所有包含设立相关信息的标签
        print(f"\n🔍 查找所有包含设立信息的HTML标签:")
        all_elements = soup.find_all(text=re.compile(r'(1944|設立|会社設立)'))
        for element in all_elements:
            parent = element.parent
            if parent:
                print(f"   标签: {parent.name}, 内容: {element.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_outline_page() 