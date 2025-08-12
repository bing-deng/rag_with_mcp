#!/usr/bin/env python3
"""
PDF内容调试工具 - 寻找第17页電圧調査内容
"""
import sys
import os
import json

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pdf_processor import PDFProcessor

def debug_pdf_content():
    print("🔍 PDF内容調査開始...")
    
    pdf_path = "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF文件不存在: {pdf_path}")
        return
    
    processor = PDFProcessor()
    
    try:
        # 1. 提取所有文本并查找相关内容
        print("\n🔍 1. 検索関連キーワード...")
        
        # 使用pdfplumber按页提取
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            print(f"📄 PDF总页数: {len(pdf.pages)}")
            
            # 搜索关键词
            search_terms = ["電圧調査", "電柱番号", "不具合状況", "発生時間帯", "発生範囲"]
            
            found_pages = {}
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        for term in search_terms:
                            if term in text:
                                if page_num not in found_pages:
                                    found_pages[page_num] = []
                                found_pages[page_num].append(term)
                                print(f"   ✅ ページ{page_num}: '{term}' 見つかった")
                except Exception as e:
                    print(f"   ⚠️ ページ{page_num}抽出エラー: {e}")
            
            # 2. 重点检查第17页
            print(f"\n🔍 2. ページ17詳細調査...")
            if len(pdf.pages) >= 17:
                try:
                    page17 = pdf.pages[16]  # 0-indexed
                    page17_text = page17.extract_text()
                    
                    if page17_text:
                        print(f"📄 ページ17内容長: {len(page17_text)}文字")
                        print(f"📄 ページ17内容プレビュー:")
                        print("=" * 50)
                        print(page17_text[:500])  # 前500字符
                        print("=" * 50)
                        
                        # 检查是否包含关键信息
                        key_info = ["電圧調査", "電柱番号", "不具合状況", "発生時間帯", "発生範囲"]
                        for info in key_info:
                            if info in page17_text:
                                print(f"✅ ページ17に'{info}'含まれている")
                                # 找到相关句子
                                lines = page17_text.split('\n')
                                for line in lines:
                                    if info in line:
                                        print(f"   → {line.strip()}")
                            else:
                                print(f"❌ ページ17に'{info}'含まれていない")
                    else:
                        print("❌ ページ17のテキスト抽出失敗")
                        
                except Exception as e:
                    print(f"❌ ページ17処理エラー: {e}")
            else:
                print("❌ PDFにページ17がありません")
            
            # 3. 显示所有找到关键词的页面完整内容
            print(f"\n🔍 3. 関連ページ完全内容表示...")
            for page_num, terms in found_pages.items():
                print(f"\n📄 ページ{page_num} (キーワード: {', '.join(terms)}):")
                print("-" * 60)
                try:
                    page_text = pdf.pages[page_num-1].extract_text()
                    if page_text:
                        print(page_text)
                    else:
                        print("テキスト抽出できませんでした")
                except Exception as e:
                    print(f"エラー: {e}")
                print("-" * 60)
    
    except Exception as e:
        print(f"❌ PDF処理エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pdf_content() 