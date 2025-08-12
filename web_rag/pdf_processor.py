"""
PDF文档处理器 - 提取和预处理PDF内容
"""
import PyPDF2
import pdfplumber
import re
from typing import List, Dict, Any
import os

class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self):
        self.chunk_size = 500  # 每个文档块的字符数
        self.overlap = 50      # 重叠字符数
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """使用PyPDF2提取PDF文本"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n=== 页面 {page_num + 1} ===\n"
                        text += page_text
                
                print(f"✅ PyPDF2提取完成 - 总长度: {len(text)} 字符")
                return text.strip()
                
        except Exception as e:
            print(f"❌ PyPDF2提取失败: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """使用pdfplumber提取PDF文本（更准确）"""
        try:
            text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"\n=== 页面 {page_num + 1} ===\n"
                        text += page_text
                
                print(f"✅ pdfplumber提取完成 - 总长度: {len(text)} 字符")
                return text.strip()
                
        except Exception as e:
            print(f"❌ pdfplumber提取失败: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """清理和规范化文本"""
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff.,!?;:()\-/]', '', text)
        
        # 规范化换行
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def split_into_chunks(self, text: str, source_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """将文本分割成适合向量化的块"""
        if not text:
            return []
        
        chunks = []
        text_length = len(text)
        start = 0
        chunk_id = 1
        
        while start < text_length:
            # 计算结束位置
            end = min(start + self.chunk_size, text_length)
            
            # 如果不是最后一块，尝试在句号或换行处分割
            if end < text_length:
                for punct in ['。', '.\n', '！', '？', '\n\n']:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos > start + 100:  # 确保块不会太小
                        end = punct_pos + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 50:  # 忽略太短的块
                chunk = {
                    "content": chunk_text,
                    "title": f"文档块 {chunk_id}",
                    "source": source_info.get("filename", "unknown") if source_info else "unknown",
                    "chunk_id": chunk_id,
                    "start_pos": start,
                    "end_pos": end,
                    "metadata": source_info or {}
                }
                chunks.append(chunk)
                chunk_id += 1
            
            # 计算下一个起始位置（带重叠）
            start = max(start + 1, end - self.overlap)
        
        print(f"✅ 文本分块完成 - 共 {len(chunks)} 个块")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """完整处理PDF文档"""
        print(f"📄 开始处理PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF文件不存在: {pdf_path}")
            return []
        
        # 获取文件信息
        filename = os.path.basename(pdf_path)
        file_size = os.path.getsize(pdf_path)
        
        print(f"📊 文件信息: {filename} ({file_size:,} 字节)")
        
        # 尝试两种提取方法
        text = self.extract_text_pdfplumber(pdf_path)
        if not text or len(text) < 100:
            print("⚠️ pdfplumber结果不佳，尝试PyPDF2...")
            text = self.extract_text_pypdf2(pdf_path)
        
        if not text or len(text) < 100:
            print("❌ 无法提取PDF内容")
            return []
        
        # 清理文本
        clean_text = self.clean_text(text)
        
        # 创建源信息
        source_info = {
            "filename": filename,
            "file_path": pdf_path,
            "file_size": file_size,
            "total_length": len(clean_text),
            "category": "manual",
            "provider": "takusoukun"
        }
        
        # 分割成块
        chunks = self.split_into_chunks(clean_text, source_info)
        
        print(f"✅ PDF处理完成: {len(chunks)} 个文档块")
        return chunks

def test_pdf_processor():
    """测试PDF处理器"""
    pdf_path = "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
    
    processor = PDFProcessor()
    chunks = processor.process_pdf(pdf_path)
    
    if chunks:
        print(f"\n=== 处理结果概览 ===")
        print(f"总块数: {len(chunks)}")
        print(f"前3个块的标题和长度:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"{i+1}. {chunk['title']} - {len(chunk['content'])} 字符")
            print(f"   内容预览: {chunk['content'][:100]}...")
        
        return chunks
    else:
        print("PDF处理失败")
        return []

if __name__ == "__main__":
    test_pdf_processor()
