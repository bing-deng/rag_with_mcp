#!/usr/bin/env python3
"""
智能内容分块器
解决当前内容分割过于细粒度的问题，提升RAG检索质量
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize

# 确保NLTK数据可用
try:
    sent_tokenize("test")
except LookupError:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')

@dataclass
class ChunkMetadata:
    """分块元数据"""
    original_url: str
    chunk_index: int
    total_chunks: int
    content_type: str
    quality_score: float
    language: str
    word_count: int

class SmartChunker:
    """智能内容分块器"""
    
    def __init__(self, 
                 min_chunk_size: int = 150,
                 max_chunk_size: int = 800, 
                 overlap_size: int = 50,
                 target_chunk_size: int = 400):
        """
        Args:
            min_chunk_size: 最小块大小（字符）
            max_chunk_size: 最大块大小（字符）
            overlap_size: 重叠大小（字符）
            target_chunk_size: 目标块大小（字符）
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.target_chunk_size = target_chunk_size
        
    def chunk_content(self, content: str, metadata: Dict = None) -> List[Tuple[str, ChunkMetadata]]:
        """
        智能分块内容
        
        Args:
            content: 原始内容
            metadata: 内容元数据
            
        Returns:
            分块后的内容和元数据列表
        """
        if not content or len(content) < self.min_chunk_size:
            return []
        
        # 1. 预处理：清理和标准化
        cleaned_content = self._preprocess_content(content)
        
        # 2. 质量评估
        quality_score = self._assess_content_quality(cleaned_content)
        if quality_score < 0.3:  # 质量太低，跳过
            return []
        
        # 3. 语言检测
        language = self._detect_language(cleaned_content)
        
        # 4. 选择分块策略
        if language == 'ja':
            chunks = self._chunk_japanese_content(cleaned_content)
        elif language in ['zh', 'zh-cn']:
            chunks = self._chunk_chinese_content(cleaned_content)
        else:
            chunks = self._chunk_english_content(cleaned_content)
        
        # 5. 后处理：重叠和质量过滤
        processed_chunks = self._post_process_chunks(chunks)
        
        # 6. 生成元数据
        result = []
        for i, chunk in enumerate(processed_chunks):
            chunk_metadata = ChunkMetadata(
                original_url=metadata.get('url', '') if metadata else '',
                chunk_index=i,
                total_chunks=len(processed_chunks),
                content_type=metadata.get('content_type', 'text') if metadata else 'text',
                quality_score=self._assess_content_quality(chunk),
                language=language,
                word_count=len(chunk.split())
            )
            result.append((chunk, chunk_metadata))
        
        return result
    
    def _preprocess_content(self, content: str) -> str:
        """预处理内容"""
        # 移除多余空白
        content = re.sub(r'\s+', ' ', content.strip())
        
        # 移除常见的噪声模式
        noise_patterns = [
            r'^(首页|主页|トップページ|Home)$',
            r'^(导航|ナビゲーション|Navigation).*',
            r'^(菜单|メニュー|Menu).*',
            r'^(页脚|フッター|Footer).*',
            r'^(版权|著作権|Copyright).*',
            r'^(联系我们|お問い合わせ|Contact)$',
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        return content.strip()
    
    def _assess_content_quality(self, content: str) -> float:
        """评估内容质量 (0-1分数)"""
        if not content:
            return 0.0
        
        score = 0.0
        
        # 1. 长度权重 (30%)
        length = len(content)
        if length < 50:
            length_score = 0.0
        elif length < 100:
            length_score = 0.3
        elif length < 200:
            length_score = 0.7
        else:
            length_score = 1.0
        score += length_score * 0.3
        
        # 2. 信息密度 (40%) - 非重复、有意义的内容比例
        words = content.split()
        if len(words) == 0:
            return 0.0
        
        unique_words = set(words)
        info_density = len(unique_words) / len(words)
        score += min(info_density * 2, 1.0) * 0.4
        
        # 3. 语言完整性 (30%) - 完整句子比例
        sentences = re.split(r'[.!?。！？]', content)
        complete_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
        if len(sentences) > 0:
            completeness = len(complete_sentences) / len(sentences)
        else:
            completeness = 0.0
        score += completeness * 0.3
        
        return min(score, 1.0)
    
    def _detect_language(self, content: str) -> str:
        """简单的语言检测"""
        # 日文检测（平假名、片假名）
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', content):
            return 'ja'
        
        # 中文检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        if chinese_chars > len(content) * 0.3:
            return 'zh'
        
        # 默认英文
        return 'en'
    
    def _chunk_japanese_content(self, content: str) -> List[str]:
        """日文内容分块"""
        # 日文句子分割
        sentences = re.split(r'[。！？]', content)
        sentences = [s.strip() + '。' for s in sentences if s.strip()]
        
        return self._merge_sentences_by_size(sentences)
    
    def _chunk_chinese_content(self, content: str) -> List[str]:
        """中文内容分块"""
        # 中文句子分割
        sentences = re.split(r'[。！？；]', content)
        sentences = [s.strip() + '。' for s in sentences if s.strip()]
        
        return self._merge_sentences_by_size(sentences)
    
    def _chunk_english_content(self, content: str) -> List[str]:
        """英文内容分块"""
        try:
            sentences = sent_tokenize(content)
        except:
            # fallback分割
            sentences = re.split(r'[.!?]+\s+', content)
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        return self._merge_sentences_by_size(sentences)
    
    def _merge_sentences_by_size(self, sentences: List[str]) -> List[str]:
        """根据大小合并句子"""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果当前块加上新句子不超过最大大小
            if len(current_chunk + sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                # 如果当前块达到最小大小，保存它
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # 开始新块
                current_chunk = sentence + " "
        
        # 处理最后一个块
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        elif chunks and len(current_chunk) > 0:
            # 如果最后一块太小，合并到前一块
            chunks[-1] += " " + current_chunk.strip()
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """后处理分块：添加重叠，质量过滤"""
        if not chunks:
            return []
        
        processed = []
        
        for i, chunk in enumerate(chunks):
            # 质量过滤
            if self._assess_content_quality(chunk) < 0.4:
                continue
            
            # 添加重叠（与前一块），但确保不超过最大长度
            if i > 0 and self.overlap_size > 0:
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.overlap_size:]
                potential_chunk = overlap_text + " " + chunk
                
                # 如果加上重叠后超过最大长度，则截断
                if len(potential_chunk) > self.max_chunk_size:
                    available_space = self.max_chunk_size - len(chunk) - 1  # -1 for space
                    if available_space > 0:
                        overlap_text = overlap_text[:available_space]
                        chunk = overlap_text + " " + chunk
                    # 如果没有空间，就不添加重叠
                else:
                    chunk = potential_chunk
            
            # 最终安全检查：确保不超过绝对最大长度
            if len(chunk) > 1900:  # 留100字符绝对安全缓冲
                chunk = chunk[:1900] + "..."
                print(f"⚠️  块过长，强制截断到1900字符")
            
            processed.append(chunk)
        
        return processed

def test_smart_chunker():
    """测试智能分块器"""
    chunker = SmartChunker()
    
    # 测试日文内容
    japanese_text = """
    株式会社関電工は、日本の電気工事業界をリードする企業です。
    当社は1944年に設立され、電力インフラの構築と保守に長年従事してきました。
    主要事業には送配電工事、電気設備工事、情報通信工事などがあります。
    関電工は技術革新を重視し、持続可能な社会の実現に向けて取り組んでいます。
    """
    
    chunks = chunker.chunk_content(japanese_text, {'url': 'test.com', 'content_type': 'paragraph'})
    
    print("🧪 智能分块器测试结果")
    print("=" * 50)
    print(f"原文长度: {len(japanese_text)} 字符")
    print(f"分块数量: {len(chunks)}")
    print()
    
    for i, (chunk, metadata) in enumerate(chunks):
        print(f"块 {i+1}:")
        print(f"  内容: {chunk[:100]}...")
        print(f"  长度: {len(chunk)} 字符")
        print(f"  质量分数: {metadata.quality_score:.2f}")
        print(f"  语言: {metadata.language}")
        print()

if __name__ == "__main__":
    test_smart_chunker()