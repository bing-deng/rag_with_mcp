#!/usr/bin/env python3
"""
模型管理器 - 单例模式
避免重复加载Sentence-Transformers和Ollama连接
"""

import threading
import requests
from typing import Optional, List
import os

# 尝试导入Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers 未安装")

class ModelManager:
    """单例模式的模型管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._sentence_model = None
            self._ollama_base_url = "http://localhost:11434"
            self._ollama_available = False
            # 向量缓存
            self._vector_cache = {}
            self._cache_size_limit = 1000  # 缓存1000个向量
            self._initialized = True
            print("🔧 初始化模型管理器 (带缓存)...")
    
    def get_sentence_model(self) -> Optional[SentenceTransformer]:
        """获取Sentence-Transformers模型 - 超级优化版"""
        if self._sentence_model is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                print("🔧 加载语义向量化模型 (仅此一次)...")
                # 使用更小更快的模型，明确禁用GPU检测
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制CPU
                
                self._sentence_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    device='cpu'
                )
                # 进一步优化参数
                self._sentence_model.max_seq_length = 128  # 大幅减少序列长度
                
                # 禁用一些可能的开销
                if hasattr(self._sentence_model, '_modules'):
                    for module in self._sentence_model._modules.values():
                        if hasattr(module, 'gradient_checkpointing'):
                            module.gradient_checkpointing = False
                
                # 预热模型
                print("🔥 预热向量模型...")
                _ = self._sentence_model.encode("预热", normalize_embeddings=True)
                print("✅ 语义模型加载完成并缓存")
            except Exception as e:
                print(f"❌ 语义模型加载失败: {e}")
                return None
        return self._sentence_model
    
    def check_ollama_service(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self._ollama_base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                if available_models:
                    self._ollama_available = True
                    print(f"✅ Ollama 服务可用，模型: {available_models}")
                    return True
            return False
        except Exception as e:
            self._ollama_available = False
            print(f"❌ Ollama 服务不可用: {e}")
            return False
    
    def is_ollama_available(self) -> bool:
        """Ollama是否可用"""
        return self._ollama_available
    
    def generate_with_ollama(self, prompt: str, model_name: str = 'llama3.2:3b', max_tokens: int = 500) -> str:
        """使用Ollama生成回答 - 超级优化版"""
        # 只在第一次或失败时检查服务
        if not self._ollama_available:
            if not self.check_ollama_service():
                return "❌ Ollama服务不可用"
        
        try:
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.05,    # 进一步降低随机性
                    "top_k": 5,            # 更严格限制候选词
                    "top_p": 0.8,          # 更严格的核采样
                    "repeat_penalty": 1.05, # 减少重复检查开销
                    "num_thread": 4,       # 限制线程数
                    "num_ctx": 1024,       # 减少上下文窗口
                    "batch_size": 1,       # 单批处理
                    "seed": 42             # 固定种子，减少随机性开销
                }
            }
            
            response = requests.post(
                f"{self._ollama_base_url}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"❌ Ollama 请求失败: {response.status_code}"
                
        except Exception as e:
            return f"❌ Ollama 生成失败: {e}"
    
    def text_to_vector(self, text: str) -> Optional[List[float]]:
        """文本转向量 - 带缓存"""
        # 检查缓存
        if text in self._vector_cache:
            return self._vector_cache[text]
        
        model = self.get_sentence_model()
        if model:
            try:
                embedding = model.encode(text, normalize_embeddings=True)
                vector = embedding.tolist()
                
                # 缓存向量，但限制缓存大小
                if len(self._vector_cache) >= self._cache_size_limit:
                    # 简单的FIFO缓存清理
                    oldest_key = next(iter(self._vector_cache))
                    del self._vector_cache[oldest_key]
                
                self._vector_cache[text] = vector
                return vector
            except Exception as e:
                print(f"❌ 向量化失败: {e}")
                return None
        return None
    
    def warm_up(self):
        """预热模型"""
        print("🔥 预热模型...")
        # 预热Sentence-Transformers
        if self.get_sentence_model():
            self.text_to_vector("test")
            print("✅ 向量模型预热完成")
        
        # 检查Ollama
        if self.check_ollama_service():
            print("✅ Ollama服务检查完成")

# 全局单例实例
_model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """获取模型管理器单例"""
    return _model_manager