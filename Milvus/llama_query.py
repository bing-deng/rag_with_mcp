#!/usr/bin/env python3
"""
基于 LLaMA 的智能查询系统
结合 Milvus 向量搜索和 LLaMA 大语言模型，实现 RAG (检索增强生成) 功能
"""

import json
import time
from typing import List, Dict, Optional
from query_milvus import MilvusQueryEngine

# 尝试导入不同的 LLaMA 实现
try:
    # 优先尝试 transformers 库（适用于 Hugging Face 模型）
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
    print("✅ 发现 transformers 库，支持 Hugging Face LLaMA 模型")
except ImportError:
    HAS_TRANSFORMERS = False

try:
    # 尝试 llama-cpp-python（适用于量化模型）
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
    print("✅ 发现 llama-cpp-python，支持量化 LLaMA 模型")
except ImportError:
    HAS_LLAMA_CPP = False

try:
    # 尝试 Ollama（本地 LLaMA 服务）
    import requests
    HAS_OLLAMA = True
    print("✅ 支持 Ollama 本地 LLaMA 服务")
except ImportError:
    HAS_OLLAMA = False

class LLaMAQueryEngine:
    """结合 LLaMA 的智能查询引擎"""
    
    def __init__(self, model_type='ollama', model_name='llama3.2:3b', collection_name='web_content', **kwargs):
        """
        初始化 LLaMA 查询引擎
        
        Args:
            model_type: 模型类型 ('ollama', 'transformers', 'llama_cpp')
            model_name: 模型名称
            collection_name: Milvus 集合名称
            **kwargs: 模型特定参数
        """
        self.model_type = model_type
        self.model_name = model_name
        self.collection_name = collection_name
        self.model = None
        self.tokenizer = None
        
        # 初始化 Milvus 查询引擎
        self.milvus_engine = MilvusQueryEngine(collection_name=collection_name)
        
        # 初始化 LLaMA 模型
        self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        """初始化 LLaMA 模型"""
        if self.model_type == 'ollama':
            self._initialize_ollama(**kwargs)
        elif self.model_type == 'transformers':
            self._initialize_transformers(**kwargs)
        elif self.model_type == 'llama_cpp':
            self._initialize_llama_cpp(**kwargs)
        else:
            print(f"❌ 不支持的模型类型: {self.model_type}")
    
    def _initialize_ollama(self, base_url='http://localhost:11434', **kwargs):
        """初始化 Ollama 模型"""
        if not HAS_OLLAMA:
            print("❌ 需要安装 requests 库来使用 Ollama")
            return
        
        self.ollama_base_url = base_url
        
        # 检查 Ollama 服务是否可用
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                print(f"✅ Ollama 服务可用，可用模型: {available_models}")
                
                if self.model_name not in available_models:
                    print(f"⚠️  模型 {self.model_name} 未安装")
                    print(f"   安装命令: ollama pull {self.model_name}")
                else:
                    print(f"✅ 模型 {self.model_name} 已就绪")
            else:
                print("❌ Ollama 服务响应异常")
        except Exception as e:
            print(f"❌ 无法连接到 Ollama 服务: {e}")
            print("   请确保 Ollama 服务已启动: ollama serve")
    
    def _initialize_transformers(self, device='auto', **kwargs):
        """初始化 Transformers 模型"""
        if not HAS_TRANSFORMERS:
            print("❌ 需要安装 transformers 和 torch 库")
            return
        
        try:
            print(f"🔧 加载 Transformers 模型: {self.model_name}")
            
            # 自动选择设备
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map=device,
                **kwargs
            )
            
            print(f"✅ Transformers 模型加载完成 (设备: {device})")
            
        except Exception as e:
            print(f"❌ Transformers 模型加载失败: {e}")
    
    def _initialize_llama_cpp(self, model_path=None, **kwargs):
        """初始化 llama-cpp-python 模型"""
        if not HAS_LLAMA_CPP:
            print("❌ 需要安装 llama-cpp-python 库")
            return
        
        if not model_path:
            print("❌ llama-cpp 需要指定 model_path 参数")
            return
        
        try:
            print(f"🔧 加载 llama-cpp 模型: {model_path}")
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=2048,  # 上下文长度
                n_threads=4,  # 线程数
                **kwargs
            )
            
            print("✅ llama-cpp 模型加载完成")
            
        except Exception as e:
            print(f"❌ llama-cpp 模型加载失败: {e}")
    
    def connect_to_milvus(self):
        """连接到 Milvus"""
        return self.milvus_engine.connect()
    
    def _generate_with_ollama(self, prompt: str, max_tokens: int = 500) -> str:
        """使用 Ollama 生成回答"""
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
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
    
    def _generate_with_transformers(self, prompt: str, max_tokens: int = 500) -> str:
        """使用 Transformers 生成回答"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除原始 prompt，只保留生成的部分
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"❌ Transformers 生成失败: {e}"
    
    def _generate_with_llama_cpp(self, prompt: str, max_tokens: int = 500) -> str:
        """使用 llama-cpp 生成回答"""
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["Human:", "Assistant:", "\n\n"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            return f"❌ llama-cpp 生成失败: {e}"
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """生成回答"""
        if self.model_type == 'ollama':
            return self._generate_with_ollama(prompt, max_tokens)
        elif self.model_type == 'transformers':
            return self._generate_with_transformers(prompt, max_tokens)
        elif self.model_type == 'llama_cpp':
            return self._generate_with_llama_cpp(prompt, max_tokens)
        else:
            return "❌ 模型未初始化"
    
    def rag_query(self, question: str, top_k: int = 5, max_tokens: int = 500) -> Dict:
        """RAG 查询：检索+生成"""
        print(f"🤖 RAG 查询: '{question}'")
        
        # 1. 从 Milvus 检索相关内容
        print("🔍 第一步：向量搜索检索相关内容...")
        search_results = self.milvus_engine.basic_search(question, top_k=top_k)
        
        if not search_results:
            return {
                "question": question,
                "retrieved_contexts": [],
                "generated_answer": "抱歉，没有找到相关的背景信息来回答您的问题。",
                "sources": []
            }
        
        # 2. 构建上下文
        contexts = []
        sources = []
        for i, result in enumerate(search_results):
            context = f"文档{i+1}: {result['content']}"
            contexts.append(context)
            sources.append({
                "id": result['id'],
                "title": result['title'],
                "url": result['url'],
                "content_type": result['content_type'],
                "similarity": result['score']
            })
        
        # 3. 构建 prompt
        context_text = "\n\n".join(contexts)
        
        prompt = f"""请基于以下背景信息回答用户的问题。如果背景信息中没有相关内容，请诚实地说明。

背景信息：
{context_text}

用户问题：{question}

请提供准确、有用的回答："""

        # 4. 生成回答
        print("🤖 第二步：LLaMA 生成智能回答...")
        generated_answer = self.generate_response(prompt, max_tokens)
        
        return {
            "question": question,
            "retrieved_contexts": contexts,
            "generated_answer": generated_answer,
            "sources": sources,
            "model_info": {
                "type": self.model_type,
                "name": self.model_name
            }
        }
    
    def interactive_chat(self):
        """交互式聊天模式"""
        print("🤖 LLaMA + Milvus 智能问答系统")
        print("=" * 50)
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'stats' 查看数据库统计")
        print("-" * 50)
        
        if not self.connect_to_milvus():
            print("❌ 无法连接到 Milvus，退出聊天模式")
            return
        
        chat_history = []
        
        try:
            while True:
                question = input("\n🙋 您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    break
                elif question.lower() == 'stats':
                    stats = self.milvus_engine.get_statistics()
                    print(f"\n📊 数据库统计:")
                    print(f"   总记录数: {stats.get('total_records', 'N/A')}")
                    if 'content_type_distribution' in stats:
                        print("   内容类型分布:")
                        for ct, count in list(stats['content_type_distribution'].items())[:5]:
                            print(f"     {ct}: {count}")
                    continue
                elif not question:
                    continue
                
                # 执行 RAG 查询
                start_time = time.time()
                result = self.rag_query(question, top_k=3, max_tokens=500)
                end_time = time.time()
                
                # 显示结果
                print(f"\n🤖 AI 回答:")
                print("-" * 40)
                print(result['generated_answer'])
                
                print(f"\n📚 参考来源 (相似度):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. [{source['content_type']}] {source['title']} ({source['similarity']:.3f})")
                
                print(f"\n⏱️  响应时间: {end_time - start_time:.2f}秒")
                
                # 保存聊天历史
                chat_history.append({
                    "question": question,
                    "answer": result['generated_answer'],
                    "timestamp": time.time()
                })
        
        except KeyboardInterrupt:
            print("\n\n👋 用户中断")
        
        finally:
            self.milvus_engine.disconnect()
            
            # 保存聊天历史
            if chat_history:
                filename = f"chat_history_{int(time.time())}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(chat_history, f, ensure_ascii=False, indent=2)
                print(f"💾 聊天历史已保存到: {filename}")

def setup_ollama_guide():
    """Ollama 安装和设置指南"""
    print("📖 Ollama 设置指南")
    print("=" * 40)
    print("1. 安装 Ollama:")
    print("   macOS: brew install ollama")
    print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   Windows: 下载 https://ollama.ai/download/windows")
    print()
    print("2. 启动 Ollama 服务:")
    print("   ollama serve")
    print()
    print("3. 下载 LLaMA 模型:")
    print("   ollama pull llama3.2:3b    # 3B 模型 (推荐)")
    print("   ollama pull llama2         # 基础模型")
    print("   ollama pull llama2:13b     # 13B 模型")
    print("   ollama pull codellama      # 代码专用")
    print()
    print("4. 测试模型:")
    print("   ollama run llama3.2:3b")

def demo_different_models():
    """演示不同 LLaMA 实现的使用"""
    print("🎭 不同 LLaMA 实现演示")
    print("=" * 50)
    
    # 测试查询
    test_question = "什么是金融监管？"
    
    # 1. Ollama 演示
    print("\n1️⃣  Ollama 演示:")
    try:
        ollama_engine = LLaMAQueryEngine(model_type='ollama', model_name='llama3.2:3b', collection_name='kandenko_website')
        if ollama_engine.connect_to_milvus():
            result = ollama_engine.rag_query(test_question, top_k=2)
            print(f"问题: {result['question']}")
            print(f"回答: {result['generated_answer'][:200]}...")
            ollama_engine.milvus_engine.disconnect()
    except Exception as e:
        print(f"❌ Ollama 演示失败: {e}")
    
    # 2. Transformers 演示 (需要大量内存)
    print("\n2️⃣  Transformers 演示:")
    print("⚠️  需要大量 GPU 内存，跳过演示")
    print("   如需使用，请确保有足够的 GPU 资源")
    
    # 3. llama-cpp 演示
    print("\n3️⃣  llama-cpp 演示:")
    print("⚠️  需要下载量化模型文件，跳过演示")
    print("   如需使用，请下载 .gguf 格式的模型文件")

def main():
    """主函数"""
    print("🤖 LLaMA + Milvus 智能查询系统")
    print("=" * 60)
    
    # 检查可用的实现
    available_implementations = []
    if HAS_OLLAMA:
        available_implementations.append("Ollama (推荐)")
    if HAS_TRANSFORMERS:
        available_implementations.append("Transformers")
    if HAS_LLAMA_CPP:
        available_implementations.append("llama-cpp-python")
    
    if not available_implementations:
        print("❌ 没有找到可用的 LLaMA 实现")
        print("\n推荐安装 Ollama:")
        setup_ollama_guide()
        return
    
    print(f"✅ 可用的实现: {', '.join(available_implementations)}")
    
    # 选择实现方式
    print("\n请选择使用方式:")
    print("1. 交互式问答 (Ollama)")
    print("2. 查看 Ollama 设置指南") 
    print("3. 演示不同实现")
    print("0. 退出")
    
    choice = input("\n请选择 (0-3): ").strip()
    
    if choice == "1":
        # 交互式问答
        engine = LLaMAQueryEngine(model_type='ollama', model_name='llama3.2:3b', collection_name='kandenko_website')
        engine.interactive_chat()
    
    elif choice == "2":
        # 设置指南
        setup_ollama_guide()
    
    elif choice == "3":
        # 演示不同实现
        demo_different_models()
    
    elif choice == "0":
        print("👋 再见！")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main() 