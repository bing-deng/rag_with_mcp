#!/usr/bin/env python3
"""
智能查询工具
自动检测并选择不同的 Milvus 集合进行查询
支持 Python 向量搜索和 LLaMA 智能问答
"""

from pymilvus import connections, utility
from query_milvus import MilvusQueryEngine, print_search_results
from llama_query import LLaMAQueryEngine
from typing import List, Dict

class SmartQueryManager:
    """智能查询管理器"""
    
    def __init__(self, host='localhost', port='19530'):
        self.host = host
        self.port = port
        self.available_collections = []
        
    def connect_and_scan(self) -> bool:
        """连接并扫描可用的集合"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ 成功连接到 Milvus: {self.host}:{self.port}")
            
            # 获取所有集合
            self.available_collections = utility.list_collections()
            print(f"📋 发现 {len(self.available_collections)} 个集合:")
            
            for i, collection_name in enumerate(self.available_collections, 1):
                print(f"  {i}. {collection_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def select_collection(self) -> str:
        """选择集合"""
        if not self.available_collections:
            print("❌ 没有可用的集合")
            return None
        
        if len(self.available_collections) == 1:
            collection_name = self.available_collections[0]
            print(f"🎯 自动选择唯一集合: {collection_name}")
            return collection_name
        
        print("\n请选择要查询的集合:")
        for i, collection_name in enumerate(self.available_collections, 1):
            print(f"  {i}. {collection_name}")
        
        while True:
            try:
                choice = input(f"\n请选择集合 (1-{len(self.available_collections)}): ").strip()
                index = int(choice) - 1
                
                if 0 <= index < len(self.available_collections):
                    selected = self.available_collections[index]
                    print(f"✅ 已选择集合: {selected}")
                    return selected
                else:
                    print("❌ 无效选择，请重试")
                    
            except ValueError:
                print("❌ 请输入数字")
    
    def python_query_mode(self, collection_name: str):
        """Python 向量查询模式"""
        print(f"\n🔍 Python 向量查询模式 - 集合: {collection_name}")
        print("=" * 50)
        
        engine = MilvusQueryEngine(collection_name=collection_name)
        if not engine.connect():
            return
        
        try:
            # 显示集合统计
            stats = engine.get_statistics()
            print(f"\n📊 集合统计:")
            print(f"   集合名称: {stats.get('collection_name')}")
            print(f"   总记录数: {stats.get('total_records')}")
            
            if 'content_type_distribution' in stats:
                print("   内容类型分布:")
                for ct, count in list(stats['content_type_distribution'].items())[:5]:
                    print(f"     {ct}: {count}")
            
            # 交互式查询
            while True:
                print(f"\n🔍 查询选项:")
                print("1. 基础语义搜索")
                print("2. 高级过滤搜索")
                print("3. 查看更多统计信息")
                print("0. 返回主菜单")
                
                choice = input("\n请选择 (0-3): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    query = input("请输入搜索查询: ").strip()
                    if query:
                        results = engine.basic_search(query, top_k=5)
                        print_search_results(results, "搜索结果")
                
                elif choice == "2":
                    query = input("请输入搜索查询: ").strip()
                    if not query:
                        continue
                        
                    print("可选过滤条件:")
                    content_type = input("内容类型 (如: paragraph, heading_h1): ").strip() or None
                    min_words_str = input("最小字数: ").strip()
                    min_words = int(min_words_str) if min_words_str.isdigit() else None
                    
                    results = engine.filtered_search(query, content_type, None, min_words, top_k=5)
                    print_search_results(results, "高级搜索结果")
                
                elif choice == "3":
                    stats = engine.get_statistics()
                    print(f"\n📊 详细统计信息:")
                    print(f"集合名称: {stats.get('collection_name')}")
                    print(f"总记录数: {stats.get('total_records')}")
                    print(f"向量维度: {stats.get('dimension')}")
                    
                    if 'content_type_distribution' in stats:
                        print("\n内容类型分布:")
                        for ct, count in stats['content_type_distribution'].items():
                            print(f"  {ct}: {count}")
                
                else:
                    print("❌ 无效选择")
        
        finally:
            engine.disconnect()
    
    def llama_query_mode(self, collection_name: str):
        """LLaMA 智能问答模式"""
        print(f"\n🤖 LLaMA 智能问答模式 - 集合: {collection_name}")
        print("=" * 50)
        
        # 检查 LLaMA 模型
        try:
            engine = LLaMAQueryEngine(
                model_type='ollama', 
                model_name='llama3.2:3b',
                collection_name=collection_name
            )
            
            if not engine.connect_to_milvus():
                return
            
            print("💬 开始智能问答 (输入 'quit' 退出):")
            print("-" * 30)
            
            while True:
                question = input("\n🙋 您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    break
                elif not question:
                    continue
                
                try:
                    # 执行 RAG 查询
                    result = engine.rag_query(question, top_k=3)
                    
                    print(f"\n🤖 AI 回答:")
                    print("-" * 30)
                    print(result['generated_answer'])
                    
                    print(f"\n📚 参考来源:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. [{source['content_type']}] {source['title'][:50]}... ({source['similarity']:.3f})")
                
                except Exception as e:
                    print(f"❌ 查询失败: {e}")
            
        except Exception as e:
            print(f"❌ LLaMA 模式初始化失败: {e}")
            print("请确保:")
            print("1. Ollama 服务正在运行: ollama serve")
            print("2. 模型已下载: ollama pull llama3.2:3b")
        
        finally:
            if 'engine' in locals():
                engine.milvus_engine.disconnect()
    
    def main_menu(self):
        """主菜单"""
        if not self.connect_and_scan():
            return
        
        collection_name = self.select_collection()
        if not collection_name:
            return
        
        while True:
            print(f"\n🎯 当前集合: {collection_name}")
            print("=" * 50)
            print("请选择查询模式:")
            print("1. 🔍 Python 向量搜索 (快速、精确)")
            print("2. 🤖 LLaMA 智能问答 (自然语言对话)")
            print("3. 🔄 切换集合")
            print("0. 退出")
            
            choice = input("\n请选择 (0-3): ").strip()
            
            if choice == "0":
                print("👋 再见！")
                break
            elif choice == "1":
                self.python_query_mode(collection_name)
            elif choice == "2":
                self.llama_query_mode(collection_name)
            elif choice == "3":
                new_collection = self.select_collection()
                if new_collection:
                    collection_name = new_collection
            else:
                print("❌ 无效选择")

def main():
    """主函数"""
    print("🧠 智能查询系统")
    print("=" * 60)
    print("支持多种查询方式:")
    print("• Python 向量搜索: 快速精确的相似性检索")
    print("• LLaMA 智能问答: 自然语言问答系统") 
    print("• 自动集合识别: 支持多个数据源")
    print("-" * 60)
    
    manager = SmartQueryManager()
    manager.main_menu()

if __name__ == "__main__":
    main() 