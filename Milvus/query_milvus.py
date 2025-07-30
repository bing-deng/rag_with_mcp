#!/usr/bin/env python3
"""
Milvus 向量数据库查询系统
提供多种查询方式：基础向量搜索、高级过滤、聚合查询等
"""

import json
import time
from typing import List, Dict, Optional, Tuple
from pymilvus import connections, Collection, utility
from model_manager import get_model_manager

# 获取全局模型管理器
model_manager = get_model_manager()

# 尝试导入向量化模型
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers 未安装，部分功能受限")

class MilvusQueryEngine:
    """Milvus 查询引擎"""
    
    def __init__(self, host='localhost', port='19530', collection_name='web_content'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.dimension = 384
        
        # 使用全局模型管理器，避免重复加载
        self.model = None  # 不再在这里初始化模型
    
    def _get_output_fields(self):
        """动态获取输出字段列表"""
        if not self.collection:
            return ["url", "title", "content"]
            
        available_fields = [field.name for field in self.collection.schema.fields]
        output_fields = ["url", "title", "content"]
        
        # 添加可选字段（如果存在）
        optional_fields = ["content_type", "word_count", "timestamp", "category"]
        for field in optional_fields:
            if field in available_fields:
                output_fields.append(field)
        
        return output_fields
    
    def connect(self):
        """连接到 Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ 成功连接到 Milvus: {self.host}:{self.port}")
            
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                print(f"✅ 集合 {self.collection_name} 已加载")
                return True
            else:
                print(f"❌ 集合 {self.collection_name} 不存在")
                print("请先运行 html_to_milvus.py 创建数据")
                return False
                
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def text_to_vector(self, text: str) -> List[float]:
        """将文本转换为向量 - 使用全局模型管理器"""
        return model_manager.text_to_vector(text)
    
    def basic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """基础向量搜索"""
        print(f"🔍 基础搜索: '{query}'")
        
        query_vector = self.text_to_vector(query)
        if not query_vector:
            return []
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}
        }
        
        try:
            results = self.collection.search(
                [query_vector],
                "embedding",
                search_params,
                limit=top_k,
                output_fields=self._get_output_fields()
            )
            
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": float(hit.score),
                        "url": hit.entity.get("url"),
                        "title": hit.entity.get("title"),
                        "content": hit.entity.get("content"),
                        "content_type": hit.entity.get("content_type", "unknown"),
                        "word_count": hit.entity.get("word_count", 0),
                        "timestamp": hit.entity.get("timestamp", 0)
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def filtered_search(self, query: str, content_type: str = None, 
                       url_contains: str = None, min_words: int = None, 
                       top_k: int = 5) -> List[Dict]:
        """带过滤条件的高级搜索"""
        print(f"🎯 高级搜索: '{query}'")
        print(f"   过滤条件: content_type={content_type}, url_contains={url_contains}, min_words={min_words}")
        
        query_vector = self.text_to_vector(query)
        if not query_vector:
            return []
        
        # 构建过滤表达式
        filter_conditions = []
        
        if content_type:
            filter_conditions.append(f'content_type like "{content_type}%"')
        
        if url_contains:
            filter_conditions.append(f'url like "%{url_contains}%"')
        
        if min_words:
            filter_conditions.append(f'word_count >= {min_words}')
        
        expr = " and ".join(filter_conditions) if filter_conditions else None
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}
        }
        
        try:
            results = self.collection.search(
                [query_vector],
                "embedding",
                search_params,
                limit=top_k,
                expr=expr,
                output_fields=self._get_output_fields()
            )
            
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.id,
                        "score": float(hit.score),
                        "url": hit.entity.get("url"),
                        "title": hit.entity.get("title"),
                        "content": hit.entity.get("content"),
                        "content_type": hit.entity.get("content_type"),
                        "word_count": hit.entity.get("word_count"),
                        "timestamp": hit.entity.get("timestamp")
                    })
            
            return search_results
            
        except Exception as e:
            print(f"❌ 高级搜索失败: {e}")
            return []
    
    def get_by_ids(self, ids: List[int]) -> List[Dict]:
        """根据ID获取具体记录"""
        print(f"📋 根据ID获取记录: {ids}")
        
        try:
            # 构建ID过滤表达式
            id_list = ", ".join(map(str, ids))
            expr = f"id in [{id_list}]"
            
            results = self.collection.query(
                expr=expr,
                output_fields=self._get_output_fields()
            )
            
            return results
            
        except Exception as e:
            print(f"❌ ID查询失败: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        print("📊 获取统计信息...")
        
        try:
            stats = {
                "collection_name": self.collection.name,
                "total_records": self.collection.num_entities,
                "dimension": self.dimension
            }
            
            # 首先检查集合schema，确定可用字段
            schema = self.collection.schema
            available_fields = [field.name for field in schema.fields]
            
            # 只查询存在的字段
            query_fields = []
            if "content_type" in available_fields:
                query_fields.append("content_type")
            if "url" in available_fields:
                query_fields.append("url")
            if "title" in available_fields:
                query_fields.append("title")
            
            if not query_fields:
                # 如果没有这些字段，只返回基础统计
                return stats
            
            # 获取内容类型分布（如果字段存在）
            if "content_type" in query_fields:
                try:
                    content_types = self.collection.query(
                        expr="id >= 0",
                        output_fields=["content_type"],
                        limit=10000
                    )
                    
                    type_counts = {}
                    for item in content_types:
                        ct = item.get('content_type', 'unknown')
                        type_counts[ct] = type_counts.get(ct, 0) + 1
                    
                    stats["content_type_distribution"] = type_counts
                except Exception as e:
                    print(f"获取content_type统计失败: {e}")
            
            # 获取URL分布（如果字段存在）
            if "url" in query_fields:
                try:
                    urls = self.collection.query(
                        expr="id >= 0",
                        output_fields=["url"],
                        limit=10000
                    )
                    
                    url_counts = {}
                    for item in urls:
                        url = item.get('url', 'unknown')
                        # 简化URL显示
                        if len(url) > 50:
                            url = url[:50] + "..."
                        url_counts[url] = url_counts.get(url, 0) + 1
                    
                    stats["url_distribution"] = url_counts
                except Exception as e:
                    print(f"获取url统计失败: {e}")
            
            return stats
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def content_type_search(self, content_type: str, limit: int = 10) -> List[Dict]:
        """按内容类型浏览"""
        print(f"📁 浏览内容类型: {content_type}")
        
        try:
            expr = f'content_type == "{content_type}"'
            
            results = self.collection.query(
                expr=expr,
                output_fields=["url", "title", "content", "word_count"],
                limit=limit
            )
            
            return results
            
        except Exception as e:
            print(f"❌ 内容类型查询失败: {e}")
            return []
    
    def recent_content(self, days: int = 7, limit: int = 10) -> List[Dict]:
        """获取最近添加的内容"""
        print(f"📅 获取最近 {days} 天的内容")
        
        try:
            cutoff_time = int(time.time()) - (days * 24 * 3600)
            expr = f"timestamp >= {cutoff_time}"
            
            results = self.collection.query(
                expr=expr,
                output_fields=["url", "title", "content", "content_type", "timestamp"],
                limit=limit
            )
            
            # 按时间排序
            results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            return results
            
        except Exception as e:
            print(f"❌ 获取最近内容失败: {e}")
            return []
    
    def similarity_between_contents(self, id1: int, id2: int) -> float:
        """计算两个内容之间的相似度"""
        print(f"🔗 计算内容相似度: ID {id1} vs ID {id2}")
        
        try:
            # 获取两个内容的向量
            results = self.collection.query(
                expr=f"id in [{id1}, {id2}]",
                output_fields=["embedding", "content"]
            )
            
            if len(results) != 2:
                print("❌ 无法找到指定的内容")
                return 0.0
            
            # 计算余弦相似度
            import numpy as np
            
            vec1 = np.array(results[0]['embedding'])
            vec2 = np.array(results[1]['embedding'])
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            print(f"相似度: {similarity:.4f}")
            print(f"内容1: {results[0]['content'][:100]}...")
            print(f"内容2: {results[1]['content'][:100]}...")
            
            return float(similarity)
            
        except Exception as e:
            print(f"❌ 相似度计算失败: {e}")
            return 0.0
    
    def disconnect(self):
        """断开连接"""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        print("🔌 已断开连接")

def print_search_results(results: List[Dict], title: str = "搜索结果"):
    """格式化打印搜索结果"""
    print(f"\n📋 {title} (共 {len(results)} 条):")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. 【{result.get('content_type', 'unknown')}】")
        print(f"   相似度: {result.get('score', 0):.4f}")
        print(f"   标题: {result.get('title', 'N/A')}")
        print(f"   来源: {result.get('url', 'N/A')}")
        print(f"   字数: {result.get('word_count', 0)}")
        
        content = result.get('content', '')
        if len(content) > 150:
            content = content[:150] + "..."
        print(f"   内容: {content}")
        print("-" * 80)

def interactive_query_demo():
    """交互式查询演示"""
    engine = MilvusQueryEngine()
    
    if not engine.connect():
        return
    
    try:
        while True:
            print("\n🔍 Milvus 查询系统")
            print("=" * 40)
            print("1. 基础语义搜索")
            print("2. 高级过滤搜索") 
            print("3. 查看统计信息")
            print("4. 按内容类型浏览")
            print("5. 查看最近内容")
            print("6. ID精确查询")
            print("7. 内容相似度比较")
            print("0. 退出")
            
            choice = input("\n请选择功能 (0-7): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                query = input("请输入搜索查询: ").strip()
                if query:
                    results = engine.basic_search(query, top_k=5)
                    print_search_results(results, "基础搜索结果")
            
            elif choice == "2":
                query = input("请输入搜索查询: ").strip()
                content_type = input("内容类型过滤 (可选, 如: paragraph, heading): ").strip() or None
                url_contains = input("URL包含 (可选): ").strip() or None
                min_words_str = input("最小字数 (可选): ").strip()
                min_words = int(min_words_str) if min_words_str.isdigit() else None
                
                if query:
                    results = engine.filtered_search(
                        query, content_type, url_contains, min_words, top_k=5
                    )
                    print_search_results(results, "高级搜索结果")
            
            elif choice == "3":
                stats = engine.get_statistics()
                print(f"\n📊 数据库统计:")
                print(f"集合名称: {stats.get('collection_name', 'N/A')}")
                print(f"总记录数: {stats.get('total_records', 'N/A')}")
                print(f"向量维度: {stats.get('dimension', 'N/A')}")
                
                if 'content_type_distribution' in stats:
                    print("\n内容类型分布:")
                    for ct, count in stats['content_type_distribution'].items():
                        print(f"  {ct}: {count}")
                
                if 'url_distribution' in stats:
                    print("\nURL分布 (前10个):")
                    sorted_urls = sorted(stats['url_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)
                    for url, count in sorted_urls[:10]:
                        print(f"  {url}: {count}")
            
            elif choice == "4":
                content_type = input("请输入内容类型 (如: paragraph, heading_h1, list): ").strip()
                if content_type:
                    results = engine.content_type_search(content_type, limit=5)
                    print_search_results(results, f"内容类型: {content_type}")
            
            elif choice == "5":
                days_str = input("查看最近几天的内容 (默认7天): ").strip()
                days = int(days_str) if days_str.isdigit() else 7
                results = engine.recent_content(days, limit=5)
                print_search_results(results, f"最近 {days} 天的内容")
            
            elif choice == "6":
                ids_str = input("请输入ID列表 (用逗号分隔, 如: 1,2,3): ").strip()
                try:
                    ids = [int(x.strip()) for x in ids_str.split(',')]
                    results = engine.get_by_ids(ids)
                    print_search_results(results, f"ID查询结果")
                except ValueError:
                    print("❌ ID格式错误")
            
            elif choice == "7":
                try:
                    id1 = int(input("请输入第一个内容ID: ").strip())
                    id2 = int(input("请输入第二个内容ID: ").strip())
                    similarity = engine.similarity_between_contents(id1, id2)
                except ValueError:
                    print("❌ ID必须是数字")
            
            else:
                print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    
    finally:
        engine.disconnect()

def main():
    """主演示函数"""
    print("🔍 Milvus Python 查询演示")
    print("=" * 50)
    
    engine = MilvusQueryEngine()
    
    if not engine.connect():
        return
    
    try:
        # 1. 显示统计信息
        stats = engine.get_statistics()
        print(f"\n📊 数据库概况:")
        print(f"   总记录数: {stats.get('total_records', 'N/A')}")
        if 'content_type_distribution' in stats:
            print("   内容类型分布:")
            for ct, count in list(stats['content_type_distribution'].items())[:5]:
                print(f"     {ct}: {count}")
        
        # 2. 基础搜索演示
        test_queries = [
            "金融监管政策",
            "企业财务报告",
            "投资风险管理",
            "市场监督制度"
        ]
        
        for query in test_queries:
            print(f"\n🔍 搜索演示: '{query}'")
            results = engine.basic_search(query, top_k=3)
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. 【{result['content_type']}】相似度: {result['score']:.4f}")
                    print(f"      {result['content'][:100]}...")
            else:
                print("  未找到相关结果")
        
        # 3. 高级搜索演示
        print(f"\n🎯 高级搜索演示:")
        advanced_results = engine.filtered_search(
            "法规制度", 
            content_type="paragraph", 
            min_words=20, 
            top_k=3
        )
        for i, result in enumerate(advanced_results, 1):
            print(f"  {i}. {result['content'][:80]}... (分数: {result['score']:.4f})")
        
        print(f"\n✅ Python 查询演示完成!")
        print("💡 运行 interactive_query_demo() 进入交互式查询模式")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        
    finally:
        engine.disconnect()

if __name__ == "__main__":
    # 运行主演示
    # main()
    
    # 如果需要交互式查询，取消下面的注释
    # print("\n" + "="*60)
    interactive_query_demo() 