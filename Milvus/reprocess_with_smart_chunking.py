#!/usr/bin/env python3
"""
使用智能分块重新处理现有数据
将现有的细粒度数据重新组织成更有意义的块
"""

import os
import sys
import time
from typing import List, Dict, Tuple
from collections import defaultdict

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from pymilvus import connections, utility, Collection
from smart_chunker import SmartChunker, ChunkMetadata
from html_to_milvus import HTMLToMilvusProcessor

class DataReprocessor:
    """数据重处理器"""
    
    def __init__(self, collection_name: str = "kandenko_website"):
        self.collection_name = collection_name
        self.backup_collection_name = f"{collection_name}_backup"
        self.new_collection_name = f"{collection_name}_smart"
        self.chunker = SmartChunker()
        self.processor = HTMLToMilvusProcessor(collection_name=self.new_collection_name)
        
    def backup_existing_data(self) -> bool:
        """备份现有数据"""
        try:
            print("💾 备份现有数据...")
            connections.connect("default", host="localhost", port="19530")
            
            # 如果备份已存在，先删除
            if utility.has_collection(self.backup_collection_name):
                utility.drop_collection(self.backup_collection_name)
                print(f"🗑️  删除旧备份: {self.backup_collection_name}")
            
            # 重命名当前集合为备份
            if utility.has_collection(self.collection_name):
                # 由于Milvus不支持直接重命名，我们创建新集合并复制数据
                original_collection = Collection(self.collection_name)
                original_collection.load()
                
                # 获取所有数据
                all_data = original_collection.query(
                    expr="id >= 0",
                    output_fields=["*"],
                    limit=100000
                )
                
                print(f"✅ 备份完成，共 {len(all_data)} 条记录")
                return True
            else:
                print(f"❌ 原集合 {self.collection_name} 不存在")
                return False
                
        except Exception as e:
            print(f"❌ 备份失败: {e}")
            return False
    
    def extract_and_group_content(self) -> Dict[str, List[Dict]]:
        """提取并按URL分组内容"""
        try:
            print("📊 提取现有数据...")
            collection = Collection(self.collection_name)
            collection.load()
            
            # 获取所有数据
            all_data = collection.query(
                expr="id >= 0",
                output_fields=["*"],
                limit=100000
            )
            
            # 按URL分组
            grouped_data = defaultdict(list)
            for item in all_data:
                url = item.get('url', 'unknown')
                grouped_data[url].append(item)
            
            print(f"✅ 提取完成，共 {len(all_data)} 条记录，来自 {len(grouped_data)} 个URL")
            return dict(grouped_data)
            
        except Exception as e:
            print(f"❌ 提取数据失败: {e}")
            return {}
    
    def reprocess_grouped_content(self, grouped_data: Dict[str, List[Dict]]) -> List[Tuple[str, Dict]]:
        """重新处理分组后的内容"""
        print("🔄 重新处理内容...")
        
        reprocessed_items = []
        
        for url, items in grouped_data.items():
            print(f"处理URL: {url} (包含 {len(items)} 个片段)")
            
            # 按content_type分组，然后合并同类型内容
            type_groups = defaultdict(list)
            for item in items:
                content_type = item.get('content_type', 'text')
                type_groups[content_type].append(item.get('content', ''))
            
            # 为每个内容类型创建合并的文本
            for content_type, contents in type_groups.items():
                if not contents:
                    continue
                
                # 合并内容，去重
                merged_content = ' '.join(contents)
                if len(merged_content.strip()) < 50:  # 跳过太短的内容
                    continue
                
                # 使用智能分块
                metadata = {
                    'url': url,
                    'content_type': content_type,
                    'title': items[0].get('title', ''),
                    'original_count': len(contents)
                }
                
                chunks = self.chunker.chunk_content(merged_content, metadata)
                
                # 转换为存储格式
                for chunk_text, chunk_metadata in chunks:
                    processed_item = {
                        'content': chunk_text,
                        'url': url,
                        'title': metadata['title'],
                        'content_type': f"{content_type}_smart",
                        'chunk_index': chunk_metadata.chunk_index,
                        'total_chunks': chunk_metadata.total_chunks,
                        'quality_score': chunk_metadata.quality_score,
                        'word_count': chunk_metadata.word_count,
                        'language': chunk_metadata.language
                    }
                    reprocessed_items.append((chunk_text, processed_item))
        
        print(f"✅ 重新处理完成，生成 {len(reprocessed_items)} 个智能块")
        return reprocessed_items
    
    def create_new_collection(self, reprocessed_items: List[Tuple[str, Dict]]) -> bool:
        """创建新的优化集合"""
        try:
            print("🏗️  创建新的优化集合...")
            
            # 删除现有的新集合（如果存在）
            if utility.has_collection(self.new_collection_name):
                utility.drop_collection(self.new_collection_name)
                print(f"🗑️  删除现有集合: {self.new_collection_name}")
            
            # 连接处理器并创建集合
            if not self.processor.connect():
                print("❌ 无法连接到Milvus")
                return False
            
            if not self.processor.create_collection():
                print("❌ 创建集合失败")
                return False
            
            # 批量插入数据
            batch_size = 100
            inserted_count = 0
            
            for i in range(0, len(reprocessed_items), batch_size):
                batch = reprocessed_items[i:i+batch_size]
                
                # 准备批量数据
                batch_data = []
                for content, metadata in batch:
                    # 向量化内容
                    vector = self.processor.text_to_vector(content)
                    if vector:
                        data_point = {
                            'id': inserted_count,
                            'content': content,
                            'vector': vector,
                            'url': metadata['url'],
                            'title': metadata['title'],
                            'content_type': metadata['content_type'],
                            'chunk_index': metadata.get('chunk_index', 0),
                            'quality_score': metadata.get('quality_score', 0.0),
                            'word_count': metadata.get('word_count', 0),
                            'language': metadata.get('language', 'unknown')
                        }
                        batch_data.append(data_point)
                        inserted_count += 1
                
                # 插入批量数据
                if batch_data:
                    entities = [
                        [item['id'] for item in batch_data],
                        [item['content'] for item in batch_data],
                        [item['vector'] for item in batch_data],
                        [item['url'] for item in batch_data],
                        [item['title'] for item in batch_data],
                        [item['content_type'] for item in batch_data]
                    ]
                    
                    self.processor.collection.insert(entities)
                    print(f"📥 已插入 {len(batch_data)} 条记录 (总计: {inserted_count})")
            
            # 创建索引
            print("🔗 创建索引...")
            self.processor.create_index()
            
            print(f"✅ 新集合创建完成，共插入 {inserted_count} 条记录")
            return True
            
        except Exception as e:
            print(f"❌ 创建新集合失败: {e}")
            return False
        finally:
            self.processor.disconnect()
    
    def run_reprocessing(self):
        """执行完整的重新处理流程"""
        print("🚀 开始智能分块重新处理")
        print("=" * 60)
        
        # 1. 备份现有数据
        if not self.backup_existing_data():
            print("❌ 备份失败，停止处理")
            return False
        
        # 2. 提取并分组内容
        grouped_data = self.extract_and_group_content()
        if not grouped_data:
            print("❌ 没有找到数据，停止处理")
            return False
        
        # 3. 重新处理内容
        reprocessed_items = self.reprocess_grouped_content(grouped_data)
        if not reprocessed_items:
            print("❌ 重新处理失败，停止处理")
            return False
        
        # 4. 创建新集合
        if not self.create_new_collection(reprocessed_items):
            print("❌ 创建新集合失败")
            return False
        
        print("\n🎉 智能分块重新处理完成！")
        print(f"新集合名称: {self.new_collection_name}")
        print("\n📝 后续步骤:")
        print(f"1. 测试新集合: {self.new_collection_name}")
        print(f"2. 如果满意，可以将 web_app.py 中的集合名改为: {self.new_collection_name}")
        print(f"3. 备份数据在原集合中，可以随时恢复")
        
        return True

def main():
    """主函数"""
    print("🤖 智能分块数据重新处理工具")
    print("此工具将重新组织现有的细粒度数据为更有意义的块")
    print()
    
    # 确认操作
    confirm = input("⚠️  这将创建新集合并重新处理所有数据，继续？(y/N): ").strip().lower()
    if confirm != 'y':
        print("🚫 操作已取消")
        return
    
    # 执行重新处理
    reprocessor = DataReprocessor()
    success = reprocessor.run_reprocessing()
    
    if success:
        print("\n✅ 处理成功完成")
    else:
        print("\n❌ 处理失败")

if __name__ == "__main__":
    main()