#!/usr/bin/env python3
"""
Milvus连接池管理器
避免频繁连接和断开Milvus
"""

import threading
from typing import Dict, Optional
from query_milvus import MilvusQueryEngine

class MilvusConnectionPool:
    """Milvus连接池"""
    
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
            self._connections: Dict[str, MilvusQueryEngine] = {}
            self._connection_lock = threading.Lock()
            self._initialized = True
            print("🔧 初始化Milvus连接池...")
    
    def get_connection(self, collection_name: str, host='localhost', port='19530') -> Optional[MilvusQueryEngine]:
        """获取或创建连接"""
        connection_key = f"{host}:{port}:{collection_name}"
        
        with self._connection_lock:
            # 如果连接已存在且有效，直接返回
            if connection_key in self._connections:
                connection = self._connections[connection_key]
                # 简单检查连接是否有效
                if connection.collection is not None:
                    return connection
                else:
                    # 连接失效，删除并重新创建
                    print(f"🔄 连接失效，重新创建: {collection_name}")
                    del self._connections[connection_key]
            
            # 创建新连接
            print(f"📡 创建新连接: {collection_name}")
            engine = MilvusQueryEngine(host=host, port=port, collection_name=collection_name)
            
            if engine.connect():
                self._connections[connection_key] = engine
                return engine
            else:
                print(f"❌ 连接创建失败: {collection_name}")
                return None
    
    def close_connection(self, collection_name: str, host='localhost', port='19530'):
        """关闭特定连接"""
        connection_key = f"{host}:{port}:{collection_name}"
        
        with self._connection_lock:
            if connection_key in self._connections:
                try:
                    self._connections[connection_key].disconnect()
                except:
                    pass
                del self._connections[connection_key]
                print(f"🔌 已关闭连接: {collection_name}")
    
    def close_all_connections(self):
        """关闭所有连接"""
        with self._connection_lock:
            for connection in self._connections.values():
                try:
                    connection.disconnect()
                except:
                    pass
            self._connections.clear()
            print("🔌 已关闭所有连接")
    
    def get_connection_count(self) -> int:
        """获取连接数量"""
        return len(self._connections)
    
    def get_connection_info(self) -> Dict[str, str]:
        """获取连接信息"""
        with self._connection_lock:
            return {key: "connected" for key in self._connections.keys()}

# 全局连接池实例
_connection_pool = MilvusConnectionPool()

def get_connection_pool() -> MilvusConnectionPool:
    """获取连接池单例"""
    return _connection_pool