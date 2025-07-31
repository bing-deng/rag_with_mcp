#!/usr/bin/env python3
"""
智能查询系统 Web 应用
Flask Web API 服务器，提供Web UI后端支持
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
from typing import Dict, List, Optional

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
os.chdir(current_dir)

from pymilvus import connections, utility
from query_milvus import MilvusQueryEngine
from llama_query import LLaMAQueryEngine
from connection_pool import get_connection_pool

# 获取全局连接池
connection_pool = get_connection_pool()

app = Flask(__name__)
CORS(app)

class WebQueryManager:
    """Web查询管理器"""
    
    def __init__(self, host=None, port=None):
        # 支持Docker环境变量
        self.host = host or os.getenv('MILVUS_HOST', 'localhost')
        self.port = port or os.getenv('MILVUS_PORT', '19530')
        self.connected = False
        self._connection_pool = {}  # 连接池
        
    def connect_to_milvus(self) -> bool:
        """连接到Milvus"""
        try:
            # 总是尝试重新连接以确保连接有效
            try:
                connections.disconnect("default")
            except:
                pass
            connections.connect("default", host=self.host, port=self.port)
            self.connected = True
            print(f"✅ Web管理器成功连接到 Milvus: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Web管理器连接Milvus失败: {e}")
            self.connected = False
            return False
    
    def get_collections(self) -> List[str]:
        """获取所有集合"""
        if not self.connect_to_milvus():
            return []
        try:
            return utility.list_collections()
        except Exception as e:
            print(f"获取集合失败: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """获取集合统计信息"""
        if not self.connect_to_milvus():
            return {}
        
        try:
            engine = MilvusQueryEngine(collection_name=collection_name)
            if engine.connect():
                stats = engine.get_statistics()
                engine.disconnect()
                return stats
            return {}
        except Exception as e:
            print(f"获取集合统计失败: {e}")
            return {}

# 全局查询管理器
query_manager = WebQueryManager()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/collections', methods=['GET'])
def get_collections():
    """获取所有集合"""
    try:
        collections = query_manager.get_collections()
        return jsonify({
            'success': True,
            'collections': collections
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/collections/<collection_name>/stats', methods=['GET'])
def get_collection_stats(collection_name: str):
    """获取集合统计信息"""
    try:
        stats = query_manager.get_collection_stats(collection_name)
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/vector-search', methods=['POST'])
def vector_search():
    """向量搜索API"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        collection_name = data.get('collection_name')
        content_type = data.get('content_type')
        min_words = data.get('min_words')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': '查询内容不能为空'}), 400
        if not collection_name:
            return jsonify({'error': '集合名称不能为空'}), 400
        
        # 创建查询引擎
        engine = MilvusQueryEngine(collection_name=collection_name)
        if not engine.connect():
            return jsonify({'error': '连接Milvus失败'}), 500
        
        try:
            # 执行搜索
            if content_type or min_words:
                # 高级过滤搜索
                min_words_int = int(min_words) if min_words and str(min_words).isdigit() else None
                results = engine.filtered_search(
                    query=query,
                    content_type=content_type,
                    url_contains=None,
                    min_words=min_words_int,
                    top_k=top_k
                )
            else:
                # 基础搜索
                results = engine.basic_search(query=query, top_k=top_k)
            
            # 格式化结果 - 安全获取字段
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', '无标题'),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'content_type': result.get('content_type', 'unknown'),
                    'distance': result.get('score', result.get('distance', 0.0))  # 统一使用score字段
                })
            
            return jsonify({
                'success': True,
                'results': formatted_results,
                'total': len(formatted_results)
            })
            
        finally:
            engine.disconnect()
            
    except Exception as e:
        print(f"向量搜索错误: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/llama-chat', methods=['POST'])
def llama_chat():
    """LLaMA AI问答API"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        collection_name = data.get('collection_name')
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        if not collection_name:
            return jsonify({'error': '集合名称不能为空'}), 400
        
        # 优化连接处理 - 使用连接池
        try:
            # 从连接池获取Milvus连接
            milvus_engine = connection_pool.get_connection(collection_name)
            if not milvus_engine:
                return jsonify({'error': 'Milvus连接失败，请检查服务状态'}), 500
            
            # 创建LLaMA查询引擎，优化速度和质量平衡
            model_name = 'llama3.2:3b'  # 使用更快的模型，减少等待时间
            engine = LLaMAQueryEngine(
                model_type='ollama',
                model_name=model_name,
                collection_name=collection_name
            )
            print(f"🚀 使用快速模型: {model_name}")
            # 直接使用池化的连接
            engine.milvus_engine = milvus_engine
            
            # 执行RAG查询 - 增加检索数量以获得更丰富信息
            result = engine.rag_query(question, top_k=10)
            
            # 格式化来源信息
            formatted_sources = []
            for source in result.get('sources', []):
                formatted_sources.append({
                    'title': source.get('title', '无标题'),
                    'content_type': source.get('content_type', ''),
                    'similarity': source.get('similarity', 0.0),
                    'url': source.get('url', '')
                })
            
            return jsonify({
                'success': True,
                'answer': result.get('generated_answer', '抱歉，无法生成回答'),
                'sources': formatted_sources
            })
            
        except Exception as inner_e:
            error_msg = str(inner_e)
            print(f"AI问答执行失败: {error_msg}")
            
            # 根据错误类型返回更友好的消息
            if 'RPC' in error_msg or 'channel' in error_msg:
                return jsonify({'error': 'Milvus服务连接中断，请刷新页面重试'}), 503
            elif 'ollama' in error_msg.lower():
                return jsonify({'error': 'AI模型服务不可用，请检查Ollama服务'}), 503
            else:
                return jsonify({'error': f'查询失败: {error_msg}'}), 500
        finally:
            # 不再手动断开连接，由连接池管理
            pass
            
    except Exception as e:
        print(f"AI问答错误: {e}")
        traceback.print_exc()
        
        # 检查是否是Ollama连接问题
        error_msg = str(e)
        if 'ollama' in error_msg.lower() or 'connection' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'Ollama服务不可用。请确保:\n1. Ollama服务正在运行: ollama serve\n2. 模型已下载: ollama pull llama3.2:3b'
            }), 503
        
        return jsonify({
            'success': False,
            'error': f'AI问答服务暂时不可用: {error_msg}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        # 检查Milvus连接
        milvus_ok = query_manager.connect_to_milvus()
        collections = query_manager.get_collections() if milvus_ok else []
        
        return jsonify({
            'status': 'healthy',
            'milvus_connected': milvus_ok,
            'collections_count': len(collections),
            'services': {
                'milvus': 'ok' if milvus_ok else 'error',
                'web_ui': 'ok'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '页面未找到'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    print("🚀 启动智能查询系统 Web 服务")
    print("=" * 60)
    print("Web UI: http://localhost:5001")
    print("API文档:")
    print("  GET  /api/collections - 获取所有集合")
    print("  GET  /api/collections/<name>/stats - 集合统计")
    print("  POST /api/vector-search - 向量搜索")
    print("  POST /api/llama-chat - AI问答")
    print("  GET  /health - 健康检查")
    print("-" * 60)
    
    # 启动服务器
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )