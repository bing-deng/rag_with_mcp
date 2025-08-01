#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动PDF搜索Web服务
"""

import os
import sys
import logging
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# 导入PDF处理模块
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder
from enhanced_qa_service import EnhancedQASystem

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 全局变量
milvus_manager = None
vector_embedder = None
enhanced_qa = None

def initialize_services():
    """初始化服务"""
    global milvus_manager, vector_embedder, enhanced_qa
    
    try:
        logger.info("正在初始化PDF搜索服务...")
        
        # 初始化Milvus管理器（使用日语优化模型）
        milvus_manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_japanese.db",
            collection_name="pdf_documents_japanese",
            dimension=768
        )
        
        # 初始化向量嵌入器（日语优化）
        vector_embedder = VectorEmbedder(
            model_name='cl-nagoya/sup-simcse-ja-base',
            dimension=768
        )
        
        # 加载集合到内存
        milvus_manager.load_collection()
        
        # 初始化增强问答系统
        enhanced_qa = EnhancedQASystem(milvus_manager, vector_embedder)
        
        # 获取统计信息
        stats = milvus_manager.get_collection_stats()
        logger.info(f"服务初始化完成 - 数据库中有 {stats.get('num_entities', 0)} 个文档块")
        
        return True
        
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        return False

@app.route('/')
def index():
    """主页"""
    return render_template('pdf_search.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        stats = milvus_manager.get_collection_stats() if milvus_manager else {}
        return jsonify({
            'status': 'healthy',
            'milvus_connected': milvus_manager is not None,
            'collection_stats': stats,
            'message': '服务运行正常'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """搜索PDF文档"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        query_text = data.get('query', '').strip()
        top_k = min(int(data.get('top_k', 5)), 10)
        
        if not query_text:
            return jsonify({'error': '查询文本不能为空'}), 400
        
        if not milvus_manager:
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        logger.info(f"搜索查询: {query_text}")
        
        # 执行搜索，获取更多结果以便过滤
        all_results = milvus_manager.search_similar(
            query_text=query_text,
            top_k=top_k * 3,  # 获取更多结果以便过滤
            embedder=vector_embedder
        )
        
        # 格式化所有结果，只做基本过滤
        filtered_results = []
        for result in all_results or []:
            text = result.get('text', '')
            # 只过滤明显的图像标题，保留所有文本内容
            if not (text.startswith('页面') and '图像' in text and ('jpeg' in text or 'png' in text)):
                filtered_results.append({
                    'id': result.get('id', ''),
                    'text': result.get('text', ''),
                    'pdf_name': result.get('pdf_name', ''),
                    'page_number': result.get('page_number', 0),
                    'chunk_type': result.get('chunk_type', 'text'),
                    'score': round(float(result.get('score', 0)), 4),
                    'relevance': 'high' if result.get('score', 0) > 0.7 else 
                               'medium' if result.get('score', 0) > 0.5 else 'low'
                })
                
                if len(filtered_results) >= top_k:
                    break
        
        return jsonify({
            'status': 'success',
            'query': query_text,
            'results_count': len(filtered_results),
            'results': filtered_results
        })
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/answer', methods=['POST'])
def answer_question():
    """智能问答"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        if not enhanced_qa:
            return jsonify({'error': '增强问答服务未初始化'}), 500
        
        logger.info(f"增强问答查询: {question}")
        
        # 使用增强问答系统
        result = enhanced_qa.answer_question(question)
        
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'question': result['question'],
                'answer': result['answer'],
                'sources': result['sources'],
                'confidence': result['confidence'],
                'keywords_found': result.get('keywords_found', []),
                'total_results': result.get('total_results', 0)
            })
        else:
            return jsonify({
                'status': 'error',
                'question': question,
                'error': result.get('error', '未知错误')
            }), 500
        
    except Exception as e:
        logger.error(f"问答失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计"""
    try:
        if not milvus_manager:
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        stats = milvus_manager.get_collection_stats()
        
        return jsonify({
            'status': 'success',
            'milvus_stats': stats,
            'config': {
                'dimension': 768,
                'collection_name': 'pdf_documents_japanese',
                'model': 'cl-nagoya/sup-simcse-ja-base'
            }
        })
        
    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 启动PDF文档智能搜索Web服务")
    print("=" * 60)
    
    # 初始化服务
    if not initialize_services():
        print("❌ 服务初始化失败，程序退出")
        sys.exit(1)
    
    print("✅ 服务初始化成功")
    print("🌐 启动Web服务器...")
    print("📍 访问地址: http://localhost:5001")
    print("⏹️  按 Ctrl+C 停止服务")
    print("=" * 60)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,  # 生产模式
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")