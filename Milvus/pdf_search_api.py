#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF搜索API服务
为前端提供PDF文档检索和问答功能
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# 导入PDF处理模块
from pdf_to_milvus import MilvusPDFManager, PDFProcessor, VectorEmbedder

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局配置
CONFIG = {
    'MILVUS_HOST': 'localhost',
    'MILVUS_PORT': '19530',
    'COLLECTION_NAME': 'pdf_documents',
    'DIMENSION': 384,
    'PDF_UPLOAD_FOLDER': 'pdf/',
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024  # 50MB
}

# 全局变量
milvus_manager = None
vector_embedder = None

def initialize_services():
    """初始化服务"""
    global milvus_manager, vector_embedder
    
    try:
        # 初始化Milvus管理器 - 使用Milvus Lite
        milvus_manager = MilvusPDFManager(
            use_lite=True,  # 使用Milvus Lite
            lite_uri="./milvus_demo.db",  # 指定数据库文件
            collection_name=CONFIG['COLLECTION_NAME'],
            dimension=CONFIG['DIMENSION']
        )
        
        # 初始化向量嵌入器
        vector_embedder = VectorEmbedder(dimension=CONFIG['DIMENSION'])
        
        # 加载集合到内存
        milvus_manager.load_collection()
        
        logger.info("所有服务初始化完成")
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
            'timestamp': datetime.now().isoformat(),
            'milvus_connected': milvus_manager is not None,
            'collection_stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """搜索PDF文档"""
    try:
        # 获取请求参数
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        query_text = data.get('query', '').strip()
        top_k = min(int(data.get('top_k', 5)), 20)  # 限制最大返回数量
        pdf_filter = data.get('pdf_filter', '').strip()
        
        if not query_text:
            return jsonify({'error': '查询文本不能为空'}), 400
        
        if not milvus_manager:
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        # 执行搜索
        logger.info(f"执行搜索: {query_text} (top_k={top_k})")
        results = milvus_manager.search_similar(
            query_text=query_text,
            top_k=top_k,
            pdf_name_filter=pdf_filter if pdf_filter else None,
            embedder=vector_embedder  # 使用共享的embedder实例
        )
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result['id'],
                'text': result['text'],
                'pdf_name': result['pdf_name'],
                'page_number': result['page_number'],
                'chunk_type': result['chunk_type'],
                'score': round(float(result['score']), 4),
                'metadata': result['metadata'],
                'relevance': 'high' if result['score'] > 0.8 else 
                           'medium' if result['score'] > 0.6 else 'low'
            })
        
        return jsonify({
            'status': 'success',
            'query': query_text,
            'results_count': len(formatted_results),
            'results': formatted_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/answer', methods=['POST'])
def answer_question():
    """基于PDF内容回答问题"""
    try:
        # 获取请求参数
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        question = data.get('question', '').strip()
        context_size = min(int(data.get('context_size', 3)), 10)
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        if not milvus_manager:
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        # 搜索相关内容
        logger.info(f"回答问题: {question}")
        search_results = milvus_manager.search_similar(
            query_text=question,
            top_k=context_size,
            embedder=vector_embedder  # 使用共享的embedder实例
        )
        
        if not search_results:
            return jsonify({
                'status': 'success',
                'question': question,
                'answer': '抱歉，没有找到相关的信息来回答您的问题。',
                'sources': [],
                'confidence': 'low'
            })
        
        # 组合上下文信息
        context_texts = []
        sources = []
        
        for result in search_results:
            context_texts.append(result['text'])
            sources.append({
                'pdf_name': result['pdf_name'],
                'page_number': result['page_number'],
                'score': result['score'],
                'text_preview': result['text'][:100] + '...' if len(result['text']) > 100 else result['text']
            })
        
        # 简单的问答逻辑（基于关键词匹配和相关性）
        answer = generate_answer(question, context_texts, search_results)
        
        # 计算置信度
        avg_score = sum(r['score'] for r in search_results) / len(search_results)
        confidence = 'high' if avg_score > 0.8 else 'medium' if avg_score > 0.6 else 'low'
        
        return jsonify({
            'status': 'success',
            'question': question,
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'context_used': len(context_texts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"问答失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def generate_answer(question: str, context_texts: List[str], search_results: List[Dict]) -> str:
    """生成基于上下文的答案"""
    # 这是一个简化的问答逻辑
    # 在生产环境中，可以集成更复杂的LLM或问答模型
    
    if not context_texts:
        return "抱歉，没有找到相关信息。"
    
    # 合并上下文
    combined_context = ' '.join(context_texts)
    
    # 简单的关键词匹配和总结
    question_lower = question.lower()
    
    # 查找最相关的文档片段
    best_match = search_results[0] if search_results else None
    
    if best_match and best_match['score'] > 0.7:
        # 高相关性，直接返回相关内容
        answer = f"根据文档内容，{best_match['text'][:300]}"
        if len(best_match['text']) > 300:
            answer += "..."
        
        answer += f"\n\n参考来源：{best_match['pdf_name']} 第{best_match['page_number']}页"
        return answer
    
    else:
        # 低相关性，提供一般性回答
        return f"在相关文档中找到以下信息：\n\n{combined_context[:500]}{'...' if len(combined_context) > 500 else ''}"

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """上传并处理新的PDF文档"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件被上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名不能为空'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': '只支持PDF文件'}), 400
        
        # 保存文件
        os.makedirs(CONFIG['PDF_UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(CONFIG['PDF_UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # 处理PDF
        processor = PDFProcessor(min_chunk_size=100, max_chunk_size=800, overlap_size=50)
        chunks = processor.process_pdf(file_path)
        
        if not chunks:
            return jsonify({'error': 'PDF处理失败'}), 500
        
        # 生成向量嵌入
        embedder = VectorEmbedder(dimension=CONFIG['DIMENSION'])
        chunks = embedder.embed_chunks(chunks)
        
        # 存储到Milvus
        success = milvus_manager.insert_chunks(chunks)
        if not success:
            return jsonify({'error': '存储到向量数据库失败'}), 500
        
        # 重新加载集合
        milvus_manager.load_collection()
        
        return jsonify({
            'status': 'success',
            'message': f'PDF文档 {file.filename} 上传并处理完成',
            'chunks_created': len(chunks),
            'file_path': file_path,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"上传处理失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计信息"""
    try:
        if not milvus_manager:
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        stats = milvus_manager.get_collection_stats()
        
        # 添加PDF文件统计
        pdf_folder = CONFIG['PDF_UPLOAD_FOLDER']
        pdf_files = []
        if os.path.exists(pdf_folder):
            pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        return jsonify({
            'status': 'success',
            'milvus_stats': stats,
            'pdf_files_count': len(pdf_files),
            'pdf_files': pdf_files,
            'config': {
                'dimension': CONFIG['DIMENSION'],
                'collection_name': CONFIG['COLLECTION_NAME']
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '页面未找到'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '内部服务器错误'}), 500

if __name__ == '__main__':
    # 初始化服务
    if not initialize_services():
        logger.error("服务初始化失败，程序退出")
        exit(1)
    
    # 启动Web服务
    logger.info("启动PDF搜索API服务...")
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )