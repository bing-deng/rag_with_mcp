#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版PDF搜索API服务
集成优化处理和错误处理功能
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# 导入我们的增强模块
from error_handler import (
    enhanced_logger, ErrorLevel, ErrorCategory,
    error_handler, retry_on_failure, SafeExecutor
)
from optimized_pdf_processor import (
    OptimizedPDFProcessor, OptimizedVectorEmbedder, OptimizedMilvusManager
)
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 全局配置
CONFIG = {
    'MILVUS_HOST': 'localhost',
    'MILVUS_PORT': '19530',
    'COLLECTION_NAME': 'pdf_documents',
    'DIMENSION': 384,
    'PDF_UPLOAD_FOLDER': 'pdf/',
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB
    'USE_OPTIMIZED_PROCESSING': True,
    'MAX_SEARCH_RESULTS': 20,
    'DEFAULT_SEARCH_RESULTS': 5,
    'SEARCH_TIMEOUT': 30,
    'VECTORIZATION_BATCH_SIZE': 32
}

# 全局变量
milvus_manager = None
vector_embedder = None
system_stats = {
    'start_time': datetime.now(),
    'requests_count': 0,
    'errors_count': 0,
    'search_queries': 0,
    'answer_queries': 0
}

@retry_on_failure(max_retries=3, delay=1.0, category=ErrorCategory.SYSTEM)
def initialize_services():
    """初始化服务"""
    global milvus_manager, vector_embedder
    
    enhanced_logger.log_error(
        level=ErrorLevel.INFO,
        category=ErrorCategory.SYSTEM,
        message="开始初始化PDF搜索服务"
    )
    
    try:
        # 选择使用优化版本还是标准版本
        if CONFIG['USE_OPTIMIZED_PROCESSING']:
            # 使用优化版本
            milvus_manager = OptimizedMilvusManager(
                use_lite=True,
                lite_uri="./enhanced_milvus.db",
                collection_name=CONFIG['COLLECTION_NAME'],
                dimension=CONFIG['DIMENSION']
            )
            vector_embedder = OptimizedVectorEmbedder(
                dimension=CONFIG['DIMENSION'],
                batch_size=CONFIG['VECTORIZATION_BATCH_SIZE']
            )
        else:
            # 使用标准版本
            milvus_manager = MilvusPDFManager(
                use_lite=True,
                lite_uri="./enhanced_milvus.db",
                collection_name=CONFIG['COLLECTION_NAME'],
                dimension=CONFIG['DIMENSION']
            )
            vector_embedder = VectorEmbedder(dimension=CONFIG['DIMENSION'])
        
        # 加载集合到内存
        milvus_manager.load_collection()
        
        enhanced_logger.log_error(
            level=ErrorLevel.INFO,
            category=ErrorCategory.SYSTEM,
            message="所有服务初始化完成"
        )
        return True
        
    except Exception as e:
        enhanced_logger.log_error(
            level=ErrorLevel.CRITICAL,
            category=ErrorCategory.SYSTEM,
            message="服务初始化失败",
            exception=e
        )
        return False

def update_request_stats(request_type: str = 'general', error: bool = False):
    """更新请求统计"""
    system_stats['requests_count'] += 1
    if error:
        system_stats['errors_count'] += 1
    if request_type == 'search':
        system_stats['search_queries'] += 1
    elif request_type == 'answer':
        system_stats['answer_queries'] += 1

@app.before_request
def before_request():
    """请求前处理"""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """请求后处理"""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        enhanced_logger.log_error(
            level=ErrorLevel.DEBUG,
            category=ErrorCategory.API,
            message=f"API请求完成: {request.method} {request.path}",
            context={
                'duration': f"{duration:.3f}s",
                'status_code': response.status_code,
                'remote_addr': request.remote_addr
            }
        )
    return response

@app.route('/')
def index():
    """主页"""
    return render_template('pdf_search.html')

@app.route('/api/health', methods=['GET'])
@error_handler(category=ErrorCategory.API, level=ErrorLevel.WARNING, reraise=False)
def health_check():
    """健康检查"""
    try:
        stats = milvus_manager.get_collection_stats() if milvus_manager else {}
        uptime = (datetime.now() - system_stats['start_time']).total_seconds()
        
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'milvus_connected': milvus_manager is not None,
            'vector_embedder_ready': vector_embedder is not None,
            'collection_stats': stats,
            'system_stats': system_stats,
            'config': {
                'use_optimized': CONFIG['USE_OPTIMIZED_PROCESSING'],
                'dimension': CONFIG['DIMENSION'],
                'collection_name': CONFIG['COLLECTION_NAME']
            }
        }
        
        update_request_stats()
        return jsonify(health_info)
        
    except Exception as e:
        update_request_stats(error=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/search', methods=['POST'])
@error_handler(category=ErrorCategory.API, level=ErrorLevel.ERROR)
def search_documents():
    """搜索PDF文档"""
    try:
        # 验证请求数据
        data = request.get_json()
        if not data:
            update_request_stats('search', error=True)
            return jsonify({'error': '请求数据不能为空'}), 400
        
        query_text = data.get('query', '').strip()
        top_k = min(int(data.get('top_k', CONFIG['DEFAULT_SEARCH_RESULTS'])), CONFIG['MAX_SEARCH_RESULTS'])
        pdf_filter = data.get('pdf_filter', '').strip()
        
        if not query_text:
            update_request_stats('search', error=True)
            return jsonify({'error': '查询文本不能为空'}), 400
        
        if not milvus_manager:
            update_request_stats('search', error=True)
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        # 记录搜索请求
        enhanced_logger.log_error(
            level=ErrorLevel.INFO,
            category=ErrorCategory.API,
            message=f"执行搜索查询: {query_text}",
            context={
                'query': query_text,
                'top_k': top_k,
                'pdf_filter': pdf_filter,
                'remote_addr': request.remote_addr
            }
        )
        
        # 执行搜索（使用超时执行）
        def search_function():
            if CONFIG['USE_OPTIMIZED_PROCESSING']:
                # 使用优化版本的搜索
                query_embedding = vector_embedder.embed_texts_batch([query_text], show_progress=False)[0]
                # 这里需要实现优化版本的搜索逻辑
                return milvus_manager.search_similar(
                    query_text=query_text,
                    top_k=top_k,
                    pdf_name_filter=pdf_filter if pdf_filter else None,
                    embedder=vector_embedder
                )
            else:
                return milvus_manager.search_similar(
                    query_text=query_text,
                    top_k=top_k,
                    pdf_name_filter=pdf_filter if pdf_filter else None,
                    embedder=vector_embedder
                )
        
        results = SafeExecutor.execute_with_timeout(
            search_function,
            timeout=CONFIG['SEARCH_TIMEOUT'],
            category=ErrorCategory.API
        )
        
        # 格式化结果
        formatted_results = []
        for result in results or []:
            formatted_results.append({
                'id': result.get('id', ''),
                'text': result.get('text', ''),
                'pdf_name': result.get('pdf_name', ''),
                'page_number': result.get('page_number', 0),
                'chunk_type': result.get('chunk_type', 'text'),
                'score': round(float(result.get('score', 0)), 4),
                'metadata': result.get('metadata', {}),
                'relevance': 'high' if result.get('score', 0) > 0.8 else 
                           'medium' if result.get('score', 0) > 0.6 else 'low'
            })
        
        update_request_stats('search')
        
        return jsonify({
            'status': 'success',
            'query': query_text,
            'results_count': len(formatted_results),
            'results': formatted_results,
            'processing_time': time.time() - request.start_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        update_request_stats('search', error=True)
        enhanced_logger.log_error(
            level=ErrorLevel.ERROR,
            category=ErrorCategory.API,
            message="搜索请求失败",
            exception=e,
            context={'query': data.get('query', 'unknown') if 'data' in locals() else 'unknown'}
        )
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/answer', methods=['POST'])
@error_handler(category=ErrorCategory.API, level=ErrorLevel.ERROR)
def answer_question():
    """基于PDF内容回答问题"""
    try:
        # 验证请求数据
        data = request.get_json()
        if not data:
            update_request_stats('answer', error=True)
            return jsonify({'error': '请求数据不能为空'}), 400
        
        question = data.get('question', '').strip()
        context_size = min(int(data.get('context_size', 3)), 10)
        
        if not question:
            update_request_stats('answer', error=True)
            return jsonify({'error': '问题不能为空'}), 400
        
        if not milvus_manager:
            update_request_stats('answer', error=True)
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        # 记录问答请求
        enhanced_logger.log_error(
            level=ErrorLevel.INFO,
            category=ErrorCategory.API,
            message=f"执行问答查询: {question}",
            context={
                'question': question,
                'context_size': context_size,
                'remote_addr': request.remote_addr
            }
        )
        
        # 搜索相关内容
        def search_for_answer():
            if CONFIG['USE_OPTIMIZED_PROCESSING']:
                query_embedding = vector_embedder.embed_texts_batch([question], show_progress=False)[0]
                return milvus_manager.search_similar(
                    query_text=question,
                    top_k=context_size,
                    embedder=vector_embedder
                )
            else:
                return milvus_manager.search_similar(
                    query_text=question,
                    top_k=context_size,
                    embedder=vector_embedder
                )
        
        search_results = SafeExecutor.execute_with_timeout(
            search_for_answer,
            timeout=CONFIG['SEARCH_TIMEOUT'],
            category=ErrorCategory.API
        )
        
        if not search_results:
            update_request_stats('answer')
            return jsonify({
                'status': 'success',
                'question': question,
                'answer': '抱歉，没有找到相关的信息来回答您的问题。',
                'sources': [],
                'confidence': 'low',
                'context_used': 0
            })
        
        # 组合上下文信息
        context_texts = []
        sources = []
        
        for result in search_results:
            context_texts.append(result.get('text', ''))
            sources.append({
                'pdf_name': result.get('pdf_name', ''),
                'page_number': result.get('page_number', 0),
                'score': result.get('score', 0),
                'text_preview': (result.get('text', '')[:100] + '...' 
                               if len(result.get('text', '')) > 100 
                               else result.get('text', ''))
            })
        
        # 生成答案
        answer = generate_enhanced_answer(question, context_texts, search_results)
        
        # 计算置信度
        avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
        confidence = 'high' if avg_score > 0.8 else 'medium' if avg_score > 0.6 else 'low'
        
        update_request_stats('answer')
        
        return jsonify({
            'status': 'success',
            'question': question,
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'context_used': len(context_texts),
            'processing_time': time.time() - request.start_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        update_request_stats('answer', error=True)
        enhanced_logger.log_error(
            level=ErrorLevel.ERROR,
            category=ErrorCategory.API,
            message="问答请求失败",
            exception=e,
            context={'question': data.get('question', 'unknown') if 'data' in locals() else 'unknown'}
        )
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def generate_enhanced_answer(question: str, context_texts: List[str], search_results: List[Dict]) -> str:
    """生成增强的答案"""
    if not context_texts:
        return "抱歉，没有找到相关信息。"
    
    # 合并上下文
    combined_context = ' '.join(context_texts)
    
    # 查找最相关的文档片段
    best_match = search_results[0] if search_results else None
    
    if best_match and best_match.get('score', 0) > 0.7:
        # 高相关性，返回详细答案
        answer_parts = []
        answer_parts.append("根据文档内容：")
        
        # 截取合适长度的内容
        best_text = best_match.get('text', '')
        if len(best_text) > 400:
            answer_parts.append(f"{best_text[:400]}...")
        else:
            answer_parts.append(best_text)
        
        answer_parts.append(f"\n\n参考来源：{best_match.get('pdf_name', '未知文档')} 第{best_match.get('page_number', 0)}页")
        
        return '\n'.join(answer_parts)
    
    elif len(search_results) >= 2:
        # 中等相关性，提供多个来源的综合信息
        answer_parts = []
        answer_parts.append("基于相关文档内容：")
        
        for i, result in enumerate(search_results[:2], 1):
            text = result.get('text', '')
            if text:
                preview = text[:200] + ('...' if len(text) > 200 else '')
                answer_parts.append(f"\n{i}. {preview}")
        
        answer_parts.append(f"\n\n以上信息来自 {len(search_results)} 个相关文档片段。")
        
        return '\n'.join(answer_parts)
    
    else:
        # 低相关性，提供一般性回答
        preview = combined_context[:300] + ('...' if len(combined_context) > 300 else '')
        return f"在相关文档中找到以下信息：\n\n{preview}"

@app.route('/api/stats', methods=['GET'])
@error_handler(category=ErrorCategory.API, level=ErrorLevel.WARNING, reraise=False)
def get_enhanced_stats():
    """获取增强的系统统计信息"""
    try:
        if not milvus_manager:
            update_request_stats(error=True)
            return jsonify({'error': 'Milvus服务未初始化'}), 500
        
        milvus_stats = milvus_manager.get_collection_stats()
        
        # PDF文件统计
        pdf_folder = CONFIG['PDF_UPLOAD_FOLDER']
        pdf_files = []
        if os.path.exists(pdf_folder):
            pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        # 错误统计
        error_summary = enhanced_logger.get_error_summary()
        
        # 系统运行时间
        uptime = (datetime.now() - system_stats['start_time']).total_seconds()
        
        stats_data = {
            'status': 'success',
            'system_info': {
                'uptime_seconds': uptime,
                'uptime_formatted': f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
                'start_time': system_stats['start_time'].isoformat(),
                'use_optimized_processing': CONFIG['USE_OPTIMIZED_PROCESSING']
            },
            'milvus_stats': milvus_stats,
            'pdf_files': {
                'count': len(pdf_files),
                'files': pdf_files
            },
            'request_stats': system_stats,
            'error_stats': error_summary,
            'config': {
                'dimension': CONFIG['DIMENSION'],
                'collection_name': CONFIG['COLLECTION_NAME'],
                'max_search_results': CONFIG['MAX_SEARCH_RESULTS'],
                'search_timeout': CONFIG['SEARCH_TIMEOUT']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        update_request_stats()
        return jsonify(stats_data)
        
    except Exception as e:
        update_request_stats(error=True)
        enhanced_logger.log_error(
            level=ErrorLevel.ERROR,
            category=ErrorCategory.API,
            message="获取统计信息失败",
            exception=e
        )
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/errors', methods=['GET'])
@error_handler(category=ErrorCategory.API, level=ErrorLevel.WARNING, reraise=False)
def get_error_logs():
    """获取错误日志"""
    try:
        error_summary = enhanced_logger.get_error_summary()
        return jsonify({
            'status': 'success',
            'error_summary': error_summary,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    update_request_stats(error=True)
    return jsonify({'error': '页面未找到'}), 404

@app.errorhandler(500)
def internal_error(error):
    update_request_stats(error=True)
    enhanced_logger.log_error(
        level=ErrorLevel.ERROR,
        category=ErrorCategory.API,
        message="内部服务器错误",
        exception=error,
        context={'endpoint': request.endpoint}
    )
    return jsonify({'error': '内部服务器错误'}), 500

if __name__ == '__main__':
    # 初始化服务
    if not initialize_services():
        enhanced_logger.log_error(
            level=ErrorLevel.CRITICAL,
            category=ErrorCategory.SYSTEM,
            message="服务初始化失败，程序退出"
        )
        exit(1)
    
    # 启动Web服务
    enhanced_logger.log_error(
        level=ErrorLevel.INFO,
        category=ErrorCategory.SYSTEM,
        message="启动增强版PDF搜索API服务",
        context={
            'host': '0.0.0.0',
            'port': 5001,
            'debug': True,
            'optimized_processing': CONFIG['USE_OPTIMIZED_PROCESSING']
        }
    )
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=True,
            threaded=True
        )
    except Exception as e:
        enhanced_logger.log_error(
            level=ErrorLevel.CRITICAL,
            category=ErrorCategory.SYSTEM,
            message="Web服务启动失败",
            exception=e
        )
        # 导出错误报告
        enhanced_logger.export_error_report()
        raise