#!/usr/bin/env python3
"""
Web RAG应用 - 基于成功的统一Cohere RAG系统
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from cohere_unified_rag import CohereUnifiedRAGService

app = Flask(__name__)
CORS(app)

# グローバルRAGサービス
rag_service = None
service_ready = False

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def initialize_service():
    """RAGサービス初期化"""
    global rag_service, service_ready
    
    try:
        print("🔧 RAGサービス初期化開始...")
        
        # サービス作成
        rag_service = CohereUnifiedRAGService()
        
        # PDFナレッジベース読み込み
        pdf_path = "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
        success = rag_service.load_pdf_knowledge(pdf_path)
        
        if success:
            service_ready = True
            print("✅ Webサービス準備完了")
            return jsonify({
                'success': True,
                'message': '電力設備RAGシステムが正常に初期化されました。',
                'status': 'ready'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'ナレッジベースの読み込みに失敗しました。',
                'status': 'error'
            })
            
    except Exception as e:
        print(f"❌ 初期化エラー: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'初期化中にエラーが発生しました: {str(e)}',
            'status': 'error'
        })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """質問応答API"""
    global rag_service, service_ready
    
    try:
        if not service_ready or not rag_service:
            return jsonify({
                'success': False,
                'message': 'サービスが初期化されていません。',
                'answer': '',
                'confidence': 0,
                'search_results': []
            })
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'message': '質問を入力してください。',
                'answer': '',
                'confidence': 0,
                'search_results': []
            })
        
        print(f"🔍 Web質問受信: '{question}'")
        
        # RAG処理実行
        result = rag_service.ask_question(question)
        
        # 検索結果を整理（Webクライアント用）
        formatted_results = []
        for i, doc in enumerate(result['search_results'], 1):
            formatted_results.append({
                'rank': i,
                'content_preview': doc.get('content', '')[:200] + '...',
                'similarity': doc.get('similarity', 0),
                'completeness_score': doc.get('completeness_score', 0),
                'total_score': doc.get('total_score', 0)
            })
        
        return jsonify({
            'success': True,
            'message': '回答が正常に生成されました。',
            'answer': result['answer'],
            'confidence': result['confidence'],
            'processing_time': result['processing_time'],
            'search_results': formatted_results
        })
        
    except Exception as e:
        print(f"❌ 質問応答エラー: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'質問処理中にエラーが発生しました: {str(e)}',
            'answer': '',
            'confidence': 0,
            'search_results': []
        })

@app.route('/api/status', methods=['GET'])
def get_status():
    """サービス状態確認"""
    global service_ready
    
    return jsonify({
        'ready': service_ready,
        'status': 'ready' if service_ready else 'not_initialized'
    })

@app.route('/api/test_questions', methods=['GET'])  
def get_test_questions():
    """テスト質問リスト"""
    test_questions = [
        {
            'id': 1,
            'question': '電圧調査では、どの4つの情報を優先的に収集すべきですか？',
            'category': '電圧調査',
            'expected_type': '4項目リスト'
        },
        {
            'id': 2,
            'question': '電圧調査について教えてください',
            'category': '電圧調査',
            'expected_type': '概要説明'
        },
        {
            'id': 3, 
            'question': '電圧異常調査での記入ポイントは何ですか？',
            'category': '電圧調査',
            'expected_type': '記入要点'
        },
        {
            'id': 4,
            'question': '電柱番号の例を教えてください',
            'category': '電柱情報',
            'expected_type': '具体例'
        },
        {
            'id': 5,
            'question': '計器番号について説明してください',
            'category': '計測器',
            'expected_type': '説明'
        }
    ]
    
    return jsonify({
        'test_questions': test_questions
    })

if __name__ == '__main__':
    print("🌐 RAG Webサービス開始...")
    print("📖 ブラウザで http://localhost:5000 にアクセスしてください")
    print("🎯 統一Cohere RAGシステム搭載")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5002)
    finally:
        # クリーンアップ
        if rag_service:
            rag_service.close()