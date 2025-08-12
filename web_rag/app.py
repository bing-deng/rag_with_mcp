"""
Flask Web应用 - RAG系统的Web界面
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import threading
import time
from rag_service import WebRAGService
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)

# 全局RAG服务实例
rag_service = None
initialization_status = {"status": "not_started", "message": "", "progress": 0}

def initialize_rag_service():
    """后台初始化RAG服务"""
    global rag_service, initialization_status
    
    try:
        initialization_status["status"] = "initializing"
        initialization_status["message"] = "正在启动RAG服务..."
        initialization_status["progress"] = 10
        
        # 初始化服务
        rag_service = WebRAGService()
        initialization_status["progress"] = 30
        
        # PDF文件路径
        pdf_path = os.path.join(os.path.dirname(__file__), "../bedrock/pdf/high_takusoukun_web_manual_separate.pdf")
        
        initialization_status["message"] = "正在处理PDF文档..."
        initialization_status["progress"] = 50
        
        # 初始化知识库
        if rag_service.initialize_knowledge_base(pdf_path):
            initialization_status["status"] = "completed"
            initialization_status["message"] = "RAG系统初始化完成!"
            initialization_status["progress"] = 100
        else:
            initialization_status["status"] = "error"
            initialization_status["message"] = "知识库初始化失败"
            
    except Exception as e:
        initialization_status["status"] = "error"
        initialization_status["message"] = f"初始化出错: {str(e)}"
        print(f"初始化出错: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def init_system():
    """启动系统初始化"""
    global initialization_status
    
    if initialization_status["status"] in ["not_started", "error"]:
        initialization_status = {"status": "not_started", "message": "", "progress": 0}
        
        # 在后台线程中初始化
        thread = threading.Thread(target=initialize_rag_service)
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "message": "开始初始化RAG系统"})
    else:
        return jsonify({"success": False, "message": "系统已在初始化中或已完成"})

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    return jsonify(initialization_status)

@app.route('/api/query', methods=['POST'])
def query():
    """处理用户查询"""
    global rag_service
    
    if not rag_service or initialization_status["status"] != "completed":
        return jsonify({
            "success": False, 
            "message": "系统尚未初始化完成，请先初始化系统"
        })
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "success": False,
                "message": "请输入问题"
            })
        
        # 执行RAG查询
        result = rag_service.query(question, top_k=3)
        
        return jsonify({
            "success": True,
            "data": {
                "question": result["question"],
                "answer": result["answer"],
                "sources": [
                    {
                        "title": doc.get("title", "未知标题"),
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "certainty": doc.get("certainty", 0),
                        "source": doc.get("source", "未知来源")
                    }
                    for doc in result["sources"]
                ],
                "source_count": result["source_count"]
            }
        })
        
    except Exception as e:
        print(f"查询出错: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "message": f"查询处理出错: {str(e)}"
        })

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "rag_ready": rag_service is not None and initialization_status["status"] == "completed"
    })

if __name__ == '__main__':
    print("🌐 启动RAG Web应用...")
    print("请在浏览器中访问: http://localhost:5001")
    print("确保已设置AWS环境变量:")
    print("  export AWS_ACCESS_KEY_ID=your_key")
    print("  export AWS_SECRET_ACCESS_KEY=your_secret")
    print("  export AWS_DEFAULT_REGION=ap-northeast-1")
    
    app.run(debug=True, host='0.0.0.0', port=5002)