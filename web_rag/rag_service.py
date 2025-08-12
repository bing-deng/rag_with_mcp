"""
整合的RAG服务 - 结合PDF处理、Weaviate和AWS Bedrock
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from bedrock.bedrock_usage import TokyoBedrockService
except ImportError as e:
    print(f"导入Bedrock服务失败: {e}")
    print("请确保bedrock模块在正确位置")
    sys.exit(1)

# 修复weaviate客户端导入
try:
    # 直接从weaviate目录导入
    weaviate_path = os.path.join(project_root, 'weaviate')
    sys.path.insert(0, weaviate_path)
    from weaviate_client import WeaviateRAGClient
except ImportError as e:
    print(f"导入Weaviate客户端失败: {e}")
    print(f"项目根目录: {project_root}")
    print(f"Weaviate路径: {weaviate_path}")
    print("请确保weaviate模块在正确位置")
    sys.exit(1)

from pdf_processor import PDFProcessor
from typing import List, Dict, Any, Optional
import time

class WebRAGService:
    """网页RAG服务类"""
    
    def __init__(self):
        try:
            print("🔧 初始化WebRAG服务...")
            print(f"项目根目录: {project_root}")
            
            self.bedrock_service = TokyoBedrockService()
            self.weaviate_client = WeaviateRAGClient()
            self.pdf_processor = PDFProcessor()
            self.collection_name = "ManualDocuments"
            print("✅ WebRAG服务初始化完成")
        except Exception as e:
            print(f"❌ WebRAG服务初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def initialize_knowledge_base(self, pdf_path: str) -> bool:
        """初始化知识库"""
        try:
            print("📚 开始初始化知识库...")
            
            # 1. 等待Weaviate就绪
            print("🔗 连接Weaviate数据库...")
            if not self.weaviate_client.wait_for_weaviate(timeout=60):
                print("❌ Weaviate连接失败")
                return False
            
            # 2. 处理PDF文档
            print("📄 处理PDF文档...")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            if not chunks:
                print("❌ PDF处理失败")
                return False
            
            print(f"✅ PDF处理完成，共生成 {len(chunks)} 个文档块")
            
            # 3. 创建Weaviate collection
            print("🗃️ 创建向量数据库collection...")
            if not self.weaviate_client.create_schema(self.collection_name):
                print("❌ Collection创建失败")
                return False
            
            # 4. 添加文档到Weaviate
            print("📝 将文档添加到向量数据库...")
            if not self.weaviate_client.add_documents(chunks, self.collection_name):
                print("❌ 文档添加失败")
                return False
            
            # 5. 等待向量化完成
            print("⏳ 等待文档向量化完成...")
            time.sleep(20)  # 增加等待时间确保向量化完成
            
            # 6. 验证数据
            count = self.weaviate_client.get_stats(self.collection_name)
            if count > 0:
                print(f"✅ 知识库初始化完成! 包含 {count} 个文档块")
                return True
            else:
                print("❌ 知识库验证失败，文档数量为0")
                return False
                
        except Exception as e:
            print(f"❌ 知识库初始化出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        try:
            print(f"🔍 执行语义搜索: {query}")
            
            # 使用Weaviate进行语义搜索
            results = self.weaviate_client.semantic_search(
                query, 
                class_name=self.collection_name, 
                limit=limit
            )
            
            if results:
                print(f"✅ 找到 {len(results)} 个相关文档")
                for i, doc in enumerate(results[:3], 1):  # 显示前3个结果的简要信息
                    print(f"  {i}. 相似度: {doc.get('certainty', 0):.3f} - {doc['content'][:50]}...")
                return results
            else:
                print("❌ 未找到相关文档")
                return []
                
        except Exception as e:
            print(f"❌ 搜索出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> Optional[str]:
        """生成基于上下文的回答"""
        try:
            if not context_docs:
                return "抱歉，我没有找到相关的文档内容来回答您的问题。请尝试用不同的方式提问，或者询问其他相关问题。"
            
            # 构建上下文
            context = "\n\n".join([
                f"【文档片段 {i+1}】\n{doc['content']}"
                for i, doc in enumerate(context_docs)
            ])
            
            # 限制上下文长度，避免超出模型限制
            if len(context) > 3000:
                context = context[:3000] + "\n\n[内容因长度限制有所截断]"
            
            # 构建增强提示
            enhanced_prompt = f"""请基于以下文档内容回答用户问题。请提供准确、详细的回答，并在适当时引用文档内容。

文档上下文：
{context}

用户问题：{query}

回答要求：
1. 基于上述文档内容提供准确的回答
2. 如果文档中没有完全相关的信息，请明确说明
3. 回答要结构化、易于理解
4. 适当时可以提供具体的操作步骤或建议

请回答："""

            print("🤖 正在生成智能回答...")
            
            # 使用Claude生成回答
            answer = self.bedrock_service.chat_with_claude(
                message=enhanced_prompt,
                system_prompt="你是一个专业的文档助手，请基于提供的文档内容准确回答用户问题。回答要清晰、有条理，并适当引用文档内容。如果文档中没有相关信息，请诚实说明。",
                max_tokens=1500
            )
            
            if answer:
                print("✅ 智能回答生成成功")
                return answer
            else:
                return "抱歉，回答生成失败。这可能是由于网络问题或服务临时不可用，请稍后重试。"
                
        except Exception as e:
            print(f"❌ 回答生成出错: {e}")
            import traceback
            traceback.print_exc()
            return "抱歉，处理您的问题时出现系统错误，请稍后重试。如果问题持续存在，请联系技术支持。"
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """完整的RAG查询流程"""
        print(f"\n{'='*60}")
        print(f"🔍 开始处理RAG查询: {question}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. 搜索相关文档
        relevant_docs = self.search_documents(question, limit=top_k)
        
        # 2. 生成回答
        answer = self.generate_answer(question, relevant_docs)
        
        # 3. 计算处理时间
        processing_time = time.time() - start_time
        
        # 4. 返回结果
        result = {
            "question": question,
            "answer": answer,
            "sources": relevant_docs,
            "source_count": len(relevant_docs),
            "processing_time": round(processing_time, 2)
        }
        
        print(f"✅ RAG查询完成，耗时: {processing_time:.2f}秒")
        print(f"{'='*60}")
        
        return result
    
    def close(self):
        """关闭连接"""
        try:
            if hasattr(self.weaviate_client, 'close'):
                self.weaviate_client.close()
            print("✅ RAG服务连接已关闭")
        except Exception as e:
            print(f"⚠️ 关闭连接时出现警告: {e}")

def test_rag_service():
    """测试RAG服务"""
    pdf_path = os.path.join(project_root, "bedrock/pdf/high_takusoukun_web_manual_separate.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF文件不存在: {pdf_path}")
        return
    
    # 初始化服务
    rag_service = None
    try:
        rag_service = WebRAGService()
        
        # 初始化知识库
        if not rag_service.initialize_knowledge_base(pdf_path):
            print("❌ 知识库初始化失败")
            return
        
        # 测试查询
        test_questions = [
            "这个系统的主要功能是什么？",
            "如何使用这个系统？",
            "系统有哪些特点和优势？"
        ]
        
        for question in test_questions:
            result = rag_service.query(question)
            
            print(f"\n{'='*80}")
            print(f"❓ 问题: {result['question']}")
            print(f"📊 找到 {result['source_count']} 个相关文档")
            print(f"⏱️ 处理时间: {result['processing_time']}秒")
            print(f"🤖 回答:\n{result['answer'][:300]}...")
            print(f"{'='*80}")
            
            time.sleep(2)  # 避免API调用过快
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag_service:
            rag_service.close()

if __name__ == "__main__":
    test_rag_service()
