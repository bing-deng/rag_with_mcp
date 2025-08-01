#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移到日语优化模型 cl-nagoya/sup-simcse-ja-base
"""

import os
import logging
import shutil
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder, PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_to_japanese_model():
    """迁移到日语模型"""
    try:
        logger.info("🚀 开始迁移到日语优化模型")
        
        # 备份原数据库
        if os.path.exists("./milvus_demo.db"):
            backup_path = "./milvus_demo_backup.db"
            shutil.copy2("./milvus_demo.db", backup_path)
            logger.info(f"✅ 原数据库已备份到: {backup_path}")
        
        # 创建新的日语优化向量嵌入器
        logger.info("🔧 初始化日语向量嵌入器...")
        japanese_embedder = VectorEmbedder(
            model_name='cl-nagoya/sup-simcse-ja-base',
            dimension=768  # 日语模型的维度
        )
        
        # 创建新的Milvus管理器（768维）
        logger.info("🗄️ 创建新的Milvus数据库...")
        japanese_milvus = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_japanese.db",
            collection_name="pdf_documents_japanese",
            dimension=768
        )
        
        # 检查是否有PDF文件需要重新处理
        pdf_dir = "./pdf"
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            if pdf_files:
                logger.info(f"📄 找到 {len(pdf_files)} 个PDF文件，开始重新处理...")
                
                pdf_processor = PDFProcessor()
                
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(pdf_dir, pdf_file)
                    logger.info(f"处理文件: {pdf_file}")
                    
                    # 提取和处理PDF
                    pages_content = pdf_processor.extract_text_from_pdf(pdf_path)
                    chunks = pdf_processor.create_chunks(pages_content, pdf_path)
                    
                    # 使用日语模型生成嵌入
                    chunks_with_embeddings = japanese_embedder.embed_chunks(chunks)
                    
                    # 存储到新数据库
                    japanese_milvus.store_chunks(chunks_with_embeddings)
                    
                    logger.info(f"✅ {pdf_file} 处理完成")
                
                # 获取统计信息
                stats = japanese_milvus.get_collection_stats()
                logger.info(f"✅ 迁移完成！新数据库包含 {stats.get('num_entities', 0)} 个文档块")
            else:
                logger.warning("⚠️ 未找到PDF文件，创建空的日语数据库")
        else:
            logger.warning("⚠️ PDF目录不存在，创建空的日语数据库")
        
        # 测试日语问答
        logger.info("🧪 测试日语问答功能...")
        test_question = "計量装置が傾いている原因として、例３では何が挙げられていますか？"
        
        try:
            japanese_milvus.load_collection()
            results = japanese_milvus.search_similar(
                query_text=test_question,
                top_k=5,
                embedder=japanese_embedder
            )
            
            if results:
                logger.info(f"✅ 找到 {len(results)} 个相关结果")
                for i, result in enumerate(results[:2], 1):
                    score = result.get('score', 0)
                    text = result.get('text', '')[:100]
                    logger.info(f"  结果 {i}: 分数={score:.4f}, 内容={text}...")
            else:
                logger.info("🔍 未找到相关结果（这是正常的，如果数据库为空）")
                
        except Exception as e:
            logger.warning(f"测试查询时出错: {e}")
        
        logger.info("🎉 日语模型迁移完成！")
        logger.info("📝 使用新配置:")
        logger.info(f"   - 模型: cl-nagoya/sup-simcse-ja-base")
        logger.info(f"   - 维度: 768")
        logger.info(f"   - 数据库: ./milvus_japanese.db")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 迁移失败: {e}")
        return False

if __name__ == "__main__":
    migrate_to_japanese_model()