#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日语模型设置
"""

import logging
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_to_milvus import VectorEmbedder, MilvusPDFManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_japanese_model_setup():
    """测试日语模型设置"""
    try:
        logger.info("🚀 测试日语模型设置")
        
        # 1. 测试向量嵌入器
        logger.info("1️⃣ 测试向量嵌入器...")
        embedder = VectorEmbedder(
            model_name='cl-nagoya/sup-simcse-ja-base',
            dimension=768
        )
        
        # 测试文本
        test_texts = [
            "計量装置が傾いている原因として、例３では何が挙げられていますか？",
            "例３：板が腐って計量装置が傾いている。電気は止まっていない。",
            "計量装置類の損傷について記入します。"
        ]
        
        logger.info("生成嵌入向量...")
        embeddings = []
        for text in test_texts:
            emb = embedder.embed_text(text)
            embeddings.append(emb)
            logger.info(f"文本: {text[:30]}... -> 向量维度: {len(emb)}")
        
        # 2. 计算相似度
        logger.info("2️⃣ 计算文本相似度...")
        import numpy as np
        
        # 问题和答案的相似度
        q_emb = np.array(embeddings[0])
        a_emb = np.array(embeddings[1])
        
        similarity = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
        logger.info(f"问题与答案相似度: {similarity:.4f}")
        
        if similarity > 0.5:
            logger.info("✅ 日语模型语义理解良好")
        else:
            logger.warning("⚠️ 相似度较低，可能需要调整")
        
        # 3. 测试数据库连接
        logger.info("3️⃣ 测试数据库连接...")
        milvus_manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_japanese.db",
            collection_name="pdf_documents_japanese",
            dimension=768
        )
        
        logger.info("✅ 数据库连接成功")
        
        # 4. 模型规格总结
        logger.info("4️⃣ 模型规格总结:")
        logger.info(f"   模型名称: cl-nagoya/sup-simcse-ja-base")
        logger.info(f"   向量维度: 768")
        logger.info(f"   数据库文件: ./milvus_japanese.db")
        logger.info(f"   集合名称: pdf_documents_japanese")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_japanese_data():
    """创建日语样本数据用于测试"""
    try:
        logger.info("📝 创建日语样本数据...")
        
        # 初始化组件
        embedder = VectorEmbedder(
            model_name='cl-nagoya/sup-simcse-ja-base',
            dimension=768
        )
        
        milvus_manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_japanese.db",
            collection_name="pdf_documents_japanese",
            dimension=768
        )
        
        # 样本数据（基于之前提供的内容）
        sample_data = [
            {
                'text': '計量装置類の損傷の記入のポイントについて説明します。どこが壊れたのか（計量装置本体・計器BOX・計器板）を記入してください。',
                'page_number': 1,
                'chunk_type': 'text'
            },
            {
                'text': 'どのような状態か（ガラス面割れ・BOX割れ・焼損等）を詳しく記入してください。',
                'page_number': 1,
                'chunk_type': 'text'
            },
            {
                'text': 'どのように壊れたのか（子どもがボールを当てた・飛来物接触等）の原因を記入してください。',
                'page_number': 1,
                'chunk_type': 'text'
            },
            {
                'text': '停電はしていないかを必ず確認してください。',
                'page_number': 1,
                'chunk_type': 'text'
            },
            {
                'text': '例１：子どもがボールをぶつけて計量装置のガラス面が割れたので直して欲しい。電気は止まっていない。',
                'page_number': 2,
                'chunk_type': 'text'
            },
            {
                'text': '例２：台風による飛来物で計器BOXが割れたので直して欲しい。電気は止まっていない。',
                'page_number': 2,
                'chunk_type': 'text'
            },
            {
                'text': '例３：板が腐って計量装置が傾いている。電気は止まっていない。',
                'page_number': 2,
                'chunk_type': 'text'
            },
            {
                'text': '計量装置が外れてひっくり返っている場合は、雨水が浸入し短絡して爆発する恐れがありますので、絶対に触らないように注意喚起してください。',
                'page_number': 3,
                'chunk_type': 'text'
            },
            {
                'text': '計器BOXまたは計量装置を取付けている板はお客さま所有のため、費用負担が発生する旨をお伝えください。',
                'page_number': 3,
                'chunk_type': 'text'
            }
        ]
        
        # 生成数据块
        from pdf_to_milvus import PDFChunk
        import hashlib
        import time
        
        chunks = []
        for i, data in enumerate(sample_data):
            # 生成ID
            content = f"sample_{i}_{time.time()}"
            chunk_id = hashlib.md5(content.encode()).hexdigest()
            
            # 生成嵌入向量
            embedding = embedder.embed_text(data['text'])
            
            chunk = PDFChunk(
                id=chunk_id,
                text=data['text'],
                page_number=data['page_number'],
                chunk_index=i,
                pdf_path="sample_japanese_document.pdf",
                pdf_name="日语样本文档",
                chunk_type=data['chunk_type'],
                metadata={'source': 'sample_data'},
                embedding=embedding
            )
            chunks.append(chunk)
        
        # 存储到数据库
        logger.info("存储样本数据到数据库...")
        milvus_manager.insert_chunks(chunks)
        
        # 获取统计信息
        stats = milvus_manager.get_collection_stats()
        logger.info(f"✅ 样本数据创建完成，数据库包含 {stats.get('num_entities', 0)} 个文档块")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建样本数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎯 日语模型设置测试")
    logger.info("=" * 60)
    
    # 1. 测试基本设置
    if test_japanese_model_setup():
        logger.info("✅ 基本设置测试通过")
        
        # 2. 创建样本数据
        if create_sample_japanese_data():
            logger.info("✅ 样本数据创建成功")
            logger.info("🎉 日语模型设置完成！")
        else:
            logger.error("❌ 样本数据创建失败")
    else:
        logger.error("❌ 基本设置测试失败")
    
    logger.info("=" * 60)