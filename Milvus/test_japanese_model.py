#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日语模型 cl-nagoya/sup-simcse-ja-base
"""

import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_japanese_model():
    """测试日语模型"""
    try:
        logger.info("正在加载日语模型: cl-nagoya/sup-simcse-ja-base")
        model = SentenceTransformer('cl-nagoya/sup-simcse-ja-base')
        
        # 测试文本
        test_texts = [
            "計量装置が傾いている原因として、例３では何が挙げられていますか？",
            "例３：板が腐って計量装置が傾いている。電気は止まっていない。",
            "計量装置類の損傷について説明します。",
            "台風による飛来物で計器BOXが割れた。"
        ]
        
        # 生成嵌入向量
        logger.info("生成嵌入向量...")
        embeddings = model.encode(test_texts)
        
        logger.info(f"模型维度: {embeddings.shape[1]}")
        logger.info(f"嵌入向量形状: {embeddings.shape}")
        
        # 计算相似度
        import numpy as np
        
        # 问题和答案的相似度
        question_emb = embeddings[0]  # 问题
        answer_emb = embeddings[1]    # 例３的答案
        
        similarity = np.dot(question_emb, answer_emb) / (
            np.linalg.norm(question_emb) * np.linalg.norm(answer_emb)
        )
        
        logger.info(f"问题与答案的相似度: {similarity:.4f}")
        
        # 测试其他相似度
        for i, text in enumerate(test_texts[1:], 1):
            sim = np.dot(question_emb, embeddings[i]) / (
                np.linalg.norm(question_emb) * np.linalg.norm(embeddings[i])
            )
            logger.info(f"问题与文本{i}的相似度: {sim:.4f} - {text[:30]}...")
        
        logger.info("✅ 日语模型测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 日语模型测试失败: {e}")
        return False

if __name__ == "__main__":
    test_japanese_model()