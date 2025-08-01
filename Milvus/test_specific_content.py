#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特定内容的智能问答功能
"""

import logging
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_content():
    """测试包含特定内容的智能问答"""
    
    # 测试文本内容
    test_content = """
計量装置類の損傷
記入のポイント
記入例
どこが壊れたのか（計量装置本体・計器BOX・計器板）
どのような状態か（ガラス面割れ・BOX割れ・焼損等）
どのように壊れたのか（子どもがボールを当てた・飛来物接触等）
停電はしていないか

例１：子どもがボールをぶつけて計量装置のガラス面が割れたので直して欲しい。電気は止まっていない。
例２：台風による飛来物で計器BOXが割れたので直して欲しい。電気は止まっていない。
例３：板が腐って計量装置が傾いている。電気は止まっていない。

（注意事項）
・計量装置が外れてひっくり返っている場合は、雨水が浸入し短絡して爆発する恐れがありますので、絶対に触らないように注意喚起してください。
・計器BOXまたは計量装置を取付けている板はお客さま所有のため、費用負担が発生する旨をお伝えてください。
"""
    
    try:
        # 初始化向量嵌入器
        vector_embedder = VectorEmbedder(dimension=384)
        
        # 测试问题
        question = "計量装置が傾いている原因として、例３では何が挙げられていますか？"
        
        logger.info(f"问题: {question}")
        logger.info(f"测试内容长度: {len(test_content)} 字符")
        
        # 直接在文本中搜索答案
        def find_answer_in_text(question, content):
            """在文本中查找答案"""
            lines = content.split('\n')
            
            # 查找例３相关内容
            for i, line in enumerate(lines):
                if '例３' in line or '例3' in line:
                    logger.info(f"找到例３: {line.strip()}")
                    
                    # 提取答案
                    if '計量装置が傾いている' in line:
                        # 查找原因部分
                        if '板が腐って' in line:
                            answer = "例３によると、板が腐って計量装置が傾いているとされています。"
                            return answer
            
            return "例３の内容が見つかりませんでした。"
        
        # 查找答案
        answer = find_answer_in_text(question, test_content)
        logger.info(f"✅ 答案: {answer}")
        
        # 测试语义搜索的效果
        logger.info("\n--- 语义搜索测试 ---")
        
        # 将内容分割成合理的块
        content_chunks = [
            "計量装置類の損傷の記入のポイントについて説明します。",
            "どこが壊れたのか（計量装置本体・計器BOX・計器板）を記入する。",
            "どのような状態か（ガラス面割れ・BOX割れ・焼損等）を記入する。", 
            "どのように壊れたのか（子どもがボールを当てた・飛来物接触等）を記入する。",
            "停電はしていないかを確認する。",
            "例１：子どもがボールをぶつけて計量装置のガラス面が割れたので直して欲しい。電気は止まっていない。",
            "例２：台風による飛来物で計器BOXが割れたので直して欲しい。電気は止まっていない。",
            "例３：板が腐って計量装置が傾いている。電気は止まっていない。",
            "計量装置が外れてひっくり返っている場合は、雨水が浸入し短絡して爆発する恐れがあります。",
            "計器BOXまたは計量装置を取付けている板はお客さま所有のため、費用負担が発生します。"
        ]
        
        # 为每个块生成向量
        logger.info("生成内容向量...")
        chunk_embeddings = []
        for chunk in content_chunks:
            embedding = vector_embedder.embed_text(chunk)
            chunk_embeddings.append(embedding)
        
        # 为问题生成向量
        question_embedding = vector_embedder.embed_text(question)
        
        # 计算相似度
        import numpy as np
        similarities = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            # 计算余弦相似度
            similarity = np.dot(question_embedding, chunk_emb) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(chunk_emb)
            )
            similarities.append((similarity, i, content_chunks[i]))
        
        # 按相似度排序
        similarities.sort(reverse=True)
        
        logger.info("最相关的内容片段:")
        for i, (sim, idx, chunk) in enumerate(similarities[:3]):
            logger.info(f"  {i+1}. 相似度: {sim:.4f}")
            logger.info(f"     内容: {chunk}")
        
        # 检查最相关的内容是否包含答案
        if similarities[0][0] > 0.7 and "例３" in similarities[0][2]:
            logger.info("✅ 语义搜索成功找到相关内容")
        else:
            logger.info("⚠️ 语义搜索可能无法准确匹配")
            
        # 测试不同的问题表述
        logger.info("\n--- 测试不同问题表述 ---")
        alternative_questions = [
            "例３で計量装置が傾く原因は何ですか？",
            "板が腐った場合の問題は何ですか？", 
            "計量装置の傾斜の原因は？",
            "例３について教えてください"
        ]
        
        for alt_q in alternative_questions:
            alt_embedding = vector_embedder.embed_text(alt_q)
            best_match = None
            best_similarity = 0
            
            for i, chunk_emb in enumerate(chunk_embeddings):
                similarity = np.dot(alt_embedding, chunk_emb) / (
                    np.linalg.norm(alt_embedding) * np.linalg.norm(chunk_emb)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = content_chunks[i]
            
            logger.info(f"问题: {alt_q}")
            logger.info(f"最佳匹配 (相似度: {best_similarity:.4f}): {best_match}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("测试特定内容的智能问答功能")
    logger.info("=" * 70)
    test_specific_content()
    logger.info("=" * 70)