#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的智能问答服务
结合关键词匹配和语义搜索
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from pdf_to_milvus import MilvusPDFManager, VectorEmbedder

logger = logging.getLogger(__name__)

class EnhancedQASystem:
    """增强的问答系统"""
    
    def __init__(self, milvus_manager: MilvusPDFManager, vector_embedder: VectorEmbedder):
        self.milvus_manager = milvus_manager
        self.vector_embedder = vector_embedder
    
    def extract_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        keywords = []
        
        # 提取特定编号（例１、例２、例３等）
        example_pattern = r'例[０-９１-９0-9]+|例\d+'
        examples = re.findall(example_pattern, question)
        keywords.extend(examples)
        
        # 提取重要的技术术语
        technical_terms = [
            '計量装置', '計量器', '計量機器',
            '傾いている', '傾斜', '腐って', '腐食',
            '原因', '理由', 'BOX', 'ガラス面',
            '割れ', '損傷', '停電'
        ]
        
        for term in technical_terms:
            if term in question:
                keywords.append(term)
        
        return keywords
    
    def keyword_search(self, keywords: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """基于关键词的精确搜索"""
        all_results = []
        
        for keyword in keywords:
            try:
                results = self.milvus_manager.search_similar(
                    query_text=keyword,
                    top_k=top_k,
                    embedder=self.vector_embedder
                )
                
                # 为每个结果添加关键词匹配信息
                for result in results or []:
                    text = result.get('text', '')
                    # 检查文本中是否包含关键词
                    if keyword in text:
                        result['keyword_match'] = keyword
                        result['exact_match'] = True
                        all_results.append(result)
                    elif any(k in text for k in keywords):
                        result['keyword_match'] = keyword
                        result['exact_match'] = False
                        all_results.append(result)
                        
            except Exception as e:
                logger.error(f"关键词搜索失败 '{keyword}': {e}")
        
        return all_results
    
    def semantic_search(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """语义搜索"""
        try:
            results = self.milvus_manager.search_similar(
                query_text=question,
                top_k=top_k,
                embedder=self.vector_embedder
            )
            
            # 为结果添加语义匹配标记
            for result in results or []:
                result['semantic_match'] = True
            
            return results or []
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []
    
    def combine_and_rank_results(self, keyword_results: List[Dict[str, Any]], 
                                semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并和排序搜索结果"""
        
        # 使用ID去重
        seen_ids = set()
        combined_results = []
        
        # 优先处理关键词精确匹配的结果
        for result in keyword_results:
            result_id = result.get('id', '')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                # 提升精确匹配的分数
                if result.get('exact_match', False):
                    result['final_score'] = result.get('score', 0) + 0.3
                else:
                    result['final_score'] = result.get('score', 0) + 0.1
                combined_results.append(result)
        
        # 添加语义搜索结果
        for result in semantic_results:
            result_id = result.get('id', '')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                result['final_score'] = result.get('score', 0)
                combined_results.append(result)
        
        # 按最终分数排序
        combined_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return combined_results
    
    def generate_answer(self, question: str, results: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict]]:
        """生成答案"""
        
        if not results:
            return "抱歉，没有找到相关信息来回答您的问题。", "low", []
        
        # 查找最相关的结果
        best_results = results[:3]
        sources = []
        answer_parts = []
        
        # 检查是否有精确匹配的结果
        exact_matches = [r for r in best_results if r.get('exact_match', False)]
        
        if exact_matches:
            # 使用精确匹配的结果
            for result in exact_matches[:2]:
                text = result.get('text', '').strip()
                page_num = result.get('page_number', 0)
                keyword = result.get('keyword_match', '')
                
                # 提取相关句子
                sentences = text.split('。')
                relevant_sentences = []
                
                for sentence in sentences:
                    if keyword in sentence or any(k in sentence for k in ['例１', '例２', '例３', '例3']):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    answer_parts.extend(relevant_sentences)
                
                sources.append({
                    'pdf_name': result.get('pdf_name', ''),
                    'page_number': page_num,
                    'score': result.get('final_score', 0),
                    'match_type': 'exact'
                })
        
        if not answer_parts:
            # 如果没有精确匹配，使用语义匹配结果
            for result in best_results[:2]:
                text = result.get('text', '').strip()
                page_num = result.get('page_number', 0)
                
                if len(text) > 20:
                    answer_parts.append(text[:200] + ('...' if len(text) > 200 else ''))
                
                sources.append({
                    'pdf_name': result.get('pdf_name', ''),
                    'page_number': page_num,
                    'score': result.get('final_score', 0),
                    'match_type': 'semantic'
                })
        
        # 生成最终答案
        if answer_parts:
            answer = ' '.join(answer_parts)
            # 判断置信度
            avg_score = sum(r.get('final_score', 0) for r in best_results[:2]) / min(len(best_results), 2)
            
            if exact_matches and avg_score > 0.8:
                confidence = "high"
            elif avg_score > 0.6:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            answer = "抱歉，没有找到充分的相关信息来回答您的问题。"
            confidence = "low"
        
        return answer, confidence, sources
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """回答问题的主要方法"""
        try:
            logger.info(f"增强问答系统处理问题: {question}")
            
            # 1. 提取关键词
            keywords = self.extract_keywords(question)
            logger.info(f"提取的关键词: {keywords}")
            
            # 2. 关键词搜索
            keyword_results = self.keyword_search(keywords) if keywords else []
            logger.info(f"关键词搜索结果: {len(keyword_results)} 个")
            
            # 3. 语义搜索
            semantic_results = self.semantic_search(question)
            logger.info(f"语义搜索结果: {len(semantic_results)} 个")
            
            # 4. 合并和排序结果
            combined_results = self.combine_and_rank_results(keyword_results, semantic_results)
            logger.info(f"合并后结果: {len(combined_results)} 个")
            
            # 5. 生成答案
            answer, confidence, sources = self.generate_answer(question, combined_results)
            
            return {
                'status': 'success',
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'sources': sources,
                'keywords_found': keywords,
                'total_results': len(combined_results)
            }
            
        except Exception as e:
            logger.error(f"增强问答失败: {e}")
            return {
                'status': 'error',
                'question': question,
                'error': str(e)
            }

# 测试函数
def test_enhanced_qa():
    """测试增强问答系统"""
    try:
        # 初始化系统组件（使用日语优化模型）
        milvus_manager = MilvusPDFManager(
            use_lite=True,
            lite_uri="./milvus_japanese.db",
            collection_name="pdf_documents_japanese",
            dimension=768
        )
        
        vector_embedder = VectorEmbedder(
            model_name='cl-nagoya/sup-simcse-ja-base',
            dimension=768
        )
        milvus_manager.load_collection()
        
        # 创建增强问答系统
        qa_system = EnhancedQASystem(milvus_manager, vector_embedder)
        
        # 测试问题
        test_question = "計量装置が傾いている原因として、例３では何が挙げられていますか？"
        
        result = qa_system.answer_question(test_question)
        
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"置信度: {result['confidence']}")
        print(f"找到的关键词: {result.get('keywords_found', [])}")
        print(f"总结果数: {result.get('total_results', 0)}")
        
        if result.get('sources'):
            print("参考来源:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. 页码: {source['page_number']}")
                print(f"     匹配类型: {source['match_type']}")
                print(f"     分数: {source['score']:.4f}")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_enhanced_qa()