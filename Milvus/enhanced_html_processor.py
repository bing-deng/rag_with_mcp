#!/usr/bin/env python3
"""
Enhanced HTML Content Processing System
Optimized data preprocessing, chunking strategies, and embedding quality for AI training
"""

import os
import re
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter

# HTML parsing
from bs4 import BeautifulSoup

# Milvus
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, 
    DataType, Collection, MilvusException
)

# Advanced text processing
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

@dataclass
class ContentBlock:
    """Enhanced content block with quality metrics"""
    content: str
    content_type: str
    url: str
    title: str
    word_count: int
    url_hash: str
    timestamp: int
    
    # Quality metrics
    semantic_density: float  # Information density score
    business_relevance: float  # Business context relevance
    readability_score: float  # Text readability
    duplicate_score: float  # Similarity to other blocks
    
    # Context information
    parent_heading: str  # Nearest parent heading
    section_context: str  # Section context
    extracted_entities: List[str]  # Named entities
    key_phrases: List[str]  # Important phrases

class EnhancedHTMLProcessor:
    """Enhanced HTML processor with advanced chunking and quality assessment"""
    
    def __init__(self, host='localhost', port='19530', collection_name='enhanced_content'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.dimension = 384
        
        # Initialize models
        self._initialize_models()
        
        # Content quality thresholds
        self.quality_thresholds = {
            'min_word_count': 5,
            'max_word_count': 500,
            'min_semantic_density': 0.3,
            'max_duplicate_score': 0.8,
            'min_readability': 0.2
        }
        
        # Business domain keywords (Japanese focus)
        self.business_keywords = {
            'financial': ['売上', '利益', '売上高', '営業利益', '純利益', '資本金', '負債', '資産', '株価', '配当', '財務', '決算'],
            'company': ['会社', '企業', '法人', '株式会社', '設立', '創立', '従業員', '本社', '支社', '事業所', '代表', '役員'],
            'business': ['事業', 'サービス', '製品', '技術', 'ソリューション', 'プロジェクト', '建設', '工事', '施工', '設計'],
            'performance': ['業績', '実績', '成果', '達成', '向上', '改善', '効果', '効率', '品質', '安全'],
            'governance': ['ガバナンス', 'CSR', '持続可能', 'サステナビリティ', 'コンプライアンス', '内部統制']
        }
        
        # Semantic chunking parameters
        self.semantic_chunk_config = {
            'max_chunk_size': 800,
            'overlap_size': 100,
            'similarity_threshold': 0.7,
            'preserve_sentence_boundary': True
        }
    
    def _initialize_models(self):
        """Initialize NLP models"""
        if HAS_SENTENCE_TRANSFORMERS:
            print("🔧 Loading enhanced embedding model...")
            # Use a more powerful multilingual model
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            self.dimension = 768  # Updated dimension for better model
            print("✅ Enhanced embedding model loaded")
        else:
            self.embedding_model = None
            print("⚠️  Using simplified embeddings")
        
        if HAS_SPACY:
            try:
                print("🔧 Loading Japanese NLP model...")
                self.nlp = spacy.load("ja_core_news_sm")
                print("✅ Japanese NLP model loaded")
            except OSError:
                print("⚠️  Japanese spaCy model not found, install with: python -m spacy download ja_core_news_sm")
                self.nlp = None
        else:
            self.nlp = None
    
    def connect_to_milvus(self):
        """Connect to Milvus with enhanced error handling"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ Connected to Milvus: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def create_enhanced_collection(self):
        """Create enhanced collection with additional quality fields"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="word_count", dtype=DataType.INT64),
            FieldSchema(name="url_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            
            # Quality metrics
            FieldSchema(name="semantic_density", dtype=DataType.FLOAT),
            FieldSchema(name="business_relevance", dtype=DataType.FLOAT),
            FieldSchema(name="readability_score", dtype=DataType.FLOAT),
            FieldSchema(name="duplicate_score", dtype=DataType.FLOAT),
            
            # Context information
            FieldSchema(name="parent_heading", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="section_context", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="extracted_entities", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="key_phrases", dtype=DataType.VARCHAR, max_length=500),
            
            # Enhanced embedding
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(fields, "Enhanced HTML content with quality metrics")
        
        if utility.has_collection(self.collection_name):
            print(f"📝 Collection {self.collection_name} exists, dropping...")
            utility.drop_collection(self.collection_name)
        
        self.collection = Collection(self.collection_name, schema)
        print(f"✅ Enhanced collection {self.collection_name} created")
        return self.collection
    
    def create_optimized_index(self):
        """Create optimized vector index with better parameters"""
        # Optimized HNSW parameters for better performance
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 32,  # Increased for better recall
                "efConstruction": 400,  # Increased for better quality
            }
        }
        
        print("🔧 Creating optimized index...")
        self.collection.create_index("embedding", index_params)
        
        # Create scalar field indexes for faster filtering
        self.collection.create_index("content_type")
        self.collection.create_index("business_relevance")
        self.collection.create_index("semantic_density")
        
        print("✅ Optimized indexes created")
    
    def semantic_text_chunking(self, text: str, parent_heading: str = "") -> List[str]:
        """Advanced semantic-aware text chunking"""
        if len(text) <= self.semantic_chunk_config['max_chunk_size']:
            return [text]
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed the limit
            if current_size + sentence_size > self.semantic_chunk_config['max_chunk_size'] and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += " " + sentence
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with Japanese support"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback sentence splitting for Japanese text
            sentences = re.split(r'[。！？]', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        overlap_words = words[-self.semantic_chunk_config['overlap_size']//5:]  # Approximate word overlap
        return " ".join(overlap_words)
    
    def calculate_semantic_density(self, content: str) -> float:
        """Calculate semantic information density"""
        if not content:
            return 0.0
        
        # Count meaningful content indicators
        word_count = len(content.split())
        if word_count == 0:
            return 0.0
        
        # Business keyword density
        business_words = 0
        for keywords in self.business_keywords.values():
            for keyword in keywords:
                business_words += content.count(keyword)
        
        # Named entity density (if spaCy available)
        entity_count = 0
        if self.nlp:
            doc = self.nlp(content)
            entity_count = len(doc.ents)
        
        # Numeric information density
        numeric_patterns = len(re.findall(r'\d+', content))
        
        # Calculate composite density score
        business_density = min(business_words / word_count, 0.5)
        entity_density = min(entity_count / max(word_count/10, 1), 0.3)
        numeric_density = min(numeric_patterns / word_count, 0.2)
        
        total_density = business_density + entity_density + numeric_density
        return min(total_density, 1.0)
    
    def calculate_business_relevance(self, content: str, content_type: str) -> float:
        """Calculate business context relevance score"""
        relevance_score = 0.0
        content_lower = content.lower()
        
        # Content type base score
        type_scores = {
            'company_info': 0.8,
            'business_div': 0.7,
            'financial_data': 0.9,
            'key_data': 0.6,
            'definition_list': 0.5,
            'table': 0.6,
            'heading_h1': 0.4,
            'heading_h2': 0.3
        }
        relevance_score += type_scores.get(content_type, 0.1)
        
        # Keyword-based relevance
        for category, keywords in self.business_keywords.items():
            category_score = sum(0.1 for keyword in keywords if keyword in content)
            if category == 'financial':
                category_score *= 1.5  # Boost financial content
            elif category == 'company':
                category_score *= 1.3
            relevance_score += min(category_score, 0.5)
        
        # Business pattern boosting
        business_patterns = [
            r'\d+年度',  # Financial year
            r'\d+(?:億|万|千)円',  # Money amounts
            r'\d+%',  # Percentages
            r'代表取締役',  # Corporate titles
            r'株式会社',  # Corporation
        ]
        
        for pattern in business_patterns:
            if re.search(pattern, content):
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def calculate_readability_score(self, content: str) -> float:
        """Calculate text readability score"""
        if not content:
            return 0.0
        
        words = content.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Sentence count
        sentences = self._split_into_sentences(content)
        sentence_count = len(sentences)
        
        if sentence_count == 0:
            return 0.0
        
        # Average words per sentence
        avg_words_per_sentence = word_count / sentence_count
        
        # Character complexity (simplified for Japanese)
        char_count = len(content)
        avg_chars_per_word = char_count / word_count if word_count > 0 else 0
        
        # Readability score (simplified formula)
        # Lower values for very short/long sentences and words
        sentence_score = 1.0 / (1.0 + abs(avg_words_per_sentence - 15) / 10)
        word_score = 1.0 / (1.0 + abs(avg_chars_per_word - 4) / 3)
        
        readability = (sentence_score + word_score) / 2
        return min(readability, 1.0)
    
    def extract_entities_and_phrases(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract named entities and key phrases"""
        entities = []
        key_phrases = []
        
        if self.nlp:
            doc = self.nlp(content)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'PERCENT', 'DATE']:
                    entities.append(f"{ent.text}({ent.label_})")
            
            # Extract noun phrases as key phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    key_phrases.append(chunk.text)
        
        # Fallback entity extraction with regex
        if not entities:
            # Extract monetary amounts
            money_matches = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?[億万千百十]?円', content)
            entities.extend([f"{m}(MONEY)" for m in money_matches[:3]])
            
            # Extract percentages
            percent_matches = re.findall(r'\d+(?:\.\d+)?%', content)
            entities.extend([f"{p}(PERCENT)" for p in percent_matches[:3]])
            
            # Extract years
            year_matches = re.findall(r'\d{4}年', content)
            entities.extend([f"{y}(DATE)" for y in year_matches[:3]])
        
        return entities[:5], key_phrases[:5]  # Limit to prevent field overflow
    
    def calculate_duplicate_score(self, content: str, existing_contents: List[str]) -> float:
        """Calculate similarity to existing content (simplified)"""
        if not existing_contents or not self.embedding_model:
            return 0.0
        
        try:
            # Calculate embeddings
            current_embedding = self.embedding_model.encode([content])
            existing_embeddings = self.embedding_model.encode(existing_contents)
            
            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(current_embedding, existing_embeddings)[0]
            
            # Return maximum similarity
            return float(np.max(similarities)) if len(similarities) > 0 else 0.0
            
        except Exception:
            # Fallback to simple text similarity
            max_similarity = 0.0
            current_words = set(content.lower().split())
            
            for existing_content in existing_contents:
                existing_words = set(existing_content.lower().split())
                intersection = len(current_words.intersection(existing_words))
                union = len(current_words.union(existing_words))
                
                if union > 0:
                    jaccard_similarity = intersection / union
                    max_similarity = max(max_similarity, jaccard_similarity)
            
            return max_similarity
    
    def enhanced_parse_html(self, html_content: str, base_url: str = "") -> List[ContentBlock]:
        """Enhanced HTML parsing with quality assessment"""
        print("🔍 Enhanced HTML parsing...")
        
        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
            tag.decompose()
        
        # Extract page title
        title = soup.find('title')
        page_title = title.get_text().strip() if title else "無題"
        
        content_blocks = []
        existing_contents = []  # For duplicate detection
        current_heading = ""
        current_section = ""
        
        # Process all content elements with context
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li', 'td', 'th', 'dt', 'dd']):
            
            # Update context for headings
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                current_heading = self._clean_text(element.get_text())
                if element.name in ['h1', 'h2']:
                    current_section = current_heading
                continue
            
            # Extract and process content
            raw_content = element.get_text()
            cleaned_content = self._clean_text(raw_content)
            
            if not cleaned_content or len(cleaned_content) < self.quality_thresholds['min_word_count']:
                continue
            
            # Determine content type
            content_type = self._determine_enhanced_content_type(element, cleaned_content)
            
            # Semantic chunking for long content
            chunks = self.semantic_text_chunking(cleaned_content, current_heading)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.split()) < self.quality_thresholds['min_word_count']:
                    continue
                
                # Calculate quality metrics
                semantic_density = self.calculate_semantic_density(chunk)
                business_relevance = self.calculate_business_relevance(chunk, content_type)
                readability_score = self.calculate_readability_score(chunk)
                duplicate_score = self.calculate_duplicate_score(chunk, existing_contents[-10:])  # Check last 10
                
                # Quality filtering
                if (semantic_density < self.quality_thresholds['min_semantic_density'] or
                    duplicate_score > self.quality_thresholds['max_duplicate_score'] or
                    readability_score < self.quality_thresholds['min_readability']):
                    continue
                
                # Extract entities and phrases
                entities, key_phrases = self.extract_entities_and_phrases(chunk)
                
                # Create enhanced content block
                chunk_content_type = content_type if len(chunks) == 1 else f"{content_type}_chunk_{i+1}"
                
                block = ContentBlock(
                    content=chunk,
                    content_type=chunk_content_type,
                    url=base_url,
                    title=page_title,
                    word_count=len(chunk.split()),
                    url_hash=hashlib.md5(base_url.encode()).hexdigest(),
                    timestamp=int(time.time()),
                    semantic_density=semantic_density,
                    business_relevance=business_relevance,
                    readability_score=readability_score,
                    duplicate_score=duplicate_score,
                    parent_heading=current_heading,
                    section_context=current_section,
                    extracted_entities=entities,
                    key_phrases=key_phrases
                )
                
                content_blocks.append(block)
                existing_contents.append(chunk)
        
        print(f"✅ Enhanced parsing complete: {len(content_blocks)} quality content blocks")
        return content_blocks
    
    def _determine_enhanced_content_type(self, element, content: str) -> str:
        """Determine enhanced content type with business context"""
        # Business-specific type detection
        if any(keyword in content for keyword in ['設立', '創立', '資本金', '従業員', '本社']):
            return 'company_info'
        
        if any(keyword in content for keyword in ['売上', '利益', '売上高', '営業利益', '純利益']):
            return 'financial_data'
        
        if re.search(r'\d+(?:億|万|千)円|\d+%|\d+年度', content):
            return 'key_data'
        
        # Element-based type detection
        if element.name == 'p':
            return 'paragraph'
        elif element.name in ['dt', 'dd']:
            return 'definition_list'
        elif element.name in ['td', 'th']:
            return 'table'
        elif element.name == 'li':
            return 'list'
        elif element.name == 'div':
            # Check div classes/ids for business context
            classes = element.get('class', [])
            div_id = element.get('id', '')
            
            if any('company' in str(cls).lower() for cls in classes) or 'company' in div_id.lower():
                return 'company_div'
            elif any('business' in str(cls).lower() for cls in classes) or 'business' in div_id.lower():
                return 'business_div'
            else:
                return 'content_div'
        
        return 'unknown'
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but preserve Japanese text
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f\uff00-\uffef.,!?;:()"\'-]', '', text)
        
        # Remove very short or repetitive content
        if len(text) < 3 or len(set(text.split())) < 2:
            return ""
        
        return text
    
    def enhanced_text_to_vector(self, text: str) -> List[float]:
        """Enhanced text vectorization with quality validation"""
        if HAS_SENTENCE_TRANSFORMERS and self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                
                # Validate embedding quality
                if np.any(np.isnan(embedding)) or np.all(embedding == 0):
                    print(f"⚠️  Invalid embedding detected, using fallback")
                    return self._fallback_vectorization(text)
                
                return embedding.tolist()
            except Exception as e:
                print(f"⚠️  Embedding failed: {e}, using fallback")
                return self._fallback_vectorization(text)
        else:
            return self._fallback_vectorization(text)
    
    def _fallback_vectorization(self, text: str) -> List[float]:
        """Improved fallback vectorization"""
        words = text.lower().split()
        
        # Create more sophisticated fallback vector
        vector = np.random.normal(0, 0.1, self.dimension)
        
        # Add deterministic components based on text features
        word_hash = hash(' '.join(sorted(set(words)))) % 1000000
        np.random.seed(word_hash)
        
        # Length-based adjustment
        length_factor = min(len(words) / 50.0, 1.0)
        vector = vector * length_factor
        
        # Business keyword boosting
        business_boost = 0
        for keywords in self.business_keywords.values():
            for keyword in keywords:
                if keyword in text:
                    business_boost += 0.1
        
        if business_boost > 0:
            boost_vector = np.random.normal(business_boost, 0.05, self.dimension)
            vector = vector + boost_vector
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def insert_enhanced_content(self, content_blocks: List[ContentBlock]) -> bool:
        """Insert enhanced content blocks with quality metrics"""
        if not content_blocks:
            print("⚠️  No content blocks to insert")
            return False
        
        print(f"📝 Inserting {len(content_blocks)} enhanced content blocks...")
        
        # Prepare data arrays
        data_arrays = [[] for _ in range(16)]  # Number of fields
        
        for i, block in enumerate(content_blocks):
            if i % 100 == 0:
                print(f"  📊 Processing block {i+1}/{len(content_blocks)}")
            
            # Truncate fields to fit schema limits
            content = block.content[:14900] if len(block.content) > 14900 else block.content
            title = block.title[:497] if len(block.title) > 497 else block.title
            url = block.url[:997] if len(block.url) > 997 else block.url
            
            # Convert lists to strings
            entities_str = "|".join(block.extracted_entities)[:497]
            phrases_str = "|".join(block.key_phrases)[:497]
            
            data_arrays[0].append(url)
            data_arrays[1].append(title)
            data_arrays[2].append(content)
            data_arrays[3].append(block.content_type)
            data_arrays[4].append(block.word_count)
            data_arrays[5].append(block.url_hash)
            data_arrays[6].append(block.timestamp)
            data_arrays[7].append(block.semantic_density)
            data_arrays[8].append(block.business_relevance)
            data_arrays[9].append(block.readability_score)
            data_arrays[10].append(block.duplicate_score)
            data_arrays[11].append(block.parent_heading[:197] if block.parent_heading else "")
            data_arrays[12].append(block.section_context[:297] if block.section_context else "")
            data_arrays[13].append(entities_str)
            data_arrays[14].append(phrases_str)
            data_arrays[15].append(self.enhanced_text_to_vector(content))
        
        try:
            insert_result = self.collection.insert(data_arrays)
            self.collection.flush()
            print(f"✅ Successfully inserted {insert_result.insert_count} enhanced content blocks")
            return True
        except Exception as e:
            print(f"❌ Enhanced insertion failed: {e}")
            return False
    
    def get_enhanced_statistics(self) -> Dict:
        """Get enhanced collection statistics"""
        if not self.collection:
            return {}
        
        try:
            stats = {
                "collection_name": self.collection.name,
                "total_blocks": self.collection.num_entities,
                "dimension": self.dimension,
                "has_enhanced_features": True
            }
            
            # Get quality distribution
            quality_data = self.collection.query(
                expr="id >= 0",
                output_fields=["content_type", "semantic_density", "business_relevance", "readability_score"],
                limit=1000
            )
            
            if quality_data:
                # Content type distribution
                type_counts = Counter(item.get('content_type', 'unknown') for item in quality_data)
                stats["content_type_distribution"] = dict(type_counts)
                
                # Quality metrics
                semantic_scores = [item.get('semantic_density', 0) for item in quality_data]
                business_scores = [item.get('business_relevance', 0) for item in quality_data]
                readability_scores = [item.get('readability_score', 0) for item in quality_data]
                
                stats["quality_metrics"] = {
                    "avg_semantic_density": np.mean(semantic_scores),
                    "avg_business_relevance": np.mean(business_scores),
                    "avg_readability": np.mean(readability_scores),
                    "high_quality_blocks": sum(1 for s in semantic_scores if s > 0.5)
                }
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Failed to get enhanced statistics: {e}")
            return {"error": str(e)}
    
    def load_collection(self):
        """Load collection with optimized parameters"""
        print("📥 Loading enhanced collection...")
        self.collection.load()
        print("✅ Enhanced collection loaded")
    
    def disconnect(self):
        """Disconnect from Milvus"""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        print("🔌 Disconnected from Milvus")

def main():
    """Demonstration of enhanced HTML processing"""
    print("🚀 Enhanced HTML Content Processing System")
    print("=" * 70)
    print("Features:")
    print("• Semantic-aware text chunking")
    print("• Quality-based content filtering")
    print("• Business context analysis")
    print("• Advanced entity extraction")
    print("• Duplicate content detection")
    print("• Enhanced embedding quality")
    print("-" * 70)
    
    # Create processor
    processor = EnhancedHTMLProcessor(collection_name="enhanced_kandenko")
    
    try:
        # Connect and setup
        if not processor.connect_to_milvus():
            return
        
        processor.create_enhanced_collection()
        processor.create_optimized_index()
        processor.load_collection()
        
        print("\n✅ Enhanced HTML processing system ready!")
        print("\nTo use this processor:")
        print("1. Replace html_content with your actual HTML")
        print("2. Use processor.enhanced_parse_html(html_content, url)")
        print("3. Use processor.insert_enhanced_content(content_blocks)")
        print("4. Use processor.get_enhanced_statistics() for quality metrics")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        processor.disconnect()

if __name__ == "__main__":
    main()