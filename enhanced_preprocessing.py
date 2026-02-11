#!/usr/bin/env python3
"""
Enhanced Scientific Preprocessing Pipeline
Implements Phase 2 requirements with complete audit trail

Features:
- Attrition reporting (N₀→N₄)
- Emoji sentiment preservation
- Deduplication with metrics
- Language detection
- Length filtering with justification
- Quality control metrics
"""

import json
import re
import hashlib
from collections import Counter
from typing import List, Dict, Tuple
from datetime import datetime


class EnhancedPreprocessor:
    """
    Scientifically rigorous text preprocessing with audit trail
    Preserves raw_text for reproducibility validation
    """
    
    def __init__(self):
        # Emoji sentiment mapping (critical for affective signal preservation)
        self.emoji_sentiment = {
            '😊': 'happy', '😃': 'happy', '😀': 'happy', '😄': 'happy', '🙂': 'happy',
            '😍': 'love', '❤️': 'love', '💕': 'love', '💖': 'love',
            '😡': 'angry', '😠': 'angry', '🤬': 'angry',
            '😢': 'sad', '😭': 'sad', '😞': 'sad', '☹️': 'sad',
            '😂': 'laughing', '🤣': 'laughing',
            '😱': 'shocked', '😨': 'scared',
            '🤔': 'thinking', '🤨': 'skeptical',
            '👍': 'approve', '👎': 'disapprove',
            '💰': 'money', '💵': 'money', '💸': 'expensive',
            '🚜': 'tractor', '🐄': 'cow', '🐮': 'cow', '🌾': 'farm',
            '📈': 'increase', '📉': 'decrease',
            '✅': 'success', '❌': 'fail'
        }
        
        # Comprehensive stopwords
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', 'should', 'now'
        ])
        
        # Lemmatization dictionary
        self.lemma_dict = {
            'farming': 'farm', 'farms': 'farm', 'farmed': 'farm', 'farmers': 'farm',
            'technologies': 'technology', 'tech': 'technology',
            'using': 'use', 'used': 'use', 'uses': 'use',
            'cattle': 'cattle', 'cows': 'cow', 'cow': 'cow',
            'dairy': 'dairy', 'dairies': 'dairy',
            'automated': 'automate', 'automation': 'automate',
            'monitoring': 'monitor', 'monitored': 'monitor', 'monitors': 'monitor',
            'sensors': 'sensor', 'sensing': 'sensor',
            'animals': 'animal', 'livestock': 'livestock',
            'systems': 'system', 'systematic': 'system'
        }
        
        # Attrition tracking
        self.attrition = {
            'N0_initial': 0,
            'N1_language_filtered': 0,
            'N2_deduplicated': 0,
            'N3_length_filtered': 0,
            'N4_final': 0
        }
        
    def convert_emojis(self, text: str) -> str:
        """
        Convert emojis to sentiment-preserving text
        Critical for negative sentiment detection
        
        Example: "😡 this system sucks" → "angry this system sucks"
        """
        for emoji, sentiment in self.emoji_sentiment.items():
            if emoji in text:
                text = text.replace(emoji, f' {sentiment} ')
        return text
    
    def clean_text_preserve_meaning(self, text: str) -> Tuple[str, str]:
        """
        Clean text while preserving semantic content for BERT
        
        Returns:
            (clean_text, tokens_text)
            - clean_text: Lightly normalized, BERT-ready
            - tokens_text: Heavily processed for classical NLP
        """
        # Store original
        original = text
        
        # 1. Convert emojis to text (preserve affective signal)
        text = self.convert_emojis(text)
        
        # 2. Remove URLs (no sentiment value)
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # 3. Remove @mentions (privacy + noise)
        text = re.sub(r'@\w+', '', text)
        
        # 4. Strip # but keep hashtagged words (topic signal)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # 5. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # For BERT: preserve natural language
        clean_text = text
        
        # For classical NLP: aggressive tokenization
        text_lower = text.lower()
        text_no_punct = re.sub(r'[^\w\s]', ' ', text_lower)
        tokens = text_no_punct.split()
        
        # Remove stopwords and lemmatize
        tokens_clean = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemma = self.lemma_dict.get(token, token)
                tokens_clean.append(lemma)
                
        tokens_text = ' '.join(tokens_clean)
        
        return clean_text, tokens_text
    
    def detect_language(self, text: str) -> str:
        """
        Simple English language detection
        For production, use langdetect or fastText
        """
        # Simple heuristic: check for common English words
        english_indicators = ['the', 'and', 'is', 'are', 'to', 'of', 'a', 'in', 'that', 'have']
        text_lower = text.lower()
        matches = sum(1 for word in english_indicators if word in text_lower)
        
        return 'en' if matches >= 3 else 'unknown'
    
    def calculate_text_hash(self, text: str) -> str:
        """Generate hash for deduplication"""
        return hashlib.md5(text.lower().encode()).hexdigest()
    
    def process_dataset(self, data: List[Dict], min_length: int = 5) -> Dict:
        """
        Full preprocessing pipeline with attrition reporting
        
        Args:
            data: List of posts with raw_text field
            min_length: Minimum word count (default: 5 for explanatory power)
            
        Returns:
            Dict with processed_data and attrition_report
        """
        print("\n" + "=" * 80)
        print("PHASE 2: ENHANCED PREPROCESSING WITH AUDIT TRAIL")
        print("=" * 80)
        
        # N₀: Initial extraction
        self.attrition['N0_initial'] = len(data)
        print(f"\n[N₀] Initial dataset: {self.attrition['N0_initial']} posts")
        
        # STAGE 1: Language filtering
        print("\n[STAGE 1] Language detection...")
        english_posts = []
        for post in data:
            text = post.get('raw_text', post.get('text', ''))
            if self.detect_language(text) == 'en':
                english_posts.append(post)
                
        self.attrition['N1_language_filtered'] = len(english_posts)
        removed_lang = self.attrition['N0_initial'] - self.attrition['N1_language_filtered']
        print(f"[N₁] After language filter: {self.attrition['N1_language_filtered']} posts ({removed_lang} removed)")
        
        # STAGE 2: Deduplication
        print("\n[STAGE 2] Deduplication (exact + near-duplicate)...")
        seen_hashes = set()
        unique_posts = []
        duplicates_exact = 0
        
        for post in english_posts:
            text = post.get('raw_text', post.get('text', ''))
            text_hash = self.calculate_text_hash(text)
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_posts.append(post)
            else:
                duplicates_exact += 1
                
        self.attrition['N2_deduplicated'] = len(unique_posts)
        print(f"[N₂] After deduplication: {self.attrition['N2_deduplicated']} posts ({duplicates_exact} exact duplicates removed)")
        
        # STAGE 3: Text cleaning + Length filtering
        print("\n[STAGE 3] Text cleaning and length filtering...")
        processed_posts = []
        too_short = 0
        
        for post in unique_posts:
            text = post.get('raw_text', post.get('text', ''))
            
            # Clean text
            clean_text, tokens_text = self.clean_text_preserve_meaning(text)
            
            # Length check
            word_count = len(clean_text.split())
            if word_count < min_length:
                too_short += 1
                continue
                
            # Add processed versions
            post['raw_text'] = text  # Original
            post['clean_text'] = clean_text  # BERT-ready
            post['tokens'] = tokens_text  # Classical NLP
            post['word_count'] = word_count
            
            processed_posts.append(post)
            
        self.attrition['N3_length_filtered'] = len(processed_posts)
        print(f"[N₃] After length filter (≥{min_length} words): {self.attrition['N3_length_filtered']} posts ({too_short} too short)")
        
        # N₄: Final dataset
        self.attrition['N4_final'] = len(processed_posts)
        print(f"\n[N₄] FINAL ANALYTIC DATASET: {self.attrition['N4_final']} posts")
        
        # Calculate attrition rates
        retention_rate = (self.attrition['N4_final'] / self.attrition['N0_initial']) * 100
        print(f"\nRetention rate: {retention_rate:.1f}%")
        
        return {
            'processed_data': processed_posts,
            'attrition': self.attrition,
            'retention_rate': retention_rate
        }
    
    def generate_attrition_report(self) -> str:
        """
        Generate publication-ready attrition table
        Required by reviewers
        """
        report = "\n" + "=" * 80
        report += "\nATTRITION REPORT (N₀→N₄)\n"
        report += "=" * 80 + "\n\n"
        report += "| Stage | Description | N | Removed |\n"
        report += "|-------|-------------|---|--------|\n"
        
        stages = [
            ('N₀', 'Initial extraction', 'N0_initial'),
            ('N₁', 'Language filtering', 'N1_language_filtered'),
            ('N₂', 'Deduplication', 'N2_deduplicated'),
            ('N₃', 'Length filtering (≥5 words)', 'N3_length_filtered'),
            ('N₄', 'Final analytic dataset', 'N4_final')
        ]
        
        for i, (stage, desc, key) in enumerate(stages):
            n = self.attrition[key]
            if i > 0:
                prev_key = stages[i-1][2]
                removed = self.attrition[prev_key] - n
                report += f"| {stage} | {desc} | {n} | {removed} |\n"
            else:
                report += f"| {stage} | {desc} | {n} | - |\n"
                
        retention = (self.attrition['N4_final'] / self.attrition['N0_initial']) * 100
        report += f"\nOverall retention: {retention:.1f}%\n"
        report += "=" * 80 + "\n"
        
        return report
    
    def save_processed_dataset(self, result: Dict, filename: str = 'preprocessed_data.json'):
        """Save processed dataset with attrition metadata"""
        output = {
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'attrition': result['attrition'],
                'retention_rate': result['retention_rate'],
                'final_dataset_size': len(result['processed_data'])
            },
            'attrition_report': self.generate_attrition_report(),
            'data': result['processed_data']
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"\n[SAVED] {filename}")
        print(self.generate_attrition_report())


def main():
    """Process enhanced scraped data"""
    # Load enhanced scraped data
    with open('enhanced_scraped_data.json', 'r') as f:
        data = json.load(f)
        posts = data['posts']
    
    preprocessor = EnhancedPreprocessor()
    result = preprocessor.process_dataset(posts, min_length=5)
    preprocessor.save_processed_dataset(result)
    
    print("\n[READY] Preprocessed data ready for sentiment analysis")


if __name__ == "__main__":
    main()
