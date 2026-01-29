#!/usr/bin/env python3
"""
Text Cleaning Module
Implements the 9-step cleaning process for social media text analysis
"""

import re
import json
from collections import Counter
import string


class TextCleaner:
    """
    9-Step Text Cleaning Process:
    1. Remove duplicate tweets/posts to filter out bots
    2. Remove usernames and links
    3. Remove special characters and punctuation
    4. Exclude meaningless words ("stop" words)
    5. Save text for sentiment analysis
    6. Remove hashtagged words
    7. Tokenize the texts (break into words)
    8. Count word combinations (bigrams)
    9. Convert tokenized words to base form (lemmatization)
    """
    
    def __init__(self):
        # Common English stop words
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
            'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
            'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
            'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
        ])
        
        # Simple lemmatization dictionary (common word forms)
        self.lemma_dict = {
            'farming': 'farm', 'farms': 'farm', 'farmed': 'farm', 'farmer': 'farm', 'farmers': 'farm',
            'technologies': 'technology', 'technological': 'technology',
            'using': 'use', 'used': 'use', 'uses': 'use',
            'cattle': 'cattle', 'cows': 'cow', 'cow': 'cow',
            'dairy': 'dairy', 'dairies': 'dairy',
            'automated': 'automate', 'automation': 'automate', 'automating': 'automate',
            'monitoring': 'monitor', 'monitored': 'monitor', 'monitors': 'monitor',
            'sensors': 'sensor', 'sensing': 'sensor',
            'data': 'data', 'dataset': 'data', 'datasets': 'data',
            'animals': 'animal', 'livestock': 'livestock',
            'managing': 'manage', 'management': 'manage', 'managed': 'manage', 'manager': 'manage',
            'systems': 'system', 'systematic': 'system',
            'precision': 'precision', 'precise': 'precision',
            'smart': 'smart', 'smarter': 'smart', 'smartest': 'smart',
            'artificial': 'ai', 'intelligence': 'ai',
            'learning': 'learn', 'learned': 'learn', 'learns': 'learn',
            'machines': 'machine', 'machinery': 'machine',
            'feeding': 'feed', 'feeds': 'feed', 'fed': 'feed',
            'tracking': 'track', 'tracked': 'track', 'tracks': 'track', 'tracker': 'track',
            'health': 'health', 'healthy': 'health', 'healthcare': 'health',
            'productivity': 'productive', 'produce': 'produce', 'production': 'produce', 'producing': 'produce',
        }
        
        self.cleaned_data = []
        self.all_texts = []
        self.all_tokens = []
        self.bigrams = []
    
    def step1_remove_duplicates(self, data):
        """Step 1: Remove duplicate posts/tweets to filter out bots"""
        print("\n[STEP 1] Removing duplicates...")
        
        seen_texts = set()
        unique_data = []
        duplicates = 0
        
        for item in data:
            text = item.get('text', '').strip().lower()
            title = item.get('title', '').strip().lower()
            
            # Create a signature from text content
            signature = f"{text[:100]}_{title[:100]}"
            
            if signature not in seen_texts and text:
                seen_texts.add(signature)
                unique_data.append(item)
            else:
                duplicates += 1
        
        print(f"  [OK] Removed {duplicates} duplicates, kept {len(unique_data)} unique items")
        return unique_data
    
    def step2_remove_usernames_links(self, text):
        """Step 2: Remove usernames (like @user) and URLs"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove usernames (@mentions)
        text = re.sub(r'@\w+', '', text)
        
        # Remove u/ reddit usernames
        text = re.sub(r'u/\w+', '', text)
        
        # Remove r/ subreddit mentions
        text = re.sub(r'r/\w+', '', text)
        
        return text.strip()
    
    def step3_remove_special_chars(self, text):
        """Step 3: Remove special characters and punctuation"""
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def step4_remove_stop_words(self, tokens):
        """Step 4: Exclude meaningless words (stop words)"""
        return [word for word in tokens if word.lower() not in self.stop_words and len(word) > 2]
    
    def step6_remove_hashtags(self, text):
        """Step 6: Remove hashtagged words"""
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        return text.strip()
    
    def step7_tokenize(self, text):
        """Step 7: Tokenize the text (break into words)"""
        return text.lower().split()
    
    def step8_create_bigrams(self, tokens):
        """Step 8: Count word combinations (bigrams)"""
        if len(tokens) < 2:
            return []
        
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
        
        return bigrams
    
    def step9_lemmatize(self, tokens):
        """Step 9: Convert tokenized words to base form"""
        lemmatized = []
        for token in tokens:
            # Use our simple dictionary, or keep original if not found
            lemmatized.append(self.lemma_dict.get(token, token))
        
        return lemmatized
    
    def clean_all(self, raw_data):
        """Apply all 9 cleaning steps to the raw data"""
        print("\n" + "=" * 80)
        print("STARTING 9-STEP CLEANING PROCESS")
        print("=" * 80)
        
        # Step 1: Remove duplicates
        unique_data = self.step1_remove_duplicates(raw_data)
        
        print("\n[STEPS 2-9] Processing each text item...")
        processed_count = 0
        
        for item in unique_data:
            text = item.get('text', '')
            title = item.get('title', '')
            
            # Combine title and text
            full_text = f"{title}. {text}" if title else text
            
            if not full_text.strip():
                continue
            
            # Step 2: Remove usernames and links
            cleaned_text = self.step2_remove_usernames_links(full_text)
            
            # Step 6: Remove hashtags (doing this before step 3)
            cleaned_text = self.step6_remove_hashtags(cleaned_text)
            
            # Step 3: Remove special characters
            cleaned_text = self.step3_remove_special_chars(cleaned_text)
            
            # Step 5: Save text for sentiment analysis (before tokenization)
            self.all_texts.append(cleaned_text)
            
            # Step 7: Tokenize
            tokens = self.step7_tokenize(cleaned_text)
            
            # Step 4: Remove stop words
            tokens = self.step4_remove_stop_words(tokens)
            
            # Step 9: Lemmatize
            tokens = self.step9_lemmatize(tokens)
            
            # Step 8: Create bigrams
            bigrams = self.step8_create_bigrams(tokens)
            
            # Store cleaned version
            cleaned_item = item.copy()
            cleaned_item['cleaned_text'] = cleaned_text
            cleaned_item['tokens'] = tokens
            cleaned_item['bigrams'] = bigrams
            cleaned_item['token_count'] = len(tokens)
            
            self.cleaned_data.append(cleaned_item)
            self.all_tokens.extend(tokens)
            self.bigrams.extend(bigrams)
            
            processed_count += 1
        
        print(f"  [OK] Processed {processed_count} items")
        print("\n" + "=" * 80)
        print("[SUCCESS] CLEANING COMPLETE!")
        print("=" * 80)
        
        return self.cleaned_data
    
    def get_statistics(self):
        """Get statistics about the cleaned data"""
        stats = {
            'total_items': len(self.cleaned_data),
            'total_tokens': len(self.all_tokens),
            'unique_tokens': len(set(self.all_tokens)),
            'total_bigrams': len(self.bigrams),
            'unique_bigrams': len(set(self.bigrams)),
            'top_words': Counter(self.all_tokens).most_common(30),
            'top_bigrams': Counter(self.bigrams).most_common(20),
            'avg_tokens_per_item': len(self.all_tokens) / len(self.cleaned_data) if self.cleaned_data else 0
        }
        
        return stats
    
    def save_cleaned_data(self, filename='cleaned_data.json'):
        """Save cleaned data to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.cleaned_data, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Cleaned data saved to {filename}")


if __name__ == "__main__":
    # Test the cleaner with sample text
    sample_text = """
    @john Check out this amazing #SmartFarm technology!
    https://example.com/article
    AI and machine learning are revolutionizing livestock management.
    Farmers are using sensors to monitor cattle health 24/7.
    """
    
    cleaner = TextCleaner()
    print("Original:", sample_text)
    print("\nStep 2:", cleaner.step2_remove_usernames_links(sample_text))
    print("Step 3:", cleaner.step3_remove_special_chars(cleaner.step2_remove_usernames_links(sample_text)))
    
    tokens = cleaner.step7_tokenize(cleaner.step3_remove_special_chars(cleaner.step2_remove_usernames_links(sample_text)))
    print("Step 7 (Tokens):", tokens)
    print("Step 4 (No stop words):", cleaner.step4_remove_stop_words(tokens))
    print("Step 9 (Lemmatized):", cleaner.step9_lemmatize(cleaner.step4_remove_stop_words(tokens)))
