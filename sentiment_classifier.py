"""
Sentiment Classifier - Apply trained model to scraped data
This script loads the trained sentiment model and applies it to cleaned data from Step 1
"""

import json
import pickle
import pandas as pd
from collections import Counter
import os


class SentimentClassifier:
    """Apply trained sentiment model to classify text data"""
    
    def __init__(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load trained model and vectorizer"""
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
    def load_model(self):
        """Load the trained model and vectorizer from disk"""
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Model files not found")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"Loaded: {self.model_path}, {self.vectorizer_path}")
    
    def classify_text(self, text):
        """Classify a single text and return sentiment with confidence"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Vectorize the text
        text_vec = self.vectorizer.transform([text])
        
        # Predict sentiment
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def classify_dataset(self, data):
        """Classify all items in the dataset"""
        print(f"Classifying {len(data)} items...")
        results = []
        
        for i, item in enumerate(data):
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(data)}...")
            
            cleaned_text = item.get('cleaned_text', '') or item.get('title', '') + ' ' + item.get('selftext', '')
            sentiment, confidence = self.classify_text(cleaned_text)
            
            result = item.copy()
            result['sentiment'] = sentiment
            result['sentiment_confidence'] = float(confidence)
            results.append(result)
        
        return results
    
    def generate_sentiment_report(self, classified_data, output_file='sentiment_report.txt'):
        """Generate a detailed sentiment analysis report"""
        print("\n" + "=" * 60)
        print("SENTIMENT ANALYSIS REPORT")
        print("=" * 60)
        
        # Count sentiments
        sentiments = [item['sentiment'] for item in classified_data]
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        # Calculate percentages
        sentiment_percentages = {
            sentiment: (count / total) * 100 
            for sentiment, count in sentiment_counts.items()
        }
        
        # Calculate average confidence
        avg_confidence = sum(item['sentiment_confidence'] for item in classified_data) / total
        
        # Prepare report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SENTIMENT ANALYSIS REPORT")
        report_lines.append("Smart Farm Technology - Social Media Opinions")
        report_lines.append("=" * 60)
        report_lines.append(f"\nTotal Posts Analyzed: {total}")
        report_lines.append(f"Average Confidence: {avg_confidence:.4f}")
        report_lines.append("\n" + "-" * 60)
        report_lines.append("SENTIMENT DISTRIBUTION")
        report_lines.append("-" * 60)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = sentiment_percentages.get(sentiment, 0)
            report_lines.append(f"\n{sentiment.upper():10s}: {count:4d} posts ({percentage:5.2f}%)")
            
            # Add visual bar
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            report_lines.append(f"             {bar}")
        
        # Find high confidence examples for each sentiment
        report_lines.append("\n" + "-" * 60)
        report_lines.append("EXAMPLE POSTS (High Confidence)")
        report_lines.append("-" * 60)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_items = [item for item in classified_data if item['sentiment'] == sentiment]
            if sentiment_items:
                # Sort by confidence and get top example
                top_item = max(sentiment_items, key=lambda x: x['sentiment_confidence'])
                
                report_lines.append(f"\n{sentiment.upper()} (confidence: {top_item['sentiment_confidence']:.4f}):")
                title = top_item.get('title', 'No title')
                report_lines.append(f"  Title: {title[:100]}...")
                text_preview = top_item.get('cleaned_text', '')[:150]
                report_lines.append(f"  Text: {text_preview}...")
        
        # Sentiment by subreddit
        report_lines.append("\n" + "-" * 60)
        report_lines.append("SENTIMENT BY SUBREDDIT")
        report_lines.append("-" * 60)
        
        subreddit_sentiments = {}
        for item in classified_data:
            subreddit = item.get('subreddit', 'unknown')
            if subreddit not in subreddit_sentiments:
                subreddit_sentiments[subreddit] = []
            subreddit_sentiments[subreddit].append(item['sentiment'])
        
        for subreddit, sentiments in sorted(subreddit_sentiments.items()):
            if subreddit == 'unknown':
                continue
            sent_counts = Counter(sentiments)
            total_sub = len(sentiments)
            report_lines.append(f"\nr/{subreddit} ({total_sub} posts):")
            for sent in ['positive', 'negative', 'neutral']:
                count = sent_counts.get(sent, 0)
                pct = (count / total_sub) * 100 if total_sub > 0 else 0
                report_lines.append(f"  {sent:8s}: {count:3d} ({pct:5.2f}%)")
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 60)
        
        # Print to console
        report_text = '\n'.join(report_lines)
        print(report_text)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {output_file}")
        
        return sentiment_counts, sentiment_percentages


def main():
    """Main execution function"""
    print("=" * 60)
    print("SENTIMENT CLASSIFICATION")
    print("=" * 60)
    
    classifier = SentimentClassifier()
    
    try:
        classifier.load_model()
        
        # Try new preprocessed file first, fallback to old cleaned_data
        if os.path.exists('preprocessed_data.json'):
            input_file = 'preprocessed_data.json'
            print(f"\\nLoading {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Extract posts array if it's wrapped in metadata
            if isinstance(data, dict) and 'data' in data:
                cleaned_data = data['data']
            else:
                cleaned_data = data
        else:
            input_file = 'cleaned_data.json'
            if not os.path.exists(input_file):
                print(f"\\nERROR: {input_file} not found. Run preprocessing first.")
                return
            print(f"\\nLoading {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
        
        print(f"Loaded {len(cleaned_data)} posts")
        
        classified_data = classifier.classify_dataset(cleaned_data)
        
        output_file = 'classified_sentiment_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(classified_data, f, indent=2, ensure_ascii=False)
        print(f"\\nSaved: {output_file}")
        
        classifier.generate_sentiment_report(classified_data)
        
        print("\\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\\nERROR: {e}")


if __name__ == "__main__":
    main()
