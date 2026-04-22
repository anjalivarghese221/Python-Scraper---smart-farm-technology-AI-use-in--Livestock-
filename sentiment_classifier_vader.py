#!/usr/bin/env python3
"""
VADER Sentiment Classifier
Lexicon-based sentiment classification with VADER (positive/negative/neutral).
"""

import json
import os
from collections import Counter

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderSentimentClassifier:
    """Apply VADER sentiment model to classify text data."""

    def __init__(self, pos_threshold=0.05, neg_threshold=-0.05):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def _get_text(item):
        text = item.get('cleaned_text', '') or item.get('clean_text', '')
        if not text:
            text = f"{item.get('title', '')} {item.get('selftext', '')} {item.get('text', '')}".strip()
        return text

    def classify_text(self, text):
        scores = self.analyzer.polarity_scores(text or '')
        compound = float(scores.get('compound', 0.0))

        if compound >= self.pos_threshold:
            sentiment = 'positive'
        elif compound <= self.neg_threshold:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        confidence = abs(compound)
        return sentiment, confidence, scores

    def classify_dataset(self, data):
        print(f"Classifying {len(data)} items with VADER...")
        enriched = []

        for i, item in enumerate(data):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(data)}...")

            text = self._get_text(item)
            sentiment, confidence, scores = self.classify_text(text)

            row = item.copy()
            row['sentiment'] = sentiment
            row['sentiment_confidence'] = float(confidence)
            row['sentiment_model'] = 'vader'
            row['sentiment_scores'] = {
                'neg': float(scores.get('neg', 0.0)),
                'neu': float(scores.get('neu', 0.0)),
                'pos': float(scores.get('pos', 0.0)),
                'compound': float(scores.get('compound', 0.0)),
            }
            enriched.append(row)

        return enriched

    def generate_sentiment_report(self, classified_data, output_file='sentiment_report_vader.txt'):
        print('\n' + '=' * 60)
        print('VADER SENTIMENT ANALYSIS REPORT')
        print('=' * 60)

        sentiments = [item.get('sentiment', 'unknown') for item in classified_data]
        sentiment_counts = Counter(sentiments)
        total = len(sentiments) if sentiments else 1

        sentiment_percentages = {
            sentiment: (count / total) * 100
            for sentiment, count in sentiment_counts.items()
        }

        avg_confidence = sum(item.get('sentiment_confidence', 0) for item in classified_data) / total

        lines = []
        lines.append('=' * 60)
        lines.append('VADER SENTIMENT ANALYSIS REPORT')
        lines.append('Smart Farm Technology - Social Media Opinions')
        lines.append('=' * 60)
        lines.append(f'\nModel: vaderSentiment')
        lines.append(f'Total Posts Analyzed: {len(classified_data)}')
        lines.append(f'Average Confidence: {avg_confidence:.4f}')
        lines.append('\n' + '-' * 60)
        lines.append('SENTIMENT DISTRIBUTION')
        lines.append('-' * 60)

        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = sentiment_percentages.get(sentiment, 0)
            bar = '█' * int(percentage / 2)
            lines.append(f'\n{sentiment.upper():10s}: {count:4d} posts ({percentage:5.2f}%)')
            lines.append(f'             {bar}')

        lines.append('\n' + '=' * 60)
        lines.append('END OF REPORT')
        lines.append('=' * 60)

        report_text = '\n'.join(lines)
        print(report_text)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f'\nReport saved to: {output_file}')
        return sentiment_counts, sentiment_percentages


def load_input_data():
    if os.path.exists('preprocessed_data.json'):
        input_file = 'preprocessed_data.json'
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'data' in data:
            return data['data'], input_file
        return data, input_file

    input_file = 'cleaned_data.json'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'{input_file} not found. Run preprocessing first.')

    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f), input_file


def main():
    print('=' * 60)
    print('VADER SENTIMENT CLASSIFICATION')
    print('=' * 60)

    try:
        data, input_file = load_input_data()
        print(f'Loaded {len(data)} rows from {input_file}')

        classifier = VaderSentimentClassifier()
        classified_data = classifier.classify_dataset(data)

        output_file = 'classified_sentiment_data_vader.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(classified_data, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {output_file}')

        classifier.generate_sentiment_report(classified_data)

        print('\n' + '=' * 60)
        print('COMPLETE')
        print('=' * 60)

    except Exception as e:
        print(f'\nERROR: {e}')


if __name__ == '__main__':
    main()
