#!/usr/bin/env python3
"""
RoBERTa Sentiment Classifier
Applies a transformer model to classify sentiment as positive/negative/neutral.
"""

import json
import os
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RobertaSentimentClassifier:
    """Apply RoBERTa sentiment model to classify text data."""

    def __init__(
        self,
        model_name='cardiffnlp/twitter-roberta-base-sentiment-latest',
        batch_size=32,
        max_length=256,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load tokenizer and model."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    @staticmethod
    def normalize_label(label):
        """Map model labels to positive/negative/neutral."""
        label_norm = (label or '').strip().lower()

        mapping = {
            'label_0': 'negative',
            'label_1': 'neutral',
            'label_2': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'positive': 'positive',
        }
        return mapping.get(label_norm, label_norm)

    def classify_batch(self, texts):
        """Classify a batch of texts, return list of (label, confidence)."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

        id2label = self.model.config.id2label
        out = []
        for pred, conf in zip(preds.tolist(), confs.tolist()):
            raw_label = id2label.get(pred, str(pred))
            out.append((self.normalize_label(raw_label), float(conf)))
        return out

    def classify_dataset(self, data):
        """Classify dataset and return enriched records."""
        print(f"Classifying {len(data)} items with RoBERTa...")

        enriched = []
        texts = []

        for item in data:
            text = item.get('cleaned_text', '') or item.get('clean_text', '')
            if not text:
                text = f"{item.get('title', '')} {item.get('selftext', '')} {item.get('text', '')}".strip()
            texts.append(text)

        for i in tqdm(range(0, len(texts), self.batch_size), desc='RoBERTa inference'):
            batch_texts = texts[i:i + self.batch_size]
            batch_preds = self.classify_batch(batch_texts)

            for j, (sentiment, confidence) in enumerate(batch_preds):
                item = data[i + j].copy()
                item['sentiment'] = sentiment
                item['sentiment_confidence'] = confidence
                item['sentiment_model'] = self.model_name
                enriched.append(item)

        return enriched

    def generate_sentiment_report(self, classified_data, output_file='sentiment_report_roberta.txt'):
        """Generate report."""
        print('\n' + '=' * 60)
        print('ROBERTA SENTIMENT ANALYSIS REPORT')
        print('=' * 60)

        sentiments = [item.get('sentiment', 'unknown') for item in classified_data]
        sentiment_counts = Counter(sentiments)
        total = len(sentiments) if sentiments else 1

        sentiment_percentages = {
            sentiment: (count / total) * 100
            for sentiment, count in sentiment_counts.items()
        }

        avg_confidence = sum(item.get('sentiment_confidence', 0) for item in classified_data) / total

        report_lines = []
        report_lines.append('=' * 60)
        report_lines.append('ROBERTA SENTIMENT ANALYSIS REPORT')
        report_lines.append('Smart Farm Technology - Social Media Opinions')
        report_lines.append('=' * 60)
        report_lines.append(f'\nModel: {self.model_name}')
        report_lines.append(f'Total Posts Analyzed: {len(classified_data)}')
        report_lines.append(f'Average Confidence: {avg_confidence:.4f}')
        report_lines.append('\n' + '-' * 60)
        report_lines.append('SENTIMENT DISTRIBUTION')
        report_lines.append('-' * 60)

        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = sentiment_percentages.get(sentiment, 0)
            bar = '█' * int(percentage / 2)
            report_lines.append(f'\n{sentiment.upper():10s}: {count:4d} posts ({percentage:5.2f}%)')
            report_lines.append(f'             {bar}')

        report_lines.append('\n' + '=' * 60)
        report_lines.append('END OF REPORT')
        report_lines.append('=' * 60)

        report_text = '\n'.join(report_lines)
        print(report_text)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f'\nReport saved to: {output_file}')
        return sentiment_counts, sentiment_percentages


def load_input_data():
    """Load the preprocessed dataset using existing pipeline conventions."""
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
    print('ROBERTA SENTIMENT CLASSIFICATION')
    print('=' * 60)

    try:
        data, input_file = load_input_data()
        print(f'Loaded {len(data)} rows from {input_file}')

        classifier = RobertaSentimentClassifier()
        classifier.load_model()

        classified_data = classifier.classify_dataset(data)

        output_file = 'classified_sentiment_data_roberta.json'
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
