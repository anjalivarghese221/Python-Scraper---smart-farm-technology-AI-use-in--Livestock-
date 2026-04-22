#!/usr/bin/env python3
"""
Phase 1 quality checks for sentiment labeling.

Tasks implemented:
1) Extract important keywords from positive/negative/neutral labeled posts
2) Generate a manual-review sample of at least 20 posts
"""

import json
import re
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer


INPUT_FILE = 'classified_sentiment_data.json'
KEYWORD_OUTPUT_FILE = 'phase1_keyword_check_by_sentiment.json'
MANUAL_SAMPLE_FILE = 'phase1_manual_review_sample_20.json'
def is_domain_relevant(post):
    text = (
        f"{post.get('title', '')} "
        f"{post.get('clean_text', '')} "
        f"{post.get('cleaned_text', '')} "
        f"{post.get('raw_text', '')} "
        f"{post.get('text', '')}"
    ).lower()

    livestock_terms = ['dairy', 'livestock', 'cattle', 'beef', 'ruminant', 'goat', 'sheep', 'swine', 'poultry']
    tech_terms = [
        'ai',
        'artificial intelligence',
        'smart farming',
        'precision livestock',
        'automation',
        'robot',
        'sensor',
        'iot',
        'machine learning',
        'computer vision',
        'algorithm',
        'predictive',
    ]

    livestock_hits = sum(1 for term in livestock_terms if re.search(rf"\b{re.escape(term)}\b", text))
    tech_hits = sum(1 for term in tech_terms if re.search(rf"\b{re.escape(term)}\b", text))

    score = livestock_hits + tech_hits
    subreddit = (post.get('subreddit', '') or '').lower()
    if subreddit in {'agtech', 'farming', 'agriculture', 'livestock', 'dairyfarming', 'precisionagriculture'}:
        score += 1

    return score >= 2


def relevance_score(post):
    text = (
        f"{post.get('title', '')} "
        f"{post.get('clean_text', '')} "
        f"{post.get('cleaned_text', '')} "
        f"{post.get('raw_text', '')} "
        f"{post.get('text', '')}"
    ).lower()

    livestock_terms = ['dairy', 'livestock', 'cattle', 'beef', 'ruminant', 'goat', 'sheep', 'swine', 'poultry']
    tech_terms = [
        'ai',
        'artificial intelligence',
        'smart farming',
        'precision livestock',
        'automation',
        'robot',
        'sensor',
        'iot',
        'machine learning',
        'computer vision',
        'algorithm',
        'predictive',
    ]

    livestock_count = sum(1 for t in livestock_terms if re.search(rf"\b{re.escape(t)}\b", text))
    tech_count = sum(1 for t in tech_terms if re.search(rf"\b{re.escape(t)}\b", text))
    subreddit = (post.get('subreddit', '') or '').lower()
    subreddit_bonus = 1 if subreddit in {'agtech', 'farming', 'agriculture', 'livestock', 'dairyfarming', 'precisionagriculture'} else 0
    return livestock_count + tech_count + subreddit_bonus


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    return data


def normalize_text(post):
    return (
        post.get('clean_text')
        or post.get('cleaned_text')
        or post.get('raw_text')
        or f"{post.get('title', '')} {post.get('text', '')}".strip()
    )


def top_keywords_for_group(texts, top_n=20):
    if not texts:
        return []

    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )

    matrix = vectorizer.fit_transform(texts)
    term_sums = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    pairs = sorted(zip(terms, term_sums), key=lambda x: x[1], reverse=True)
    return [{'term': term, 'count': int(count)} for term, count in pairs[:top_n]]


def stratified_manual_sample(data, total_n=20):
    by_sentiment = {'positive': [], 'negative': [], 'neutral': []}
    for item in data:
        sentiment = item.get('sentiment', '').lower()
        if sentiment in by_sentiment:
            by_sentiment[sentiment].append(item)

    allocation = {'positive': 7, 'negative': 7, 'neutral': 6}
    sample = []

    for sentiment, n in allocation.items():
        pool = sorted(
            by_sentiment[sentiment],
            key=lambda x: (relevance_score(x), float(x.get('sentiment_confidence', 0.0))),
            reverse=True,
        )
        if len(pool) <= n:
            sample.extend(pool)
        else:
            sample.extend(pool[:n])

    # If any class was too small, fill from remaining posts
    if len(sample) < total_n:
        used_ids = {id(x) for x in sample}
        remaining = sorted(
            [x for x in data if id(x) not in used_ids],
            key=lambda x: (relevance_score(x), float(x.get('sentiment_confidence', 0.0))),
            reverse=True,
        )
        needed = min(total_n - len(sample), len(remaining))
        if needed > 0:
            sample.extend(remaining[:needed])

    trimmed = []
    for idx, item in enumerate(sample[:total_n], 1):
        trimmed.append({
            'review_id': idx,
            'sentiment_label': item.get('sentiment', 'unknown'),
            'sentiment_confidence': float(item.get('sentiment_confidence', 0.0)),
            'domain_relevance_score': relevance_score(item),
            'subreddit': item.get('subreddit', ''),
            'created_date': item.get('created_date', ''),
            'title': item.get('title', ''),
            'text_preview': normalize_text(item)[:500],
            'url': item.get('url', ''),
            'manual_relevance_check': '',
            'manual_label_check': '',
            'manual_notes': '',
        })

    return trimmed


def main():
    data = load_data(INPUT_FILE)

    # Keep Reddit-only rows for this phase if source is present
    reddit_data = [d for d in data if d.get('source', 'reddit') == 'reddit']
    if reddit_data:
        data = reddit_data

    domain_filtered = [d for d in data if is_domain_relevant(d)]
    if domain_filtered:
        data = domain_filtered

    grouped_texts = {'positive': [], 'negative': [], 'neutral': []}
    for item in data:
        sentiment = item.get('sentiment', '').lower()
        if sentiment in grouped_texts:
            text = normalize_text(item)
            if text:
                grouped_texts[sentiment].append(text)

    keyword_report = {
        'metadata': {
            'input_file': INPUT_FILE,
            'total_posts_used': len(data),
            'sentiment_counts': dict(Counter([x.get('sentiment', 'unknown') for x in data])),
            'method': 'CountVectorizer unigram+bigram (stopwords removed, min_df=2)',
        },
        'keywords_by_sentiment': {
            'positive': top_keywords_for_group(grouped_texts['positive']),
            'negative': top_keywords_for_group(grouped_texts['negative']),
            'neutral': top_keywords_for_group(grouped_texts['neutral']),
        },
    }

    manual_sample = stratified_manual_sample(data, total_n=20)

    with open(KEYWORD_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(keyword_report, f, indent=2, ensure_ascii=False)

    with open(MANUAL_SAMPLE_FILE, 'w', encoding='utf-8') as f:
        json.dump(manual_sample, f, indent=2, ensure_ascii=False)

    print('Phase 1 quality check complete')
    print(f'  Saved keyword report: {KEYWORD_OUTPUT_FILE}')
    print(f'  Saved manual sample (n={len(manual_sample)}): {MANUAL_SAMPLE_FILE}')


if __name__ == '__main__':
    main()
