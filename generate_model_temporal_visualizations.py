#!/usr/bin/env python3
"""
Generate model-specific temporal visualizations for Logistic, RoBERTa, and VADER.
Creates publication-style figures under visualizations/<model>/.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    'logistic': 'classified_sentiment_data.json',
    'roberta': 'classified_sentiment_data_roberta.json',
    'vader': 'classified_sentiment_data_vader.json',
}

SENTIMENT_COLORS = {
    'positive': '#2ecc71',
    'neutral': '#95a5a6',
    'negative': '#e74c3c',
}


def parse_post_datetime(post):
    if isinstance(post.get('created_date'), str):
        try:
            return datetime.strptime(post['created_date'][:10], '%Y-%m-%d')
        except Exception:
            pass

    if isinstance(post.get('created_at'), str):
        val = post['created_at'].replace('Z', '+00:00')
        try:
            return datetime.fromisoformat(val)
        except Exception:
            pass

    if post.get('created_utc') is not None:
        try:
            return datetime.fromtimestamp(float(post['created_utc']))
        except Exception:
            pass

    return None


def monthly_rollup(rows):
    monthly = {}

    for row in rows:
        dt = parse_post_datetime(row)
        if not dt:
            continue

        key = f'{dt.year}-{dt.month:02d}'
        if key not in monthly:
            monthly[key] = {'total': 0, 'positive': 0, 'neutral': 0, 'negative': 0}

        monthly[key]['total'] += 1
        s = str(row.get('sentiment', '')).lower()
        if s in ('positive', 'negative', 'neutral'):
            monthly[key][s] += 1

    months = sorted(monthly.keys())
    totals = [monthly[m]['total'] for m in months]

    pos_pct, neu_pct, neg_pct = [], [], []
    for m in months:
        total = monthly[m]['total'] or 1
        pos_pct.append(100.0 * monthly[m]['positive'] / total)
        neu_pct.append(100.0 * monthly[m]['neutral'] / total)
        neg_pct.append(100.0 * monthly[m]['negative'] / total)

    return months, totals, pos_pct, neu_pct, neg_pct


def make_plot(model_name, input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        rows = json.load(f)

    months, totals, pos_pct, neu_pct, neg_pct = monthly_rollup(rows)
    if not months:
        raise ValueError(f'No valid timestamp rows found in {input_file}')

    x = np.arange(len(months))
    trend = np.poly1d(np.polyfit(x, totals, 1))(x)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2, 1]})

    # A) Monthly post volume (style aligned with requested example)
    ax1 = axes[0]
    ax1.bar(x, totals, width=0.08, color='gray', alpha=0.95, edgecolor='gray')
    ax1.plot(x, trend, 'r--', linewidth=2, label='Trend')
    ax1.set_ylabel('Number of Posts', fontsize=13)
    ax1.set_title('A) Monthly Post Volume Over Time', fontsize=19, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Reduce tick density for readability
    step = max(1, len(months) // 12)
    tick_idx = list(range(0, len(months), step))
    tick_labels = [datetime.strptime(months[i], '%Y-%m').strftime('%b %Y') for i in tick_idx]
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

    # B) Sentiment trend (model-specific)
    ax2 = axes[1]
    ax2.plot(x, pos_pct, marker='o', linewidth=2, color=SENTIMENT_COLORS['positive'], label='Positive')
    ax2.plot(x, neu_pct, marker='s', linewidth=2, color=SENTIMENT_COLORS['neutral'], label='Neutral')
    ax2.plot(x, neg_pct, marker='^', linewidth=2, color=SENTIMENT_COLORS['negative'], label='Negative')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Sentiment %', fontsize=12)
    ax2.set_title('B) Monthly Sentiment Distribution', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

    fig.suptitle(f'Temporal Sentiment Trend Analysis ({model_name.upper()})', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    for model, input_file in MODELS.items():
        if not os.path.exists(input_file):
            print(f'[SKIP] {model}: missing {input_file}')
            continue

        out_dir = os.path.join('visualizations', model)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, 'temporal_sentiment_trend_analysis.png')
        make_plot(model, input_file, out_file)
        print(f'[SAVED] {out_file}')


if __name__ == '__main__':
    main()
