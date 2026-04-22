#!/usr/bin/env python3
"""
Visualize top 10 positive and negative semantic sentiment drivers.
Reads topic_stability_coherence_report.json and saves a presentation-ready figure.
"""

import json
import os
import matplotlib.pyplot as plt


def load_drivers(path='topic_stability_coherence_report.json'):
    with open(path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    sem = report.get('semantic_drivers', {})
    neg = sem.get('top_negative_drivers', [])[:10]
    pos = sem.get('top_positive_drivers', [])[:10]
    return neg, pos


def plot_drivers(neg, pos, out_file='visualizations/semantic_drivers_top10.png'):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    neg_words = [x['word'] for x in neg][::-1]
    neg_vals = [x['log_odds'] for x in neg][::-1]

    pos_words = [x['word'] for x in pos][::-1]
    pos_vals = [x['log_odds'] for x in pos][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=False)

    # Negative drivers
    ax1 = axes[0]
    ax1.barh(neg_words, neg_vals, color='#dc2626', alpha=0.9)
    ax1.axvline(0, color='black', linewidth=1)
    ax1.set_title('Top 10 Negative Sentiment Drivers', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Log-Odds (more negative → stronger negative driver)')
    ax1.grid(axis='x', alpha=0.25)

    for i, v in enumerate(neg_vals):
        ax1.text(v - 0.02, i, f'{v:.2f}', va='center', ha='right', fontsize=9, color='#7f1d1d')

    # Positive drivers
    ax2 = axes[1]
    ax2.barh(pos_words, pos_vals, color='#16a34a', alpha=0.9)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_title('Top 10 Positive Sentiment Drivers', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Log-Odds (more positive → stronger positive driver)')
    ax2.grid(axis='x', alpha=0.25)

    for i, v in enumerate(pos_vals):
        ax2.text(v + 0.02, i, f'{v:.2f}', va='center', ha='left', fontsize=9, color='#14532d')

    fig.suptitle('Semantic Drivers of Sentiment (Top 10 + Top 10)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_file}')


if __name__ == '__main__':
    neg, pos = load_drivers()
    if not neg or not pos:
        raise ValueError('Missing semantic drivers in topic_stability_coherence_report.json. Run topic_stability_analysis_final.py first.')
    plot_drivers(neg, pos)
