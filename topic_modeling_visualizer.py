#!/usr/bin/env python3
"""
Create clear, presentation-ready visualizations for LDA topic modeling results.
Reads lda_coherence_results.json and saves charts to visualizations/topic_modeling/
"""

import json
import os
import math
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_results(path='lda_coherence_results.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_coherence_sensitivity(results, out_dir):
    sens = results['sensitivity_analysis']['results']
    ks = [r['k'] for r in sens]

    # Handle both schemas: cv_coherence or cv_coherence_approx
    cvs = []
    for r in sens:
        if 'cv_coherence' in r:
            cvs.append(r['cv_coherence'])
        else:
            cvs.append(r.get('cv_coherence_approx', 0.0))

    perplexity = [r.get('perplexity', 0.0) for r in sens]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Coherence vs k
    ax = axes[0]
    ax.plot(ks, cvs, marker='o', linewidth=2.5, color='#2563eb')
    for k, cv in zip(ks, cvs):
        ax.text(k, cv + 0.005, f"{cv:.3f}", ha='center', fontsize=10)
    ax.axhline(0.5, color='#f59e0b', linestyle='--', linewidth=1.5, label='Acceptable (0.5)')
    ax.axhline(0.6, color='#16a34a', linestyle='--', linewidth=1.5, label='Strong (0.6)')
    ax.set_title('LDA Coherence Sensitivity by Topic Count (k)', fontsize=12, weight='bold')
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('C_v Coherence')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # Perplexity vs k
    ax2 = axes[1]
    ax2.plot(ks, perplexity, marker='o', linewidth=2.5, color='#7c3aed')
    for k, p in zip(ks, perplexity):
        ax2.text(k, p, f"{p:.3f}", ha='center', va='bottom', fontsize=10)
    ax2.set_title('LDA Perplexity by Topic Count (k)', fontsize=12, weight='bold')
    ax2.set_xlabel('Number of Topics (k)')
    ax2.set_ylabel('Log Perplexity')
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'lda_sensitivity_coherence_perplexity.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_optimal_topic_word_bars(results, out_dir):
    optimal = results['optimal_model']
    topics = optimal['topics']
    k = optimal['k']

    cols = 2
    rows = math.ceil(len(topics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    palette = ['#1d4ed8', '#0ea5e9', '#16a34a', '#ca8a04', '#dc2626', '#7c3aed', '#0891b2', '#65a30d']

    for idx, topic in enumerate(topics):
        ax = axes[idx]
        words = topic['top_words'][:10]
        probs = topic['top_word_probs'][:10]

        # Reverse for top-at-top horizontal bars
        words = words[::-1]
        probs = probs[::-1]

        ax.barh(words, probs, color=palette[idx % len(palette)], alpha=0.9)
        ax.set_title(f"Topic {topic['topic_id'] + 1} Top Terms", fontsize=11, weight='bold')
        ax.set_xlabel('Probability')
        ax.grid(axis='x', alpha=0.2)

    # Hide unused axes
    for j in range(len(topics), len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'LDA Optimal Model (k={k}) - Topic-Term Distributions', fontsize=14, weight='bold', y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'lda_optimal_topics_top_terms.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_topic_summary_card(results, out_dir):
    # compact one-slide summary visual
    optimal = results['optimal_model']
    k = optimal['k']
    cv = optimal.get('cv_coherence', optimal.get('cv_coherence_approx', 0.0))
    prep = results['preprocessing']

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    fig.text(0.05, 0.92, 'LDA Topic Modeling Summary', fontsize=22, weight='bold', color='#111827')

    lines = [
        f"• Optimal topics (k): {k}",
        f"• C_v coherence: {cv:.3f}",
        f"• Corpus size used: {prep.get('corpus_size'):,}",
        f"• Vocabulary size: {prep.get('vocabulary_size'):,}",
        f"• Passes / Iterations: {results['hyperparameters'].get('passes')} / {results['hyperparameters'].get('iterations')}",
        f"• Random seed: {results['hyperparameters'].get('random_seed')}",
    ]

    y = 0.80
    for line in lines:
        fig.text(0.08, y, line, fontsize=16, color='#1f2937')
        y -= 0.10

    # benchmark badges
    badge_color = '#16a34a' if cv >= 0.6 else ('#f59e0b' if cv >= 0.5 else '#dc2626')
    status = 'Strong' if cv >= 0.6 else ('Acceptable' if cv >= 0.5 else 'Needs justification')
    fig.text(
        0.08, 0.18,
        f"Benchmark status: {status}",
        fontsize=15,
        color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=badge_color, edgecolor='none')
    )

    out_path = os.path.join(out_dir, 'lda_summary_card.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print('=' * 70)
    print('TOPIC MODELING VISUALIZATION GENERATOR')
    print('=' * 70)

    results = load_results('lda_coherence_results.json')
    out_dir = os.path.join('visualizations', 'topic_modeling')
    ensure_dir(out_dir)

    plot_coherence_sensitivity(results, out_dir)
    plot_optimal_topic_word_bars(results, out_dir)
    plot_topic_summary_card(results, out_dir)

    print('\nDone. Use these in slides:')
    print(f"- {os.path.join(out_dir, 'lda_sensitivity_coherence_perplexity.png')}")
    print(f"- {os.path.join(out_dir, 'lda_optimal_topics_top_terms.png')}")
    print(f"- {os.path.join(out_dir, 'lda_summary_card.png')}")


if __name__ == '__main__':
    main()
