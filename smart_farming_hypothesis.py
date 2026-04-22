#!/usr/bin/env python3
"""
Topic-specific hypothesis testing for Smart Farming / AI in Livestock.

PRIMARY HYPOTHESIS (domain-specific):
H0: Mean sentiment is equal between livestock-focused discourse and non-livestock discourse.
H1: Mean sentiment differs between livestock-focused and non-livestock discourse.

SECONDARY HYPOTHESIS (temporal event-window):
H0: Mean sentiment is equal before vs during the high-volume surge window.
H1: Mean sentiment differs before vs during the surge window.

Decision rule (both tests):
1) Shapiro-Wilk normality per group
2) If both normal -> Welch independent t-test
   Else -> Mann-Whitney U (two-sided)
3) Always report statistic, p-value, Cohen's d, 95% CI.

Outputs:
- smart_farming_hypothesis_results.json
- visualizations/hypothesis_livestock_vs_nonlivestock_boxplot.png
- visualizations/hypothesis_livestock_vs_nonlivestock_mean_ci.png
- visualizations/hypothesis_pre_vs_surge_boxplot.png
"""

import json
from datetime import datetime
from math import sqrt
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.05
SURGE_START = datetime(2025, 7, 1)

LIVESTOCK_KEYWORDS = {
    'livestock', 'dairy', 'cattle', 'cow', 'cows', 'beef', 'herd',
    'milking', 'barn', 'ruminant', 'poultry', 'swine', 'goat', 'sheep'
}

LIVESTOCK_SUBREDDIT_HINTS = {
    'livestock', 'dairy', 'dairyfarming', 'cattle', 'ranching', 'meridairy', 'wheresthebeef'
}


def load_dataset() -> Tuple[List[Dict], str]:
    candidates = [
        'classified_sentiment_data_clean_expanded.json',
        'classified_sentiment_data_clean.json',
        'classified_sentiment_data.json',
    ]
    for path in candidates:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f), path
        except FileNotFoundError:
            continue
    raise FileNotFoundError('No dataset found in expected candidates.')


def parse_date(s: str):
    if not s:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def sentiment_to_score(label: str) -> float:
    label = str(label or '').lower()
    if label == 'positive':
        return 1.0
    if label == 'negative':
        return -1.0
    return 0.0


def is_livestock_focused(item: Dict) -> bool:
    sub = str(item.get('subreddit', '')).lower()
    if sub in LIVESTOCK_SUBREDDIT_HINTS:
        return True

    text = ' '.join([
        str(item.get('title', '')),
        str(item.get('text', '')),
        str(item.get('clean_text', '')),
        str(item.get('raw_text', '')),
    ]).lower()

    return any(kw in text for kw in LIVESTOCK_KEYWORDS)


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(y) - np.mean(x)) / sqrt(pooled))  # group2 - group1


def mean_diff_ci_welch(group1, group2, alpha=0.05):
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    m1, m2 = np.mean(g1), np.mean(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    n1, n2 = len(g1), len(g2)

    diff = m2 - m1
    se = sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return float(diff), float(diff), float(diff)

    num = (v1 / n1 + v2 / n2) ** 2
    den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else n1 + n2 - 2

    tcrit = stats.t.ppf(1 - alpha / 2, df)
    lo = diff - tcrit * se
    hi = diff + tcrit * se
    return float(diff), float(lo), float(hi)


def shapiro_checked(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) > 5000:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=5000, replace=False)
    W, p = stats.shapiro(arr)
    return float(W), float(p), int(len(arr))


def run_test(group1, group2, label1, label2):
    W1, p1, n1 = shapiro_checked(group1)
    W2, p2, n2 = shapiro_checked(group2)
    both_normal = (p1 > ALPHA) and (p2 > ALPHA)

    if both_normal:
        test = 'Welch independent t-test'
        stat, p = stats.ttest_ind(group1, group2, equal_var=False)
        stat_label = 't'
    else:
        test = 'Mann-Whitney U (two-sided)'
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        stat_label = 'U'

    d = cohen_d(group1, group2)
    diff, lo, hi = mean_diff_ci_welch(group1, group2, alpha=ALPHA)  # group2-group1

    out = {
        'groups': {'group1': label1, 'group2': label2},
        'n': {'group1': int(len(group1)), 'group2': int(len(group2))},
        'means': {'group1': float(np.mean(group1)), 'group2': float(np.mean(group2))},
        'normality_shapiro': {
            label1: {'W': W1, 'p': p1, 'n_used': n1},
            label2: {'W': W2, 'p': p2, 'n_used': n2},
            'both_normal': bool(both_normal)
        },
        'inferential_test': {
            'test_used': test,
            'test_statistic_label': stat_label,
            'test_statistic': float(stat),
            'p_value': float(p),
            'cohens_d_group2_minus_group1': float(d),
            'ci_95_mean_difference_group2_minus_group1': [float(lo), float(hi)],
            'decision_alpha_0_05': 'Reject H0' if p < ALPHA else 'Fail to reject H0'
        }
    }
    return out


def plot_box(groups, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(groups, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='#93c5fd'), medianprops=dict(color='#1d4ed8', linewidth=2))
    ax.set_title(title, weight='bold')
    ax.set_ylabel('Sentiment score (-1, 0, +1)')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_ci(group1, group2, label1, label2, out_path):
    m1 = float(np.mean(group1))
    m2 = float(np.mean(group2))
    _, lo, hi = mean_diff_ci_welch(group1, group2, alpha=ALPHA)  # group2-group1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([label1, label2], [m1, m2], color=['#64748b', '#2563eb'])
    ax.axhline(m1, color='#334155', linestyle='--', linewidth=1.5, label=f'{label1} mean')
    ax.text(0.02, 0.94, f'95% CI ({label2}-{label1}): [{lo:.3f}, {hi:.3f}]', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8))
    ax.set_title(f'Mean Sentiment: {label1} vs {label2}', weight='bold')
    ax.set_ylabel('Mean sentiment score')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    data, dataset_path = load_dataset()

    # Primary grouping: livestock-focused vs non-livestock
    livestock_scores = []
    non_livestock_scores = []

    # Secondary grouping: pre-surge vs surge
    pre_surge_scores = []
    surge_scores = []

    for item in data:
        score = sentiment_to_score(item.get('sentiment'))

        if is_livestock_focused(item):
            livestock_scores.append(score)
        else:
            non_livestock_scores.append(score)

        dt = parse_date(item.get('created_date', ''))
        if dt is not None:
            if dt < SURGE_START:
                pre_surge_scores.append(score)
            else:
                surge_scores.append(score)

    primary = run_test(livestock_scores, non_livestock_scores, 'Livestock-focused', 'Non-livestock')
    secondary = run_test(pre_surge_scores, surge_scores, 'Pre-surge (<2025-07)', 'Surge window (>=2025-07)')

    results = {
        'metadata': {
            'dataset': dataset_path,
            'analysis_date': datetime.now().isoformat(),
            'alpha': ALPHA,
            'two_tailed': True,
            'surge_start': SURGE_START.strftime('%Y-%m-%d')
        },
        'hypotheses': {
            'primary': {
                'H0': 'No significant difference in mean sentiment between livestock-focused and non-livestock discourse.',
                'H1': 'Significant difference in mean sentiment between livestock-focused and non-livestock discourse.'
            },
            'secondary': {
                'H0': 'No significant difference in mean sentiment before vs during the surge window.',
                'H1': 'Significant difference in mean sentiment before vs during the surge window.'
            }
        },
        'primary_test_livestock_vs_nonlivestock': primary,
        'secondary_test_pre_surge_vs_surge': secondary
    }

    with open('smart_farming_hypothesis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    plot_box(
        [livestock_scores, non_livestock_scores],
        ['Livestock-focused', 'Non-livestock'],
        'Sentiment: Livestock-focused vs Non-livestock Discourse',
        'visualizations/hypothesis_livestock_vs_nonlivestock_boxplot.png'
    )
    plot_mean_ci(
        livestock_scores, non_livestock_scores,
        'Livestock-focused', 'Non-livestock',
        'visualizations/hypothesis_livestock_vs_nonlivestock_mean_ci.png'
    )
    plot_box(
        [pre_surge_scores, surge_scores],
        ['Pre-surge', 'Surge window'],
        'Sentiment: Pre-surge vs Surge Window',
        'visualizations/hypothesis_pre_vs_surge_boxplot.png'
    )

    print('=' * 80)
    print('SMART FARMING HYPOTHESIS TESTING COMPLETE')
    print('=' * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Primary test used: {primary['inferential_test']['test_used']}")
    print(f"Primary stat: {primary['inferential_test']['test_statistic_label']}={primary['inferential_test']['test_statistic']:.4f}, p={primary['inferential_test']['p_value']:.4g}")
    print(f"Primary Cohen's d: {primary['inferential_test']['cohens_d_group2_minus_group1']:.4f}")
    print(f"Primary 95% CI: {primary['inferential_test']['ci_95_mean_difference_group2_minus_group1']}")
    print('-' * 80)
    print(f"Secondary test used: {secondary['inferential_test']['test_used']}")
    print(f"Secondary stat: {secondary['inferential_test']['test_statistic_label']}={secondary['inferential_test']['test_statistic']:.4f}, p={secondary['inferential_test']['p_value']:.4g}")
    print(f"Secondary Cohen's d: {secondary['inferential_test']['cohens_d_group2_minus_group1']:.4f}")
    print(f"Secondary 95% CI: {secondary['inferential_test']['ci_95_mean_difference_group2_minus_group1']}")
    print('=' * 80)
    print('Saved: smart_farming_hypothesis_results.json')
    print('Saved visuals:')
    print('  visualizations/hypothesis_livestock_vs_nonlivestock_boxplot.png')
    print('  visualizations/hypothesis_livestock_vs_nonlivestock_mean_ci.png')
    print('  visualizations/hypothesis_pre_vs_surge_boxplot.png')


if __name__ == '__main__':
    main()
