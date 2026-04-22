#!/usr/bin/env python3
"""
Regional hypothesis testing using subreddit-name geographic proxies.

Note: Dataset has no explicit user geolocation. Regions are inferred from subreddit names,
so this is an exploratory robustness check, not a definitive geographic analysis.

Outputs:
- regional_hypothesis_results.json
- visualizations/regional_proxy_boxplot.png
- visualizations/regional_proxy_counts.png
"""

import json
from datetime import datetime
from math import sqrt
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.05

REGION_KEYWORDS = {
    'North America': [
        'canada', 'toronto', 'vancouver', 'usa', 'texas', 'iowa', 'maine',
        'ohio', 'pennsylvania', 'northcarolina', 'louisiana', 'michigan',
        'virginia', 'stlouis', 'utah'
    ],
    'Europe': [
        'uk', 'ireland', 'france', 'europe', 'germany', 'spain', 'italy', 'netherlands'
    ],
    'Asia': [
        'india', 'pakistan', 'china', 'japan', 'korea', 'indonesia', 'thailand', 'vietnam'
    ],
    'Africa': [
        'africa', 'kenya', 'zambia', 'nigeria', 'ghana', 'southafrica'
    ],
    'Oceania': [
        'australia', 'newzealand', 'nz'
    ]
}


def load_dataset():
    candidates = [
        'classified_sentiment_data_clean_expanded.json',
        'classified_sentiment_data_clean.json',
        'classified_sentiment_data.json',
    ]
    for p in candidates:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f), p
        except FileNotFoundError:
            continue
    raise FileNotFoundError('No input dataset found.')


def sentiment_to_score(label):
    l = str(label or '').lower()
    if l == 'positive':
        return 1.0
    if l == 'negative':
        return -1.0
    return 0.0


def infer_region(subreddit):
    s = str(subreddit or '').lower()
    for region, kws in REGION_KEYWORDS.items():
        for kw in kws:
            if kw in s:
                return region
    return None


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
    return float((np.mean(y) - np.mean(x)) / sqrt(pooled))


def welch_ci(x, y, alpha=0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)
    diff = my - mx
    se = np.sqrt(vx / nx + vy / ny)
    if se == 0:
        return float(diff), float(diff), float(diff)
    num = (vx / nx + vy / ny) ** 2
    den = ((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1)
    df = num / den if den > 0 else nx + ny - 2
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(diff), float(diff - tcrit * se), float(diff + tcrit * se)


def shapiro_checked(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) > 5000:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, 5000, replace=False)
    W, p = stats.shapiro(arr)
    return float(W), float(p)


def main():
    data, dataset_path = load_dataset()

    region_scores = defaultdict(list)
    for row in data:
        region = infer_region(row.get('subreddit', ''))
        if not region:
            continue
        region_scores[region].append(sentiment_to_score(row.get('sentiment')))

    counts = {k: len(v) for k, v in region_scores.items()}
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    if len(ranked) < 2:
        raise ValueError('Not enough region-proxy data to run a two-group test.')

    # Choose top-2 regions by sample size for robust comparison
    g1_name, _ = ranked[0]
    g2_name, _ = ranked[1]
    g1 = np.asarray(region_scores[g1_name], dtype=float)
    g2 = np.asarray(region_scores[g2_name], dtype=float)

    W1, p1 = shapiro_checked(g1)
    W2, p2 = shapiro_checked(g2)
    both_normal = (p1 > ALPHA) and (p2 > ALPHA)

    if both_normal:
        test_used = 'Welch independent t-test'
        stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        stat_label = 't'
    else:
        test_used = 'Mann-Whitney U (two-sided)'
        stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        stat_label = 'U'

    d = cohen_d(g1, g2)
    diff, ci_lo, ci_hi = welch_ci(g1, g2, alpha=ALPHA)  # group2-group1

    result = {
        'metadata': {
            'dataset': dataset_path,
            'analysis_date': datetime.now().isoformat(),
            'alpha': ALPHA,
            'two_tailed': True,
            'method_note': 'Geographic regions inferred from subreddit-name proxies only.'
        },
        'region_proxy_counts': counts,
        'tested_groups': {
            'group1': g1_name,
            'group2': g2_name,
            'n_group1': int(len(g1)),
            'n_group2': int(len(g2)),
            'mean_group1': float(np.mean(g1)),
            'mean_group2': float(np.mean(g2))
        },
        'normality_shapiro': {
            g1_name: {'W': W1, 'p': p1},
            g2_name: {'W': W2, 'p': p2},
            'both_normal': bool(both_normal)
        },
        'inferential_test': {
            'test_used': test_used,
            'test_statistic_label': stat_label,
            'test_statistic': float(stat),
            'p_value': float(p_val),
            'cohens_d_group2_minus_group1': float(d),
            'ci_95_mean_difference_group2_minus_group1': [float(ci_lo), float(ci_hi)],
            'decision_alpha_0_05': 'Reject H0' if p_val < ALPHA else 'Fail to reject H0'
        }
    }

    with open('regional_hypothesis_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    # Visual 1: counts by region proxy
    plt.figure(figsize=(9, 5))
    names = [k for k, _ in ranked]
    vals = [v for _, v in ranked]
    plt.bar(names, vals, color='#64748b')
    plt.title('Region-Proxy Sample Sizes (from subreddit names)', weight='bold')
    plt.ylabel('Posts')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig('visualizations/regional_proxy_counts.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visual 2: boxplot for tested pair
    plt.figure(figsize=(8, 6))
    plt.boxplot([g1, g2], labels=[g1_name, g2_name], patch_artist=True,
                boxprops=dict(facecolor='#93c5fd'), medianprops=dict(color='#1d4ed8', linewidth=2))
    plt.title(f'Sentiment by Region Proxy: {g1_name} vs {g2_name}', weight='bold')
    plt.ylabel('Sentiment score (-1,0,+1)')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig('visualizations/regional_proxy_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('=' * 80)
    print('REGIONAL PROXY HYPOTHESIS TEST COMPLETE')
    print('=' * 80)
    print(f"Top regions tested: {g1_name} (n={len(g1)}) vs {g2_name} (n={len(g2)})")
    print(f"Test used: {test_used}")
    print(f"{stat_label}={float(stat):.4f}, p={float(p_val):.4g}")
    print(f"Cohen's d: {float(d):.4f}")
    print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print('Saved: regional_hypothesis_results.json')
    print('Saved visuals: visualizations/regional_proxy_counts.png, visualizations/regional_proxy_boxplot.png')


if __name__ == '__main__':
    main()
