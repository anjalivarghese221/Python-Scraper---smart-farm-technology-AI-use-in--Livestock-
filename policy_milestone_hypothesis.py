#!/usr/bin/env python3
"""
Policy milestone hypothesis testing for sentiment change (Before vs After 2024).

H0: mu_before = mu_after
H1: mu_before != mu_after

Decision rule:
1) Shapiro-Wilk normality check on both groups
2) If both normal -> Welch independent t-test
   Else -> Mann-Whitney U test (two-tailed)

Always reports:
- test statistic
- p-value
- Cohen's d
- 95% CI for mean difference (After - Before)

Outputs:
- policy_milestone_hypothesis_results.json
- visualizations/policy_milestone_boxplot.png
- visualizations/policy_milestone_mean_ci.png
- visualizations/policy_milestone_distribution.png
"""

import json
from datetime import datetime
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


MILESTONE_DATE = datetime(2024, 1, 1)
ALPHA = 0.05


def load_dataset():
    candidates = [
        'classified_sentiment_data_clean_expanded.json',
        'classified_sentiment_data_clean.json',
        'classified_sentiment_data.json',
    ]
    for path in candidates:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, path
        except FileNotFoundError:
            continue
    raise FileNotFoundError('No dataset found in expected candidates.')


def parse_date(s):
    if not s:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def sentiment_to_score(label):
    label = str(label or '').lower()
    if label == 'positive':
        return 1.0
    if label == 'negative':
        return -1.0
    return 0.0


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
    return float((np.mean(y) - np.mean(x)) / sqrt(pooled))  # After - Before


def mean_diff_ci_welch(x, y, alpha=0.05):
    """95% CI for mean difference (After - Before) using Welch SE + Satterthwaite df."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)

    diff = my - mx
    se = sqrt(vx / nx + vy / ny)
    if se == 0:
        return float(diff), float(diff), float(diff)

    num = (vx / nx + vy / ny) ** 2
    den = ((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1)
    df = num / den if den > 0 else nx + ny - 2

    tcrit = stats.t.ppf(1 - alpha / 2, df)
    lo = diff - tcrit * se
    hi = diff + tcrit * se
    return float(diff), float(lo), float(hi)


def shapiro_checked(arr):
    """Shapiro-Wilk with cap at 5000 samples for SciPy stability notes."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) > 5000:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=5000, replace=False)
    stat, p = stats.shapiro(arr)
    return float(stat), float(p), int(len(arr))


def make_visuals(before_scores, after_scores, mean_before, mean_after, ci_lo, ci_hi):
    # 1) Boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([before_scores, after_scores], labels=['Before 2024', 'After 2024'], patch_artist=True,
               boxprops=dict(facecolor='#93c5fd'), medianprops=dict(color='#1d4ed8', linewidth=2))
    ax.set_title('Sentiment Score Distribution: Before vs After 2024 Policy Milestone', weight='bold')
    ax.set_ylabel('Sentiment score (-1, 0, +1)')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig('visualizations/policy_milestone_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Mean + CI
    fig, ax = plt.subplots(figsize=(8, 6))
    means = [mean_before, mean_after]
    ax.bar(['Before 2024', 'After 2024'], means, color=['#64748b', '#2563eb'], alpha=0.9)
    # CI for (after-before), displayed around after as reference difference bar label
    ax.axhline(mean_before, color='#334155', linestyle='--', linewidth=1.5, label='Before mean')
    ax.errorbar(['After 2024'], [mean_after],
                yerr=[[mean_after - (mean_before + ci_lo)], [(mean_before + ci_hi) - mean_after]],
                fmt='none', ecolor='black', capsize=6, linewidth=1.5)
    ax.set_title('Mean Sentiment Before vs After (with 95% CI context)', weight='bold')
    ax.set_ylabel('Mean sentiment score')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('visualizations/policy_milestone_mean_ci.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Distribution bars
    bins = [-1.5, -0.5, 0.5, 1.5]
    before_hist, _ = np.histogram(before_scores, bins=bins)
    after_hist, _ = np.histogram(after_scores, bins=bins)

    labels = ['Negative (-1)', 'Neutral (0)', 'Positive (+1)']
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, before_hist, width=w, label='Before 2024', color='#94a3b8')
    ax.bar(x + w / 2, after_hist, width=w, label='After 2024', color='#2563eb')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Sentiment Class Counts: Before vs After 2024', weight='bold')
    ax.set_ylabel('Post count')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig('visualizations/policy_milestone_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print('=' * 80)
    print('POLICY MILESTONE HYPOTHESIS TESTING (Before vs After 2024)')
    print('=' * 80)

    data, dataset_path = load_dataset()
    print(f'Using dataset: {dataset_path}')
    print(f'Total rows: {len(data)}')

    before_scores = []
    after_scores = []

    for item in data:
        dt = parse_date(item.get('created_date', ''))
        if dt is None:
            continue
        score = sentiment_to_score(item.get('sentiment'))
        if dt < MILESTONE_DATE:
            before_scores.append(score)
        else:
            after_scores.append(score)

    before_scores = np.asarray(before_scores, dtype=float)
    after_scores = np.asarray(after_scores, dtype=float)

    print(f'Before group n: {len(before_scores)}')
    print(f'After group n:  {len(after_scores)}')

    # Normality checks
    sw_b_stat, sw_b_p, sw_b_n = shapiro_checked(before_scores)
    sw_a_stat, sw_a_p, sw_a_n = shapiro_checked(after_scores)

    both_normal = (sw_b_p > ALPHA) and (sw_a_p > ALPHA)

    if both_normal:
        test_name = 'Welch independent t-test'
        t_stat, p_val = stats.ttest_ind(before_scores, after_scores, equal_var=False)
        test_stat_label = 't'
        test_stat_value = float(t_stat)
    else:
        test_name = 'Mann-Whitney U (two-sided)'
        u_stat, p_val = stats.mannwhitneyu(before_scores, after_scores, alternative='two-sided')
        test_stat_label = 'U'
        test_stat_value = float(u_stat)

    mean_before = float(np.mean(before_scores))
    mean_after = float(np.mean(after_scores))
    d_val = cohen_d(before_scores, after_scores)
    diff, ci_lo, ci_hi = mean_diff_ci_welch(before_scores, after_scores, alpha=ALPHA)

    decision = 'Reject H0' if p_val < ALPHA else 'Fail to reject H0'

    result = {
        'metadata': {
            'dataset': dataset_path,
            'analysis_date': datetime.now().isoformat(),
            'milestone_date': MILESTONE_DATE.strftime('%Y-%m-%d'),
            'alpha': ALPHA,
            'two_tailed': True,
        },
        'hypotheses': {
            'H0': 'No significant difference in mean sentiment scores before vs after 2024 milestone (mu_before = mu_after)',
            'H1': 'Significant difference in mean sentiment scores before vs after 2024 milestone (mu_before != mu_after)'
        },
        'group_stats': {
            'before': {'n': int(len(before_scores)), 'mean_sentiment': mean_before, 'std': float(np.std(before_scores, ddof=1))},
            'after': {'n': int(len(after_scores)), 'mean_sentiment': mean_after, 'std': float(np.std(after_scores, ddof=1))},
            'mean_difference_after_minus_before': diff,
            'ci_95_mean_difference': [ci_lo, ci_hi]
        },
        'normality_shapiro_wilk': {
            'before': {'W': sw_b_stat, 'p': sw_b_p, 'n_used': sw_b_n},
            'after': {'W': sw_a_stat, 'p': sw_a_p, 'n_used': sw_a_n},
            'both_normal': bool(both_normal)
        },
        'inferential_test': {
            'test_used': test_name,
            'test_statistic_label': test_stat_label,
            'test_statistic': test_stat_value,
            'p_value': float(p_val),
            'cohens_d': float(d_val),
            'decision_alpha_0_05': decision
        }
    }

    with open('policy_milestone_hypothesis_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    make_visuals(before_scores, after_scores, mean_before, mean_after, ci_lo, ci_hi)

    print('\n' + '-' * 80)
    print('RESULT SUMMARY')
    print('-' * 80)
    print(f"Normality (Shapiro): before p={sw_b_p:.4g}, after p={sw_a_p:.4g}")
    print(f"Test used: {test_name}")
    print(f"{test_stat_label} = {test_stat_value:.4f}, p = {p_val:.4g}")
    print(f"Cohen's d (After-Before): {d_val:.4f}")
    print(f"95% CI (After-Before): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Decision: {decision}")
    print('Saved: policy_milestone_hypothesis_results.json')
    print('Saved visuals:')
    print('  visualizations/policy_milestone_boxplot.png')
    print('  visualizations/policy_milestone_mean_ci.png')
    print('  visualizations/policy_milestone_distribution.png')


if __name__ == '__main__':
    main()
