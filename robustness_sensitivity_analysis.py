#!/usr/bin/env python3
"""
Robustness and sensitivity checks for smart-farming sentiment inference.

Implements:
- Time-window sensitivity (milestone shift ±7 days)
- Reanalysis excluding top 5% volume days
- Subsampling stability test
- Multiple-testing correction (Bonferroni + Benjamini-Hochberg FDR)
"""

import json
from datetime import datetime, timedelta
from collections import Counter
from math import sqrt

import numpy as np
from scipy import stats

ALPHA = 0.05
BASE_CUTOFF = datetime(2025, 7, 1)


def load_data():
    for p in ['classified_sentiment_data_clean_expanded.json', 'classified_sentiment_data_clean.json', 'classified_sentiment_data.json']:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f), p
        except FileNotFoundError:
            continue
    raise FileNotFoundError('Dataset not found')


def parse_date(s):
    if not s:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def score(label):
    l = str(label or '').lower()
    if l == 'positive':
        return 1.0
    if l == 'negative':
        return -1.0
    return 0.0


def cohens_d(x, y):
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
    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    n1, n2 = len(x), len(y)
    diff = m2 - m1
    se = sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return float(diff), float(diff), float(diff)
    num = (v1 / n1 + v2 / n2) ** 2
    den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else n1 + n2 - 2
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(diff), float(diff - tcrit * se), float(diff + tcrit * se)


def shapiro_p(a):
    a = np.asarray(a, dtype=float)
    if len(a) > 5000:
        rng = np.random.default_rng(42)
        a = rng.choice(a, 5000, replace=False)
    return float(stats.shapiro(a)[1])


def run_before_after_test(rows, cutoff):
    before, after = [], []
    for r in rows:
        dt = parse_date(r.get('created_date', ''))
        if not dt:
            continue
        s = score(r.get('sentiment'))
        if dt < cutoff:
            before.append(s)
        else:
            after.append(s)
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)

    p_b = shapiro_p(before) if len(before) >= 3 else 0.0
    p_a = shapiro_p(after) if len(after) >= 3 else 0.0
    normal = (p_b > ALPHA) and (p_a > ALPHA)

    if normal:
        stat, p = stats.ttest_ind(before, after, equal_var=False)
        stat_label = 't'
        test = 'Welch independent t-test'
    else:
        stat, p = stats.mannwhitneyu(before, after, alternative='two-sided')
        stat_label = 'U'
        test = 'Mann-Whitney U (two-sided)'

    d = cohens_d(before, after)
    diff, lo, hi = welch_ci(before, after)
    return {
        'cutoff': cutoff.strftime('%Y-%m-%d'),
        'n_before': int(len(before)),
        'n_after': int(len(after)),
        'mean_before': float(np.mean(before)) if len(before) else None,
        'mean_after': float(np.mean(after)) if len(after) else None,
        'test_used': test,
        'stat_label': stat_label,
        'stat': float(stat),
        'p_value': float(p),
        'cohens_d_after_minus_before': float(d),
        'ci_95_after_minus_before': [float(lo), float(hi)],
        'decision': 'Reject H0' if p < ALPHA else 'Fail to reject H0'
    }


def bonferroni(ps):
    m = len(ps)
    return [min(1.0, p * m) for p in ps]


def bh_fdr(ps):
    # Benjamini-Hochberg
    m = len(ps)
    order = np.argsort(ps)
    ranked = np.array(ps)[order]
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(m, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def main():
    rows, dataset = load_data()

    # Base + shifted cutoffs
    base = run_before_after_test(rows, BASE_CUTOFF)
    minus7 = run_before_after_test(rows, BASE_CUTOFF - timedelta(days=7))
    plus7 = run_before_after_test(rows, BASE_CUTOFF + timedelta(days=7))

    # Excluding top 5% volume days
    daily = Counter()
    for r in rows:
        d = r.get('created_date', '')
        if isinstance(d, str) and len(d) >= 10:
            daily[d[:10]] += 1
    thresh = np.percentile(list(daily.values()), 95)
    high_days = {d for d, c in daily.items() if c >= thresh}
    reduced = [r for r in rows if (r.get('created_date', '')[:10] not in high_days)]
    excl_top_days = run_before_after_test(reduced, BASE_CUTOFF)

    # Subsampling stability (80% sample, 100 runs)
    rng = np.random.default_rng(42)
    pvals, ds = [], []
    n = len(rows)
    k = int(n * 0.8)
    for _ in range(100):
        idx = rng.choice(n, size=k, replace=False)
        sample = [rows[i] for i in idx]
        r = run_before_after_test(sample, BASE_CUTOFF)
        pvals.append(r['p_value'])
        ds.append(r['cohens_d_after_minus_before'])
    subsampling = {
        'runs': 100,
        'sample_fraction': 0.8,
        'median_p_value': float(np.median(pvals)),
        'significant_rate_p_lt_0_05': float(np.mean(np.array(pvals) < 0.05)),
        'median_cohens_d': float(np.median(ds))
    }

    # Multiple testing correction over key p-values
    p_labels = ['base_cutoff', 'cutoff_minus_7d', 'cutoff_plus_7d', 'exclude_top_5pct_volume_days']
    p_raw = [base['p_value'], minus7['p_value'], plus7['p_value'], excl_top_days['p_value']]
    p_bonf = bonferroni(p_raw)
    p_fdr = bh_fdr(p_raw)

    multiple = []
    for lbl, pr, pb, pf in zip(p_labels, p_raw, p_bonf, p_fdr):
        multiple.append({
            'test': lbl,
            'p_raw': float(pr),
            'p_bonferroni': float(pb),
            'p_fdr_bh': float(pf),
            'significant_bonferroni': bool(pb < ALPHA),
            'significant_fdr': bool(pf < ALPHA)
        })

    out = {
        'metadata': {
            'dataset': dataset,
            'analysis_date': datetime.now().isoformat(),
            'alpha': ALPHA,
            'base_cutoff': BASE_CUTOFF.strftime('%Y-%m-%d')
        },
        'time_window_sensitivity': {
            'base': base,
            'minus_7_days': minus7,
            'plus_7_days': plus7
        },
        'exclude_top_5pct_volume_days': {
            'n_original': len(rows),
            'n_reduced': len(reduced),
            'n_removed': len(rows) - len(reduced),
            'threshold_posts_per_day': float(thresh),
            'result': excl_top_days
        },
        'subsampling_test': subsampling,
        'multiple_testing_correction': multiple
    }

    with open('robustness_sensitivity_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('=' * 80)
    print('ROBUSTNESS & SENSITIVITY ANALYSIS COMPLETE')
    print('=' * 80)
    print(f"Base test p={base['p_value']:.4g}, d={base['cohens_d_after_minus_before']:.3f}")
    print(f"-7d p={minus7['p_value']:.4g}, +7d p={plus7['p_value']:.4g}")
    print(f"Excl top 5% volume days p={excl_top_days['p_value']:.4g}")
    print(f"Subsampling sig-rate={subsampling['significant_rate_p_lt_0_05']:.2%}")
    print('Saved: robustness_sensitivity_results.json')


if __name__ == '__main__':
    main()
