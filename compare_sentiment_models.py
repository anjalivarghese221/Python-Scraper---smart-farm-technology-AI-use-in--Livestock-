#!/usr/bin/env python3
"""Compare sentiment outputs from Logistic Regression, RoBERTa, and VADER."""

import json
import os
from collections import Counter

LOGISTIC_FILE = 'classified_sentiment_data.json'
ROBERTA_FILE = 'classified_sentiment_data_roberta.json'
VADER_FILE = 'classified_sentiment_data_vader.json'
OUT_JSON = 'sentiment_model_comparison.json'
OUT_TXT = 'sentiment_model_comparison.txt'


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sentiment_distribution(rows):
    c = Counter((r.get('sentiment') or '').lower() for r in rows)
    total = len(rows) if rows else 1
    return {
        'total': len(rows),
        'positive': {'count': c.get('positive', 0), 'pct': (c.get('positive', 0) / total) * 100},
        'negative': {'count': c.get('negative', 0), 'pct': (c.get('negative', 0) / total) * 100},
        'neutral': {'count': c.get('neutral', 0), 'pct': (c.get('neutral', 0) / total) * 100},
    }


def pairwise_agreement(rows_a, rows_b, label_a='A', label_b='B'):
    n = min(len(rows_a), len(rows_b))
    if n == 0:
        return {'compared_rows': 0, 'agreement_count': 0, 'agreement_pct': 0.0}

    agreement = 0
    transitions = Counter()

    for i in range(n):
        a = (rows_a[i].get('sentiment') or '').lower()
        b = (rows_b[i].get('sentiment') or '').lower()
        if a == b:
            agreement += 1
        transitions[f'{a}->{b}'] += 1

    return {
        'compared_rows': n,
        'agreement_count': agreement,
        'agreement_pct': (agreement / n) * 100,
        'labels': {'left': label_a, 'right': label_b},
        'label_transitions': dict(transitions),
    }


def main():
    if not os.path.exists(LOGISTIC_FILE):
        raise FileNotFoundError(
            f'{LOGISTIC_FILE} not found. Run logistic pipeline/classifier first for comparison.'
        )

    logistic = load_json(LOGISTIC_FILE)
    roberta = load_json(ROBERTA_FILE) if os.path.exists(ROBERTA_FILE) else None
    vader = load_json(VADER_FILE) if os.path.exists(VADER_FILE) else None

    log_dist = sentiment_distribution(logistic)
    rob_dist = sentiment_distribution(roberta) if roberta is not None else None
    vader_dist = sentiment_distribution(vader) if vader is not None else None

    agreements = {}
    if roberta is not None:
        agreements['logistic_vs_roberta'] = pairwise_agreement(logistic, roberta, 'Logistic', 'RoBERTa')
    if vader is not None:
        agreements['logistic_vs_vader'] = pairwise_agreement(logistic, vader, 'Logistic', 'VADER')
    if roberta is not None and vader is not None:
        agreements['roberta_vs_vader'] = pairwise_agreement(roberta, vader, 'RoBERTa', 'VADER')

    comparison = {
        'logistic_file': LOGISTIC_FILE,
        'roberta_file': ROBERTA_FILE if roberta is not None else None,
        'vader_file': VADER_FILE if vader is not None else None,
        'logistic_distribution': log_dist,
        'roberta_distribution': rob_dist,
        'vader_distribution': vader_dist,
        'agreements': agreements,
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append('=' * 72)
    lines.append('SENTIMENT MODEL COMPARISON (LOGISTIC vs ROBERTA vs VADER)')
    lines.append('=' * 72)
    lines.append(f'Logistic rows: {log_dist["total"]}')
    if rob_dist is not None:
        lines.append(f'RoBERTa rows : {rob_dist["total"]}')
    if vader_dist is not None:
        lines.append(f'VADER rows   : {vader_dist["total"]}')
    lines.append('')

    lines.append('DISTRIBUTION')
    lines.append('-' * 72)
    for label in ['positive', 'negative', 'neutral']:
        base = f'{label:8s} | Logistic: {log_dist[label]["count"]:4d} ({log_dist[label]["pct"]:6.2f}%)'
        if rob_dist is not None:
            base += f' | RoBERTa: {rob_dist[label]["count"]:4d} ({rob_dist[label]["pct"]:6.2f}%)'
        if vader_dist is not None:
            base += f' | VADER: {vader_dist[label]["count"]:4d} ({vader_dist[label]["pct"]:6.2f}%)'
        lines.append(base)

    for key, agree in agreements.items():
        lines.append('')
        lines.append(f'PAIRWISE AGREEMENT: {key}')
        lines.append('-' * 72)
        lines.append(f'Compared rows : {agree["compared_rows"]}')
        lines.append(f'Agreement     : {agree["agreement_count"]} ({agree["agreement_pct"]:.2f}%)')
        lines.append('TOP LABEL TRANSITIONS')
        transitions = Counter(agree.get('label_transitions', {}))
        for k, v in transitions.most_common(12):
            lines.append(f'{k:20s}: {v}')

    text = '\n'.join(lines)
    print(text)

    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f'\nSaved: {OUT_JSON}')
    print(f'Saved: {OUT_TXT}')


if __name__ == '__main__':
    main()
