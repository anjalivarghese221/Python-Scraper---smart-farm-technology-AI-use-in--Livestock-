import json
from pathlib import Path

DATA = Path('classified_sentiment_data_roberta.json')
OUT = Path('roberta_selected_domain_examples_full.txt')

rows = json.loads(DATA.read_text(encoding='utf-8'))

# Pick one clearly domain-relevant example per sentiment
selectors = [
    ('negative', 'frontiers'),
    ('positive', '0o33ykp9s0sf1'),
    ('neutral', 'michigan-state-university'),
]

lines = []
for label, needle in selectors:
    match = None
    for r in rows:
        if (r.get('sentiment') or '').lower() != label:
            continue
        blob = ' '.join([
            r.get('title', '') or '',
            r.get('text', '') or '',
            r.get('raw_text', '') or '',
            r.get('url', '') or '',
        ]).lower()
        if needle in blob:
            match = r
            break

    lines.append('\n' + '=' * 90)
    lines.append(label.upper())
    if not match:
        lines.append(f'No match found for selector: {needle}')
        continue

    lines.append(f"date: {match.get('created_date')}")
    lines.append(f"subreddit: {match.get('subreddit')}")
    lines.append(f"confidence: {match.get('sentiment_confidence')}")
    lines.append(f"title: {match.get('title')}")
    lines.append('content:')
    lines.append((match.get('text') or match.get('raw_text') or '').strip())
    lines.append(f"url: {match.get('url')}")

OUT.write_text('\n'.join(lines), encoding='utf-8')
print(f'saved={OUT}')
