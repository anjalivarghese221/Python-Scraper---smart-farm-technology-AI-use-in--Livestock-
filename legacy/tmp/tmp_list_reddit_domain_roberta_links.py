import json
import re
from pathlib import Path

rows = json.loads(Path('classified_sentiment_data_roberta.json').read_text(encoding='utf-8'))

livestock = re.compile(r"\b(livestock|dairy|cattle|cow|cows|sheep|goat|goats|swine|pig|pigs|poultry|herd|beef|milk)\b", re.I)
ai = re.compile(r"\b(ai|artificial intelligence|machine learning|computer vision|iot|sensor|sensors|robot|robotics|automation|automated|precision)\b", re.I)

for label in ['negative', 'positive', 'neutral']:
    candidates = []
    for r in rows:
        if (r.get('sentiment') or '').lower() != label:
            continue
        url = r.get('url') or ''
        if not url.startswith('https://www.reddit.com/'):
            continue
        text = ' '.join([
            r.get('title') or '',
            r.get('text') or '',
            r.get('raw_text') or '',
            r.get('subreddit') or '',
        ])
        if livestock.search(text) and ai.search(text):
            candidates.append(r)

    candidates.sort(key=lambda x: float(x.get('sentiment_confidence') or 0), reverse=True)

    print(f"\n=== {label.upper()} ===")
    for i, r in enumerate(candidates[:5], 1):
        print(f"[{i}] conf={float(r.get('sentiment_confidence') or 0):.4f} date={r.get('created_date')} sub=r/{r.get('subreddit')}")
        print(f"title: {(r.get('title') or '').replace(chr(10), ' ')}")
        print(f"url: {r.get('url')}")
