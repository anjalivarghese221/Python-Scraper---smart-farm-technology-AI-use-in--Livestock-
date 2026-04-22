import json
import re
from pathlib import Path

DATA = Path('classified_sentiment_data_roberta.json')
OUT = Path('roberta_livestock_ai_examples.txt')

livestock_pat = re.compile(r"\b(livestock|dairy|cattle|cow|cows|sheep|goat|goats|swine|pig|pigs|poultry|herd|ruminant|beef)\b", re.I)
ai_pat = re.compile(r"\b(ai|artificial intelligence|machine learning|computer vision|iot|sensor|sensors|robot|robotics|automation|precision)\b", re.I)

rows = json.loads(DATA.read_text(encoding='utf-8'))

lines = []
for label in ['negative', 'positive', 'neutral']:
    cands = []
    for r in rows:
        if (r.get('sentiment') or '').lower() != label:
            continue
        text = f"{r.get('title','')} {r.get('text','')} {r.get('raw_text','')} {r.get('subreddit','')}"
        if livestock_pat.search(text) and ai_pat.search(text):
            cands.append(r)

    cands.sort(key=lambda x: x.get('sentiment_confidence', 0), reverse=True)
    if not cands:
        lines.append(f"\n===== {label.upper()} =====\nNo strict livestock+AI match found.\n")
        continue

    r = cands[0]
    lines.append(f"\n===== {label.upper()} =====")
    lines.append(f"date: {r.get('created_date')}")
    lines.append(f"subreddit: {r.get('subreddit')}")
    lines.append(f"confidence: {r.get('sentiment_confidence')}")
    lines.append(f"title: {(r.get('title') or '').strip()}")
    lines.append("content:")
    lines.append((r.get('text') or r.get('raw_text') or '').strip())
    lines.append(f"url: {r.get('url')}")

OUT.write_text('\n'.join(lines), encoding='utf-8')
print(f"saved={OUT}")
