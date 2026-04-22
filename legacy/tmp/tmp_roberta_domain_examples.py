import json
import re
from pathlib import Path

DATA = Path('classified_sentiment_data_roberta.json')
OUT = Path('roberta_domain_examples_with_context.txt')

domain_pat = re.compile(
    r"\b(livestock|dairy|cattle|cow|farm|farming|agriculture|agricultural|sheep|goat|swine|poultry)\b"
    r".*|.*\b(ai|artificial intelligence|machine learning|iot|sensor|sensors|robot|robotics|computer vision|automation|precision)\b",
    re.I,
)

tech_pat = re.compile(r"\b(ai|artificial intelligence|machine learning|iot|sensor|sensors|robot|robotics|computer vision|automation|precision)\b", re.I)
live_pat = re.compile(r"\b(livestock|dairy|cattle|cow|farm|farming|agriculture|agricultural|sheep|goat|swine|poultry)\b", re.I)

rows = json.loads(DATA.read_text(encoding='utf-8'))

lines = []
for label in ['negative', 'positive', 'neutral']:
    cands = []
    for r in rows:
        sent = (r.get('sentiment') or '').lower()
        if sent != label:
            continue
        full = f"{r.get('title','')} {r.get('text','')} {r.get('raw_text','')} {r.get('subreddit','')}"
        if not (tech_pat.search(full) and live_pat.search(full)):
            continue
        cands.append(r)

    cands.sort(key=lambda x: x.get('sentiment_confidence', 0), reverse=True)
    if not cands:
        lines.append(f"\n===== {label.upper()} =====\nNo domain-matched example found.\n")
        continue

    r = cands[0]
    title = (r.get('title') or '').strip()
    content = (r.get('text') or r.get('raw_text') or '').strip()

    lines.append(f"\n===== {label.upper()} (DOMAIN-MATCHED) =====")
    lines.append(f"date: {r.get('created_date')}")
    lines.append(f"subreddit: {r.get('subreddit')}")
    lines.append(f"confidence: {r.get('sentiment_confidence')}")
    lines.append(f"title: {title}")
    lines.append("content:")
    lines.append(content)
    lines.append(f"url: {r.get('url')}")
    lines.append("is_ai_term_present: yes")
    lines.append("is_livestock_or_ag_term_present: yes")

OUT.write_text("\n".join(lines), encoding='utf-8')
print(f"saved={OUT}")
