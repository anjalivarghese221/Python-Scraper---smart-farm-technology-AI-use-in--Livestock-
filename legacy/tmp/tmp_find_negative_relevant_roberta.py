import json
import re
from pathlib import Path

rows = json.loads(Path('classified_sentiment_data_roberta.json').read_text(encoding='utf-8'))

# Topic relevance: must include BOTH livestock/ag and AI/automation signals
livestock = re.compile(r"\b(livestock|dairy|cattle|cow|cows|sheep|goat|goats|swine|pig|pigs|poultry|herd|beef|milk|milking|farm|farming|agriculture|agricultural)\b", re.I)
ai = re.compile(r"\b(ai|artificial intelligence|machine learning|computer vision|iot|sensor|sensors|robot|robotics|automation|automated|precision|digital)\b", re.I)

# Exclude obvious off-topic noise contexts
exclude = re.compile(
    r"\b(epstein|lottery|uber|reef|cathelp|phasmophobia|mmo|yogscast|gift ideas|worldbuilding|gangstalking|conspiracy|gaming|game)\b",
    re.I,
)

candidates = []
for r in rows:
    if (r.get('sentiment') or '').lower() != 'negative':
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

    if exclude.search(text):
        continue

    if livestock.search(text) and ai.search(text):
        candidates.append(r)

candidates.sort(key=lambda x: float(x.get('sentiment_confidence') or 0), reverse=True)

print(f"count={len(candidates)}")
for i, r in enumerate(candidates[:15], 1):
    title = (r.get('title') or '').replace('\n', ' ').strip()
    content = (r.get('text') or r.get('raw_text') or '').replace('\n', ' ').strip()
    if len(content) > 220:
        content = content[:220] + '...'
    print(f"\n[{i}] conf={float(r.get('sentiment_confidence') or 0):.4f} | date={r.get('created_date')} | sub=r/{r.get('subreddit')}")
    print(f"title: {title}")
    print(f"content: {content}")
    print(f"url: {r.get('url')}")
