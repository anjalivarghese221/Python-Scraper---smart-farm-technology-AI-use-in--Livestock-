import json
import re
from pathlib import Path

rows = json.loads(Path('classified_sentiment_data_roberta.json').read_text(encoding='utf-8'))

livestock = re.compile(r"\b(livestock|dairy|cattle|cow|cows|sheep|goat|goats|swine|pig|pigs|poultry|herd|beef|milk|milking)\b", re.I)
ai = re.compile(r"\b(ai|artificial intelligence|machine learning|computer vision|iot|sensor|sensors|robot|robotics|automation|automated|precision|digital)\b", re.I)

relevant_subreddits = {
    'agriculture', 'farming', 'farminguk', 'livestock', 'dairy', 'dairyfarming',
    'agtech', 'precisionag', 'precisionagriculture', 'iot', 'machinelearning',
    'artificialinteligence', 'cellularagriculture', 'wheresthebeef', 'meridairy'
}

exclude = re.compile(
    r"\b(epstein|lottery|uber|reef|cathelp|phasmophobia|mmo|yogscast|gift ideas|worldbuilding|gangstalking|conspiracy)\b",
    re.I,
)

results = {k: [] for k in ['negative', 'positive', 'neutral']}

for r in rows:
    label = (r.get('sentiment') or '').lower()
    if label not in results:
        continue

    url = r.get('url') or ''
    if not url.startswith('https://www.reddit.com/'):
        continue

    sub = (r.get('subreddit') or '').lower()
    text = ' '.join([
        r.get('title') or '',
        r.get('text') or '',
        r.get('raw_text') or '',
        sub,
    ])

    if exclude.search(text):
        continue

    if not (livestock.search(text) and ai.search(text)):
        continue

    if sub not in relevant_subreddits:
        continue

    results[label].append(r)

for k in results:
    results[k].sort(key=lambda x: float(x.get('sentiment_confidence') or 0), reverse=True)

out_path = Path('roberta_reddit_domain_links_strict.txt')
lines = []
for label in ['negative', 'positive', 'neutral']:
    lines.append(f"\n=== {label.upper()} (STRICT RELEVANCE) ===")
    items = results[label][:10]
    if not items:
        lines.append('None found')
        continue

    for i, r in enumerate(items, 1):
        lines.append(
            f"[{i}] conf={float(r.get('sentiment_confidence') or 0):.4f} | date={r.get('created_date')} | "
            f"sub=r/{r.get('subreddit')}"
        )
        lines.append(f"title: {(r.get('title') or '').replace(chr(10), ' ')}")
        lines.append(f"url: {r.get('url')}")

out_path.write_text('\n'.join(lines), encoding='utf-8')
print(f'saved={out_path}')
