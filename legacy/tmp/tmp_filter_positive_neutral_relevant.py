import json
import re
from pathlib import Path

p = Path('classified_sentiment_data_roberta.json')
rows = json.loads(p.read_text())

tech = re.compile(
    r"\b(ai|artificial intelligence|automation|automated|machine learning|drone|sensor|iot|precision|robot|computer vision|satellite|data center|fermentation|gene[- ]editing)\b",
    re.I,
)
ag = re.compile(
    r"\b(farm|farming|agri|agriculture|livestock|cattle|beef|dairy|ranch|pasture|herd|poultry|hog|swine|chicken|animal husbandry)\b",
    re.I,
)
exclude = re.compile(
    r"\b(state farm|insurance|programming|coding|real estate|home sensor|askreddit|mmo|game|skyrim|gangstalking|epstein|lottery)\b",
    re.I,
)

def relevant(r):
    txt = ' '.join(str(r.get(k, '')) for k in ['title', 'text', 'raw_text', 'clean_text'])
    if not (tech.search(txt) and ag.search(txt)):
        return False
    if exclude.search(txt):
        return False
    return True

for target in ('positive', 'neutral'):
    cand = []
    for r in rows:
        if r.get('source') != 'reddit':
            continue
        if (r.get('sentiment') or '').lower() != target:
            continue
        if not relevant(r):
            continue
        cand.append(r)

    cand.sort(key=lambda x: float(x.get('sentiment_confidence', 0)), reverse=True)
    print(f"\n=== {target.upper()} ===")
    print('count', len(cand))
    for i, r in enumerate(cand[:25], 1):
        t = (r.get('title') or '').replace('\n', ' ').strip()
        print(
            f"\n[{i}] conf={float(r.get('sentiment_confidence', 0)):.4f} "
            f"sub=r/{r.get('subreddit', '')} date={r.get('created_date', '')}"
        )
        print(t)
        print(r.get('url', ''))
