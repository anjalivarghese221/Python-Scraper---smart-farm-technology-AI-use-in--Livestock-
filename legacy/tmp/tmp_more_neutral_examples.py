import json
import re

p = 'classified_sentiment_data_roberta.json'
rows = json.load(open(p))

tech = re.compile(
    r"\b(ai|artificial intelligence|automation|automated|machine learning|drone|sensor|iot|precision|robot|computer vision|satellite|fermentation|gene[- ]editing|smart farming)\b",
    re.I,
)
ag = re.compile(
    r"\b(farm|farming|agri|agriculture|livestock|cattle|beef|dairy|ranch|pasture|herd|poultry|pig|swine|chicken|greenhouse|crop|husbandry)\b",
    re.I,
)
exclude = re.compile(
    r"\b(codingjobs|jobhub|hiring|charity|lottery|metro|gpus vs fpga|mmo|game|astrology|askreddit)\b",
    re.I,
)

cand = []
for r in rows:
    if r.get('source') != 'reddit':
        continue
    if (r.get('sentiment') or '').lower() != 'neutral':
        continue

    txt = ' '.join(str(r.get(k, '')) for k in ['title', 'text', 'raw_text', 'clean_text'])
    if not (tech.search(txt) and ag.search(txt)):
        continue
    if exclude.search(txt):
        continue

    url = str(r.get('url', ''))
    if not url:
        continue

    cand.append(r)

cand.sort(key=lambda x: float(x.get('sentiment_confidence', 0)), reverse=True)

print('count', len(cand))
for i, r in enumerate(cand[:30], 1):
    print(
        f"[{i}] {float(r.get('sentiment_confidence', 0)):.4f} | "
        f"r/{r.get('subreddit', '')} | {r.get('created_date', '')} | "
        f"{(r.get('title', '') or '').strip()} | {r.get('url', '')}"
    )
