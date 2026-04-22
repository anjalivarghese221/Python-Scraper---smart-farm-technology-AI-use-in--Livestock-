import json
import re
from pathlib import Path

p = Path('classified_sentiment_data_roberta.json')
rows = json.loads(p.read_text())

tech = re.compile(
    r"\b(ai|artificial intelligence|automation|automated|machine learning|ml\b|drone|sensor|iot|precision|robot|computer vision|satellite|data center)\b",
    re.I,
)
ag = re.compile(
    r"\b(farm|farming|agri|agriculture|livestock|cattle|beef|dairy|ranch|pasture|herd|poultry|hog|swine|chicken)\b",
    re.I,
)
exclude = re.compile(
    r"\b(state farm|insurance|programming|coding|real estate|home sensor|askreddit|plumbers?)\b",
    re.I,
)
subs_allow = {
    'cattle',
    'ranching',
    'dairyfarming',
    'farming',
    'agriculture',
    'precisionag',
    'agtech',
    'homesteading',
    'livestock',
    'beef',
    'dairy',
    'clarksonsfarm',
    'futurology',
    'legaladvice',
    'amazonemployees',
}

cand = []
for r in rows:
    if r.get('source') != 'reddit':
        continue
    if (r.get('sentiment') or '').lower() != 'negative':
        continue

    text = ' '.join(str(r.get(k, '')) for k in ['title', 'text', 'raw_text', 'clean_text'])
    if not (tech.search(text) and ag.search(text)):
        continue
    if exclude.search(text):
        continue

    sub = (r.get('subreddit') or '').lower()
    strong = bool(tech.search(text) and ag.search(text))
    if not (sub in subs_allow or strong):
        continue

    cand.append(r)

cand.sort(key=lambda x: float(x.get('sentiment_confidence', 0)), reverse=True)

print('count', len(cand))
for i, r in enumerate(cand[:30], 1):
    t = (r.get('title') or '').replace('\n', ' ').strip()
    print(f"\n[{i}] conf={float(r.get('sentiment_confidence', 0)):.4f} sub=r/{r.get('subreddit', '')} date={r.get('created_date', '')}")
    print(t)
    print(r.get('url', ''))
