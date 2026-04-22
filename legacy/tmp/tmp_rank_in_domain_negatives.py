import json
import re

with open('classified_sentiment_data_domain_smart_farming_livestock.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

neg = [x for x in data if x.get('sentiment') == 'negative']
livestock_re = re.compile(r"\b(livestock|dairy|cattle|cow|herd|farm|farming|ranch)\b", re.I)
tech_re = re.compile(r"\b(ai|artificial intelligence|machine learning|computer vision|sensor|sensors|iot|robot|robotics|automation|smart|precision|drone)\b", re.I)

cands = []
for x in neg:
    title = x.get('title', '') or ''
    body = x.get('text', '') or x.get('raw_text', '') or ''
    blob = f"{title} {body}"
    if livestock_re.search(blob) and tech_re.search(blob):
        length = len(body.strip())
        score = 0
        if 120 <= length <= 2200:
            score += 4
        elif length > 2200:
            score += 2
        else:
            score += 1
        if x.get('subreddit', '').lower() in {'agriculture', 'farming', 'livestock', 'dairyfarming', 'agtech', 'precisionag', 'cattle', 'iot', 'machinelearning'}:
            score += 2
        score += int(float(x.get('sentiment_confidence', 0)) * 100)
        cands.append((score, x))

cands.sort(key=lambda t: t[0], reverse=True)

for i, (s, it) in enumerate(cands[:8], 1):
    body = (it.get('text', '') or it.get('raw_text', '') or '').strip().replace('\n', ' ')
    print(f"{i}|{s}|{it.get('subreddit')}|{it.get('created_date')}|{float(it.get('sentiment_confidence', 0)):.4f}")
    print(it.get('title', ''))
    print(body[:240])
    print('---')
