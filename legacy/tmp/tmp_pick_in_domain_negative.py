import json
import re

with open('classified_sentiment_data_domain_smart_farming_livestock.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

neg = [x for x in data if x.get('sentiment') == 'negative']

livestock_re = re.compile(r"\b(livestock|dairy|cattle|cow|herd|farm|farming|ranch)\b", re.I)
tech_re = re.compile(r"\b(ai|artificial intelligence|machine learning|computer vision|sensor|sensors|iot|robot|robotics|automation|smart|precision)\b", re.I)

candidates = []
for x in neg:
    title = x.get('title', '') or ''
    body = x.get('text', '') or x.get('raw_text', '') or ''
    blob = f"{title} {body}"
    if livestock_re.search(blob) and tech_re.search(blob):
        # prioritize medium-length, clearly topical bodies
        length = len(body.strip())
        score = 0
        if 120 <= length <= 2000:
            score += 4
        elif length > 2000:
            score += 2
        else:
            score += 1
        if x.get('subreddit', '').lower() in {'agriculture', 'farming', 'livestock', 'dairyfarming', 'agtech', 'precisionag', 'cattle', 'iot', 'machinelearning'}:
            score += 2
        score += int(float(x.get('sentiment_confidence', 0)) * 100)
        candidates.append((score, x))

candidates.sort(key=lambda t: t[0], reverse=True)

if not candidates:
    print('NONE')
else:
    item = candidates[0][1]
    title = item.get('title', 'N/A')
    body = (item.get('text', '') or item.get('raw_text', '') or '').strip()
    print('Title:', title)
    print('Date:', item.get('created_date', 'N/A'))
    print('Subreddit:', item.get('subreddit', 'N/A'))
    print('Sentiment:', item.get('sentiment'))
    print('Confidence:', round(float(item.get('sentiment_confidence', 0)), 4))
    print('---CONTENT_START---')
    print(body)
    print('---CONTENT_END---')
