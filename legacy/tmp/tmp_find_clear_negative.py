import json
import re

with open('classified_sentiment_data_domain_smart_farming_livestock.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

neg = [x for x in data if x.get('sentiment') == 'negative']
neg_cues = re.compile(r"\b(risk|cost|expensive|problem|issue|fail|failing|ban|scared|concern|worry|crisis|decline|loss|toxic|danger|barrier|debt|drop|bad|worse|worst)\b", re.I)

cands = []
for x in neg:
    text = (x.get('text') or x.get('raw_text') or '').strip()
    if len(text) < 80:
        continue
    blob = (x.get('title', '') + ' ' + text)
    score = 0
    if neg_cues.search(blob):
        score += 3
    score += min(len(text) // 200, 5)
    if x.get('subreddit') in {'Agriculture', 'farming', 'livestock', 'dairyfarming', 'agtech', 'Cattle', 'Ranching'}:
        score += 1
    cands.append((score, x))

cands.sort(key=lambda t: t[0], reverse=True)

if not cands:
    print('NONE')
else:
    s, item = cands[0]
    content = (item.get('text') or item.get('raw_text') or '').replace('\n', ' ').strip()
    print('Title:', item.get('title', 'N/A'))
    print('Content:', content[:1200])
    print('Date:', item.get('created_date', 'N/A'))
    print('Subreddit:', item.get('subreddit', 'N/A'))
    print('Sentiment:', item.get('sentiment'))
    print('Confidence:', round(float(item.get('sentiment_confidence', 0)), 4))
    print('Score:', s)
