import json
import re

with open('classified_sentiment_data_domain_smart_farming_livestock.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

neg = [x for x in data if x.get('sentiment') == 'negative']
relevant_subs = {
    'Agriculture', 'farming', 'livestock', 'dairyfarming', 'Cattle',
    'agtech', 'PrecisionAg', 'IOT', 'MachineLearning', 'drones', 'arduino', 'PLC', 'Ranching'
}
livestock = r'(livestock|dairy|cattle|cow|herd|farm)'
tech = r'(ai|machine learning|sensor|camera|iot|robot|automation|computer vision|precision)'

cand = []
for x in neg:
    text = (x.get('title', '') + ' ' + (x.get('text') or x.get('raw_text') or '')).lower()
    if (x.get('subreddit') in relevant_subs or re.search(livestock, text)) and re.search(tech, text):
        cand.append(x)

cand.sort(key=lambda x: len((x.get('text') or x.get('raw_text') or '')), reverse=True)

if not cand:
    print('NONE')
else:
    item = cand[0]
    content = (item.get('text') or item.get('raw_text') or '').strip()
    print('Title:', item.get('title', 'N/A'))
    print('Content:', content[:900])
    print('Date:', item.get('created_date', 'N/A'))
    print('Subreddit:', item.get('subreddit', 'N/A'))
    print('Sentiment:', item.get('sentiment'))
    print('Confidence:', round(float(item.get('sentiment_confidence', 0)), 4))
