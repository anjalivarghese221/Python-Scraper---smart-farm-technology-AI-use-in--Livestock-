import json
from collections import Counter


def is_domain_relevant(row):
    text = (
        (row.get('cleaned_text', '') or '') + ' ' +
        (row.get('clean_text', '') or '') + ' ' +
        (row.get('title', '') or '') + ' ' +
        (row.get('text', '') or '') + ' ' +
        (row.get('raw_text', '') or '')
    ).lower()
    livestock_terms = {
        'livestock', 'cattle', 'cow', 'cows', 'dairy', 'beef', 'ruminant',
        'sheep', 'goat', 'goats', 'swine', 'pig', 'pigs', 'poultry',
        'broiler', 'layer', 'farm', 'farming', 'agri', 'agriculture',
        'ranch', 'ranching', 'herd'
    }
    tech_terms = {
        'ai', 'artificial intelligence', 'machine learning', 'ml',
        'computer vision', 'deep learning', 'neural', 'sensor', 'sensors',
        'iot', 'wearable', 'wearables', 'camera', 'cameras', 'automation',
        'robot', 'robots', 'robotics', 'smart farming', 'precision livestock',
        'precision agriculture', 'data', 'analytics', 'monitoring',
        'predictive', 'algorithm', 'algorithms', 'digital twin'
    }
    return any(t in text for t in livestock_terms) and any(t in text for t in tech_terms)

with open('classified_sentiment_data.json','r',encoding='utf-8') as f:
    data = json.load(f)

rows = [r for r in data if r.get('source','reddit')=='reddit']
rows = [r for r in rows if is_domain_relevant(r)]
neg = [r for r in rows if str(r.get('sentiment','')).lower()=='negative']

hits = []
for r in neg:
    txt = ((r.get('cleaned_text','') or '') + ' ' + (r.get('clean_text','') or '') + ' ' + (r.get('title','') or '') + ' ' + (r.get('text','') or '')).lower()
    if 'island' in txt:
        hits.append(r)

print('filtered rows',len(rows), 'negative',len(neg),'island in negative',len(hits))
print('subreddits', Counter([h.get('subreddit','') for h in hits]).most_common(10))
for i,h in enumerate(hits[:8],1):
    t=h.get('title','')
    body=(h.get('text','') or h.get('cleaned_text','') or h.get('clean_text',''))
    print('\n',i, h.get('subreddit',''), '|', t[:120])
    print((body or '')[:220].replace('\n',' '))
