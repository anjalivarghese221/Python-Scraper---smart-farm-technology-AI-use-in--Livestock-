import json
import re

with open('classified_sentiment_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

livestock_terms = [
    'livestock', 'dairy', 'cattle', 'cow', 'herd', 'farm', 'farming',
    'ranch', 'beef', 'poultry', 'sheep', 'goat'
]
tech_terms = [
    'ai', 'artificial intelligence', 'machine learning', 'computer vision',
    'sensor', 'sensors', 'iot', 'robot', 'robotics', 'automation', 'smart',
    'precision', 'drone', 'algorithm', 'predictive'
]


def has_any(text, terms):
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)

rows = []
for x in data:
    txt = (x.get('title', '') + ' ' + (x.get('text') or x.get('raw_text') or '') + ' ' + x.get('subreddit', '')).lower()
    has_l = has_any(txt, livestock_terms)
    has_t = has_any(txt, tech_terms)
    has_island = bool(re.search(r"\bisland\b", txt))
    rows.append((has_l, has_t, has_island))

print('total', len(data))
print('both_livestock_and_tech', sum(1 for l, t, _ in rows if l and t))
print('either_livestock_or_tech', sum(1 for l, t, _ in rows if l or t))
print('either_no_island', sum(1 for l, t, i in rows if (l or t) and not i))
print('both_no_island', sum(1 for l, t, i in rows if l and t and not i))
