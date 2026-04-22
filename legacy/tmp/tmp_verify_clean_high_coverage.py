import json
import re

with open('classified_sentiment_data_clean_high_coverage.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

text_blob = '\n'.join(((x.get('title', '') + ' ' + (x.get('text') or x.get('raw_text') or '')).lower()) for x in d)
print('count', len(d))
print('island_matches', len(re.findall(r'\bisland\b', text_blob)))
