import json
import re
from collections import Counter

with open('classified_sentiment_data_clean_high_coverage.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

counts = Counter()
rows_with = Counter()

for row in data:
    text = (
        (row.get('cleaned_text', '') or '') + ' ' +
        (row.get('clean_text', '') or '') + ' ' +
        (row.get('title', '') or '') + ' ' +
        (row.get('text', '') or '') + ' ' +
        (row.get('raw_text', '') or '')
    ).lower()
    sentiment = (row.get('sentiment') or 'neutral').lower()
    n = len(re.findall(r'\bmoisture\b', text))
    if n > 0:
        counts[sentiment] += n
        rows_with[sentiment] += 1

print('moisture_token_counts_by_sentiment=', dict(counts))
print('rows_containing_moisture_by_sentiment=', dict(rows_with))
