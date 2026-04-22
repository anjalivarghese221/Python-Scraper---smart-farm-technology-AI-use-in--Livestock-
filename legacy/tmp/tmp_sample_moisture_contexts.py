import json
import re

with open('classified_sentiment_data_clean_high_coverage.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
for r in data:
    text = (
        (r.get('title', '') or '') + ' ' +
        (r.get('text', '') or '') + ' ' +
        (r.get('raw_text', '') or '')
    )
    if re.search(r'\bmoisture\b', text.lower()):
        rows.append(r)

print('total_rows_with_moisture=', len(rows))

for sentiment in ['positive', 'negative', 'neutral']:
    subset = [r for r in rows if (r.get('sentiment') or '').lower() == sentiment]
    print(f"\n--- {sentiment.upper()} ({len(subset)}) ---")
    for r in subset[:6]:
        title = (r.get('title') or '').replace('\n', ' ').strip()
        text = (r.get('text') or r.get('raw_text') or '').replace('\n', ' ').strip()
        snippet = text[:220] + ('...' if len(text) > 220 else '')
        print(f"subreddit=r/{r.get('subreddit','')} | conf={r.get('sentiment_confidence',0):.3f}")
        print(f"title={title}")
        print(f"snippet={snippet}")
        print('-' * 40)
