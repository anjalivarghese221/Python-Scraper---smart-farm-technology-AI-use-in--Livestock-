import json
import re
import textwrap
from pathlib import Path

DATA_PATH = Path('classified_sentiment_data_clean_high_coverage.json')
OUTPUT_PATH = Path('moisture_posts_for_manual_review.json')

data = json.loads(DATA_PATH.read_text())
rows = []
for r in data:
    text = ((r.get('title', '') or '') + ' ' + (r.get('text', '') or '') + ' ' + (r.get('raw_text', '') or ''))
    if re.search(r'\bmoisture\b', text.lower()):
        rows.append(r)

rows = sorted(rows, key=lambda x: (x.get('created_date') or ''))
print('TOTAL', len(rows))

export_rows = []

for i, r in enumerate(rows, 1):
    title = (r.get('title') or '').replace('\n', ' ').strip()
    txt = (r.get('text') or r.get('raw_text') or '').replace('\n', ' ').strip()
    snippet = textwrap.shorten(txt, width=220, placeholder='...')
    print(f"\n[{i}] date={r.get('created_date','N/A')} | subreddit=r/{r.get('subreddit','')} | sentiment={r.get('sentiment','')} ({r.get('sentiment_confidence',0):.3f})")
    print(f"title: {title}")
    print(f"content: {snippet}")
    print(f"url: {r.get('url','')}")

    export_rows.append({
        'date': r.get('created_date', 'N/A'),
        'subreddit': r.get('subreddit', ''),
        'sentiment': r.get('sentiment', ''),
        'sentiment_confidence': r.get('sentiment_confidence', 0),
        'title': r.get('title', ''),
        'content': r.get('text') or r.get('raw_text') or '',
        'url': r.get('url', ''),
    })

OUTPUT_PATH.write_text(json.dumps(export_rows, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nSaved JSON review file: {OUTPUT_PATH}")
