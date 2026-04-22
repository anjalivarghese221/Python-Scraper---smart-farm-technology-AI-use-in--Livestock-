import json
from pathlib import Path

rows=json.loads(Path('classified_sentiment_data_roberta.json').read_text(encoding='utf-8'))
needle='ZVKoDap4iKk'.lower()
for r in rows:
    blob=' '.join([(r.get('title') or ''),(r.get('text') or ''),(r.get('raw_text') or ''),(r.get('url') or '')]).lower()
    if needle in blob:
        print('date:',r.get('created_date'))
        print('subreddit:',r.get('subreddit'))
        print('sentiment:',r.get('sentiment'))
        print('confidence:',r.get('sentiment_confidence'))
        print('title:',r.get('title'))
        print('content:\n'+(r.get('text') or r.get('raw_text') or '').strip())
        print('url:',r.get('url'))
        break
else:
    print('not found')
