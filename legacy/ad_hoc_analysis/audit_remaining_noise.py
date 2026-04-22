import json
import re
from collections import Counter, defaultdict

INPUT = 'classified_sentiment_data_clean_high_coverage.json'

patterns = {
    'exam_cheating_spam': re.compile(r'\b(clep exam|proctor(track|io|u)|take my exam|pay someone to|hiraedu|whatsapp\s*:\s*\+?\d+)\b', re.I),
    'insurance_false_positive': re.compile(r'\bstate farm\b', re.I),
    'generic_promo_spam': re.compile(r'\b(100% success|call:\s*\+?\d+|contact us for help)\b', re.I),
    'non_domain_game_noise': re.compile(r'\b(acnh|totk|clash of clans|stormworks)\b', re.I),
}

with open(INPUT, 'r', encoding='utf-8') as f:
    data = json.load(f)

flagged = defaultdict(list)
for row in data:
    text = (row.get('title', '') + ' ' + (row.get('text') or row.get('raw_text') or '') + ' ' + row.get('subreddit', '')).lower()
    for name, rx in patterns.items():
        if rx.search(text):
            flagged[name].append(row)

print(f'total_rows={len(data)}')
for name, rows in flagged.items():
    print(f'{name}={len(rows)}')

print('\nTop subreddits still present (potentially broad):')
subs = Counter((r.get('subreddit') or 'unknown') for r in data)
for sub, cnt in subs.most_common(20):
    print(f'  r/{sub}: {cnt}')

print('\nSample flagged rows:')
for name, rows in flagged.items():
    print(f'\n[{name}]')
    for r in rows[:3]:
        content = (r.get('text') or r.get('raw_text') or '').replace('\n', ' ')
        print(f"- r/{r.get('subreddit','')} | {r.get('created_date','')} | {r.get('title','')[:110]}")
        print(f"  {content[:220]}{'...' if len(content)>220 else ''}")
