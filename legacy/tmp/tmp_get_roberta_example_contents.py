import json
from pathlib import Path

out_lines = []

data = json.loads(Path('classified_sentiment_data_roberta.json').read_text())

targets = [
    ('negative', '2025-09-05', 'Agriculture', 'https://i.redd.it/6dr04zhyucnf1.jpeg'),
    ('positive', '2021-11-30', 'Yogscast', 'https://www.reddit.com/r/Yogscast/comments/r5tg29/jingle_jam_2021_schedule_and_links/hmowdu6/'),
    ('neutral', '2020-09-16', 'livestock', 'https://www.reddit.com/gallery/ituosq'),
]

for label, date, subreddit, url in targets:
    row = next((
        r for r in data
        if (r.get('sentiment') or '').lower() == label
        and r.get('created_date') == date
        and (r.get('subreddit') or '') == subreddit
        and (r.get('url') or '') == url
    ), None)

    out_lines.append('\n' + '=' * 90)
    out_lines.append(label.upper())
    if not row:
        out_lines.append('Not found')
        continue

    out_lines.append(f"date: {row.get('created_date')}")
    out_lines.append(f"subreddit: {row.get('subreddit')}")
    out_lines.append(f"confidence: {row.get('sentiment_confidence')}")
    out_lines.append(f"title: {row.get('title') or ''}")

    text = row.get('text') or row.get('raw_text') or ''
    out_lines.append('content:')
    out_lines.append(text)
    out_lines.append(f"url: {row.get('url')}")

out_path = Path('roberta_examples_full_content.txt')
out_path.write_text('\n'.join(out_lines), encoding='utf-8')
print(f"saved={out_path}")
