import json
from pathlib import Path

input_path = Path('moisture_posts_for_manual_review.json')
output_path = Path('moisture_posts_content_for_review.txt')

data = json.loads(input_path.read_text(encoding='utf-8'))

lines = []
for i, row in enumerate(data, 1):
    lines.append(f"[{i}] date={row.get('date','N/A')} | subreddit=r/{row.get('subreddit','')} | sentiment={row.get('sentiment','')} ({row.get('sentiment_confidence',0):.3f})")
    lines.append(f"title: {row.get('title','')}")
    lines.append("content:")
    lines.append(row.get('content', '') or '')
    lines.append(f"url: {row.get('url','')}")
    lines.append("-" * 100)

output_path.write_text("\n".join(lines), encoding='utf-8')
print(f"saved={output_path}")
print(f"rows={len(data)}")
