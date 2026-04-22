import json
import textwrap
from pathlib import Path

DATA_PATH = Path("classified_sentiment_data_clean_high_coverage.json")

data = json.loads(DATA_PATH.read_text())
rows = []
for r in data:
    title = (r.get("title") or "").strip()
    text = (r.get("text") or r.get("raw_text") or "").strip()
    conf = r.get("sentiment_confidence")
    if conf is None:
        continue
    if not title and not text:
        continue
    rows.append(r)

rows_sorted = sorted(rows, key=lambda x: (float(x.get("sentiment_confidence", 1.0)), x.get("subreddit", "")))

print("MANUAL_REVIEW_CANDIDATES=12")
for i, r in enumerate(rows_sorted[:12], 1):
    title = (r.get("title") or "").replace("\n", " ").strip()
    text = (r.get("text") or r.get("raw_text") or "").replace("\n", " ").strip()
    snippet = textwrap.shorten(text, width=240, placeholder="...")
    print(f"\n[{i}] subreddit: r/{r.get('subreddit', 'unknown')} | sentiment: {r.get('sentiment')} | confidence: {r.get('sentiment_confidence'):.3f}")
    print(f"date: {r.get('created_date', 'N/A')} | type: {r.get('item_type', 'N/A')} | source: {r.get('source', 'N/A')}")
    print(f"title: {title if title else '[no title]'}")
    print(f"content: {snippet if snippet else '[no content]'}")
    print(f"url: {r.get('url', 'N/A')}")
