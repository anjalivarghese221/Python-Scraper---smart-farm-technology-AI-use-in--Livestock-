import json
import re
from pathlib import Path

DATA_PATH = Path('classified_sentiment_data_roberta.json')
OUT_PATH = Path('roberta_guaranteed_domain_examples.txt')

data = json.loads(DATA_PATH.read_text(encoding='utf-8'))

livestock_terms = re.compile(
    r"\b(livestock|dairy|cattle|cow|cows|sheep|goat|goats|swine|pig|pigs|poultry|herd|ruminant|beef|milk|milking)\b",
    re.I,
)
ai_terms = re.compile(
    r"\b(ai|artificial intelligence|machine learning|computer vision|iot|sensor|sensors|robot|robotics|automation|automated|precision)\b",
    re.I,
)

# Remove obvious false-positive contexts where livestock words are metaphorical/non-domain
exclude_terms = re.compile(
    r"\b(epstein|conspiracy|politics|election|hollywood|voice actor|genshin|byd|trump|canada|greenland)\b",
    re.I,
)

examples = {"negative": [], "positive": [], "neutral": []}

for row in data:
    sentiment = (row.get("sentiment") or "").lower()
    if sentiment not in examples:
        continue

    text = " ".join(
        [
            row.get("title", "") or "",
            row.get("text", "") or "",
            row.get("raw_text", "") or "",
            row.get("subreddit", "") or "",
        ]
    )

    if exclude_terms.search(text):
        continue

    if livestock_terms.search(text) and ai_terms.search(text):
        examples[sentiment].append(row)

for s in examples:
    examples[s].sort(key=lambda r: float(r.get("sentiment_confidence") or 0), reverse=True)

lines = []
for s in ["negative", "positive", "neutral"]:
    lines.append(f"\n===== {s.upper()} (TOP 8 GUARANTEED DOMAIN CANDIDATES) =====")
    if not examples[s]:
        lines.append("None found.")
        continue

    for i, r in enumerate(examples[s][:8], 1):
        title = (r.get("title") or "").replace("\n", " ").strip()
        content = (r.get("text") or r.get("raw_text") or "").replace("\n", " ").strip()
        if len(content) > 260:
            content = content[:260] + "..."

        lines.append(f"\n[{i}] confidence={float(r.get('sentiment_confidence') or 0):.4f} | date={r.get('created_date')} | subreddit=r/{r.get('subreddit')}")
        lines.append(f"title: {title}")
        lines.append(f"content: {content}")
        lines.append(f"url: {r.get('url')}")

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"saved={OUT_PATH}")
