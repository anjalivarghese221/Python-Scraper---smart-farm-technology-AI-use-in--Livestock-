import json

TARGET_SUBREDDITS = {'farming', 'Agriculture', 'livestock', 'dairyfarming'}

with open('classified_sentiment_data_domain_smart_farming_livestock.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

candidates = [
    x for x in data
    if x.get('sentiment') == 'negative'
    and x.get('subreddit') in TARGET_SUBREDDITS
]

# Prefer richer text bodies
candidates.sort(key=lambda x: len((x.get('text') or x.get('raw_text') or '').strip()), reverse=True)

selected = []
seen_titles = set()
for item in candidates:
    title = (item.get('title') or '').strip()
    if not title or title in seen_titles:
        continue
    seen_titles.add(title)
    selected.append(item)
    if len(selected) == 5:
        break

print(f"Found {len(candidates)} negative candidates in target subreddits")
print(f"Returning {len(selected)} examples\n")

for i, item in enumerate(selected, 1):
    content = (item.get('text') or item.get('raw_text') or '').strip().replace('\n', ' ')
    print(f"{i}. Title: {item.get('title', 'N/A')}")
    print(f"   Subreddit: r/{item.get('subreddit', 'N/A')}")
    print(f"   Date: {item.get('created_date', 'N/A')}")
    print(f"   Sentiment: {item.get('sentiment')} | Confidence: {float(item.get('sentiment_confidence', 0)):.4f}")
    print(f"   Content: {content[:280]}{'...' if len(content) > 280 else ''}")
    print()
