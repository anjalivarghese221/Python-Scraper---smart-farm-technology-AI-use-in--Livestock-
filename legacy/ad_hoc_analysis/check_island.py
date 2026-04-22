import json

with open('classified_sentiment_data_clean_expanded.json') as f:
    data = json.load(f)

island_posts = [p for p in data if 'island' in (p.get('text', '') + p.get('title', '')).lower()]
print(f'Total island posts: {len(island_posts)}\n')

neg = [p for p in island_posts if p.get('sentiment') == 'negative']
pos = [p for p in island_posts if p.get('sentiment') == 'positive']
neu = [p for p in island_posts if p.get('sentiment') == 'neutral']

print(f'Negative: {len(neg)} ({100*len(neg)/len(island_posts):.1f}%)')
print(f'Positive: {len(pos)} ({100*len(pos)/len(island_posts):.1f}%)')
print(f'Neutral: {len(neu)} ({100*len(neu)/len(island_posts):.1f}%)')

print("\n" + "="*80)
print("SAMPLE NEGATIVE ISLAND POSTS:\n")

for i, post in enumerate(neg[:2]):
    print(f"{i+1}. Title: {post.get('title')}")
    text = post.get('text', post.get('raw_text', ''))
    print(f"   Content: {text[:300]}")
    print(f"   Subreddit: r/{post.get('subreddit')}")
    print()
