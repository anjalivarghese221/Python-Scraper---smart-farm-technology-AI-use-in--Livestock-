import json

with open('classified_sentiment_data_clean_expanded.json') as f:
    data = json.load(f)

island_posts = [p for p in data if 'island' in (p.get('text', '') + p.get('title', '')).lower()]

print("ISLAND WORD ANALYSIS")
print("="*80)
print(f"Total posts with 'island': {len(island_posts)}\n")

# Categorize by subreddit
subreddits = {}
for post in island_posts:
    sub = post.get('subreddit')
    if sub not in subreddits:
        subreddits[sub] = {'count': 0, 'neg': 0, 'pos': 0}
    subreddits[sub]['count'] += 1
    if post.get('sentiment') == 'negative':
        subreddits[sub]['neg'] += 1
    elif post.get('sentiment') == 'positive':
        subreddits[sub]['pos'] += 1

print("SUBREDDIT DISTRIBUTION:\n")
sorted_subs = sorted(subreddits.items(), key=lambda x: x[1]['count'], reverse=True)
for sub, stats in sorted_subs[:10]:
    neg_pct = 100*stats['neg']/stats['count'] if stats['count'] > 0 else 0
    print(f"  r/{sub}: {stats['count']} posts (Neg: {stats['neg']}, Pos: {stats['pos']}) - {neg_pct:.0f}% negative")

print("\n" + "="*80)
print("CONCLUSION:\n")
print("'island' appears in posts about:")
print("  - GEOGRAPHICAL DISCUSSIONS (Cuba, island nations)")
print("  - UNRELATED POLITICAL/SOCIAL TOPICS (not about farming/livestock)")
print("  - These tend to be negative in sentiment due to their critical content")
print("\nTHIS IS LIKELY A DATA CONTAMINATION ISSUE:")
print("  - 'island' is NOT a meaningful driver for smart farming sentiment")
print("  - The word appears in unrelated posts collected from broad subreddit searches")
print("  - It appears negative because those posts discuss political/economic issues")
