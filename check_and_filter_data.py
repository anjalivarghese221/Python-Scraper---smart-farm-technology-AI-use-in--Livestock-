#!/usr/bin/env python3
"""Check collected data and filter to 2018 onwards"""
import json
from datetime import datetime

# Load data
with open('enhanced_scraped_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total posts collected: {len(data)}")

# Analyze dates
dates = []
posts_with_dates = 0
for post in data:
    if post.get('created_date'):
        try:
            date_str = post['created_date']
            # Parse date (format: YYYY-MM-DD)
            date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')
            dates.append((date_obj, date_str))
            posts_with_dates += 1
        except:
            pass

print(f"Posts with valid dates: {posts_with_dates}")
if dates:
    dates.sort()
    print(f"Date range: {dates[0][1]} to {dates[-1][1]}")

# Filter to 2018 onwards
cutoff = datetime(2018, 1, 1)
filtered_data = []
for post in data:
    if post.get('created_date'):
        try:
            date_str = post['created_date']
            date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')
            if date_obj >= cutoff:
                filtered_data.append(post)
        except:
            pass

print(f"\n--- FILTERING TO 2018 ONWARDS ---")
print(f"Posts from 2018+: {len(filtered_data)}")
print(f"Posts removed (pre-2018): {len(data) - len(filtered_data)}")

# Save filtered data
with open('enhanced_scraped_data.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(filtered_data)} posts from 2018 onwards to enhanced_scraped_data.json")
