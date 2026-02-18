#!/usr/bin/env python3
"""
Simple Reddit scraper using requests - NO COMPLEX BOOLEAN QUERIES
Works with Reddit JSON API directly - gets 2500+ posts
"""

import requests
import json
import time
from datetime import datetime

def scrape_reddit_search(query, limit=100):
    """Scrape Reddit search - simple and reliable"""
    url = f"https://www.reddit.com/search.json"
    params = {
        'q': query,
        'limit': limit,
        'sort': 'relevance',
        't': 'all'  # All time
    }
    
    headers = {'User-Agent': 'SmartFarmResearch/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            posts = []
            
            for child in data.get('data', {}).get('children', []):
                post_data = child.get('data', {})
                
                # Filter to 2018 onwards
                created_utc = post_data.get('created_utc', 0)
                post_date = datetime.fromtimestamp(created_utc)
                if post_date.year < 2018:
                    continue
                
                posts.append({
                    'source': 'reddit',
                    'subreddit': post_data.get('subreddit', ''),
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'raw_text': f"{post_data.get('title', '')} {post_data.get('selftext', '')}".strip(),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_date': post_date.strftime('%Y-%m-%d'),
                    'url': post_data.get('url', '')
                })
            
            return posts
        else:
            print(f"  HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"  Error: {e}")
        return []

def scrape_subreddit(subreddit, limit=100):
    """Scrape top posts from a subreddit"""
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {
        'limit': limit,
        't': 'all'
    }
    
    headers = {'User-Agent': 'SmartFarmResearch/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            posts = []
            
            for child in data.get('data', {}).get('children', []):
                post_data = child.get('data', {})
                
                # Filter to 2018 onwards
                created_utc = post_data.get('created_utc', 0)
                post_date = datetime.fromtimestamp(created_utc)
                if post_date.year < 2018:
                    continue
                
                posts.append({
                    'source': 'reddit',
                    'subreddit': post_data.get('subreddit', ''),
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'raw_text': f"{post_data.get('title', '')} {post_data.get('selftext', '')}".strip(),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_date': post_date.strftime('%Y-%m-%d'),
                    'url': post_data.get('url', '')
                })
            
            return posts
        else:
            return []
    except Exception as e:
        print(f"  Error: {e}")
        return []

def main():
    print("=" * 80)
    print("REDDIT COLLECTION - TARGET 2500+ POSTS (2018 ONWARDS)")
    print("=" * 80)
    
    all_posts = []
    seen_urls = set()
    
    # 35 SIMPLE KEYWORD QUERIES
    queries = [
        ("AI livestock", 100),
        ("smart farming", 100),
        ("precision agriculture", 100),
        ("farm automation", 100),
        ("agricultural technology", 100),
        ("dairy technology", 100),
        ("livestock monitoring", 100),
        ("farm sensors", 100),
        ("agricultural robotics", 100),
        ("automated milking", 100),
        ("cattle tracking", 100),
        ("farm data", 100),
        ("IoT agriculture", 100),
        ("computer vision farming", 100),
        ("machine learning agriculture", 100),
        ("wearable sensors animals", 80),
        ("drone livestock", 80),
        ("smart collar cattle", 80),
        ("barn automation", 80),
        ("agricultural AI", 80),
        ("farm management software", 80),
        ("livestock health sensors", 80),
        ("predictive farming", 80),
        ("automated feeding", 80),
        ("precision dairy", 80),
        ("farm tech", 80),
        ("agricultural sensors", 80),
        ("livestock technology", 80),
        ("smart agriculture", 80),
        ("dairy automation", 80),
        ("cattle monitoring", 60),
        ("farm innovation", 60),
        ("precision livestock", 60),
        ("agtech", 60),
        ("digital farming", 60)
    ]
    
    print(f"\nPhase 1: Searching with {len(queries)} queries")
    print("=" * 80)
    
    for i, (query, limit) in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] '{query}' (limit: {limit})")
        print(f"  Current total: {len(all_posts)} posts")
        
        posts = scrape_reddit_search(query, limit)
        
        # Deduplicate
        new_posts = 0
        for post in posts:
            if post['url'] not in seen_urls:
                seen_urls.add(post['url'])
                all_posts.append(post)
                new_posts += 1
        
        print(f"  ✓ Added {new_posts} new posts | Total: {len(all_posts)}")
        print(f"  ⏳ Waiting 8 seconds...")
        time.sleep(8)
    
    print(f"\n\nPhase 1 complete: {len(all_posts)} posts")
    print("\n" + "=" * 80)
    print("Phase 2: Subreddit scraping")
    print("=" * 80)
    
    # SUBREDDIT COLLECTION
    subreddits = [
        ('AgTech', 100),
        ('farming', 100),
        ('agriculture', 100),
        ('dairy', 80),
        ('livestock', 80),
        ('Homesteading', 60),
        ('homestead', 60),
        ('technology', 50),
        ('startups', 40),
        ('smallbusiness', 40),
        ('askscience', 30),
        ('MachineLearning', 30),
        ('datascience', 30),
        ('IoT', 30),
        ('robotics', 30)
    ]
    
    for i, (sub, limit) in enumerate(subreddits, 1):
        print(f"\n[{i}/{len(subreddits)}] r/{sub} (limit: {limit})")
        print(f"  Current total: {len(all_posts)} posts")
        
        posts = scrape_subreddit(sub, limit)
        
        new_posts = 0
        for post in posts:
            if post['url'] not in seen_urls:
                seen_urls.add(post['url'])
                all_posts.append(post)
                new_posts += 1
        
        print(f"  ✓ Added {new_posts} new posts | Total: {len(all_posts)}")
        print(f"  ⏳ Waiting 10 seconds...")
        time.sleep(10)
    
    # SAVE
    print("\n\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total posts: {len(all_posts)}")
    print(f"Target: 2500")
    
    if len(all_posts) >= 2500:
        print(f"✓✓✓ TARGET REACHED!")
    else:
        print(f"Progress: {len(all_posts)/2500*100:.1f}%")
    
    output = {
        'posts': all_posts,
        'metadata': {
            'total': len(all_posts),
            'collection_date': '2026-02-18',
            'no_political_filters': True
        }
    }
    
    with open('enhanced_scraped_data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved to enhanced_scraped_data.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
