#!/usr/bin/env python3
"""
Run original working scraper with multiple queries to get 1000+ posts
Then add temporal metadata for analysis
"""

import json
import time
from datetime import datetime
from scraper import SmartFarmScraper

def enhance_with_temporal_data(posts):
    """Add temporal metadata to posts from original scraper"""
    enhanced = []
    for post in posts:
        # Parse created_date if exists
        if 'created_date' in post:
            try:
                dt = datetime.strptime(post['created_date'], '%Y-%m-%d')
                post['created_at'] = dt.isoformat()
                post['year'] = dt.year
                post['month'] = dt.month
                post['quarter'] = (dt.month - 1) // 3 + 1
                post['year_month'] = f"{dt.year}-{dt.month:02d}"
                post['year_quarter'] = f"{dt.year}-Q{post['quarter']}"
            except:
                pass
        
        # Ensure raw_text exists
        if 'raw_text' not in post:
            post['raw_text'] = f"{post.get('title', '')} {post.get('text', '')}".strip()
        
        enhanced.append(post)
    
    return enhanced

def main():
    print("="*80)
    print("ENHANCED DATA COLLECTION (Using Working Scraper)")
    print("="*80)
    
    scraper = SmartFarmScraper()
    
    # Multiple queries - OPTIMIZED FOR MAXIMUM COLLECTION
    queries = [
        # Core broad queries (MAXED AT 100)
        ("AI livestock monitoring", 100),
        ("smart farming sensors cattle dairy", 100),
        ("precision livestock farming technology", 100),
        ("IoT agriculture cattle", 100),
        ("automated milking robot", 100),
        ("farm automation AI", 100),
        ("machine learning agriculture", 100),
        ("dairy technology monitoring", 100),
        ("cattle tracking sensors", 100),
        ("precision agriculture", 100),
        
        # Specific applications (80-100)
        ("robotic dairy milking", 100),
        ("automated feeding system", 100),
        ("smart collar cattle", 100),
        ("farm management software", 100),
        ("agricultural innovation", 100),
        ("wearable sensors animals", 80),
        ("automated barn management", 80),
        ("drone livestock", 80),
        ("computer vision dairy", 80),
        ("predictive analytics farming", 80),
        ("farm data analytics", 80),
        ("livestock health sensors", 80),
        ("precision dairy", 80),
        ("smart agriculture", 100),
        ("agricultural robotics", 80)
    ]
    
    total_queries = len(queries)
    print(f"\nCollecting data with {total_queries} query strategies...")
    print(f"Target: 2500+ posts\n")
    print("=" * 80)
    
    for i, (query, limit) in enumerate(queries, 1):
        current_count = len(scraper.get_data())
        progress_pct = (i / total_queries) * 100
        print(f"\n[QUERY {i}/{total_queries}] ({progress_pct:.1f}% complete)")
        print(f"Query: '{query}' | Requesting: {limit} posts")
        print(f"Posts collected so far: {current_count}")
        print("-" * 80)
        
        scraper.scrape_reddit_search(query, limit=limit)
        
        new_count = len(scraper.get_data())
        added = new_count - current_count
        print(f"✓ Added {added} relevant posts | Total now: {new_count}")
        print("⏳ Waiting 8 seconds to avoid rate limiting...")
        time.sleep(8)  # Increased delay to avoid rate limits
    
    # Also scrape from subreddits - EXPANDED
    print("\n\n" + "=" * 80)
    print("SUBREDDIT COLLECTION")
    print("=" * 80)
    subreddits_to_scrape = [
        ('AgTech', 100),
        ('farming', 100), 
        ('agriculture', 100),
        ('dairy', 80),
        ('livestock', 60),
        ('Homesteading', 60),
        ('homestead', 60),
        # ADDITIONAL SUBREDDITS
        ('technology', 50),  # General tech discussions
        ('startups', 40),    # Agtech startups
        ('smallbusiness', 40),  # Farm business tech
        ('askscience', 30),  # Scientific discussions
        ('MachineLearning', 30),  # AI/ML applications
        ('datascience', 30),  # Data-driven farming
        ('IoT', 30),  # IoT sensors
        ('robotics', 30)  # Agricultural robotics
    ]
    
    for i, (sub, limit) in enumerate(subreddits_to_scrape, 1):
        current_count = len(scraper.get_data())
        print(f"\n[SUBREDDIT {i}/{len(subreddits_to_scrape)}] r/{sub}")
        print(f"Posts collected so far: {current_count}")
        try:
            scraper.scrape_top_posts_from_subreddit(sub, limit=limit, time_filter='all')
            new_count = len(scraper.get_data())
            added = new_count - current_count
            print(f"✓ Added {added} posts from r/{sub} | Total: {new_count}")
            print("⏳ Waiting 10 seconds to avoid rate limiting...")
            time.sleep(10)
        except Exception as e:
            print(f"✗ Error with r/{sub}: {e}")
    
    posts = scraper.get_data()
    
    print(f"\n" + "=" * 80)
    print(f"COLLECTION COMPLETE ✓")
    print("=" * 80)
    print(f"Total posts collected: {len(posts)}")
    print(f"Target was: 2500 posts")
    if len(posts) >= 2500:
        print(f"✓ TARGET REACHED!")
    else:
        print(f"⚠ Collected {len(posts)}/{2500} ({(len(posts)/2500)*100:.1f}%)")
    
    print(f"\nAdding temporal metadata...")
    # Add temporal metadata
    enhanced_posts = enhance_with_temporal_data(posts)
    print(f"✓ Temporal metadata added to {len(enhanced_posts)} posts")
    
    # Save
    output = {
        'metadata': {
            'collection_date': datetime.now().isoformat(),
            'total_posts': len(enhanced_posts),
            'queries_used': len(queries)
        },
        'posts': enhanced_posts
    }
    
    with open('enhanced_scraped_data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: enhanced_scraped_data.json")
    print(f"Ready for preprocessing!")

if __name__ == "__main__":
    main()
