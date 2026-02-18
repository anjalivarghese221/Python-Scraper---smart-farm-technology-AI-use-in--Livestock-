#!/usr/bin/env python3
"""
Collect 2500+ posts from Reddit for smart farming/livestock AI analysis
No political filters - includes all relevant discourse
"""

import json
from enhanced_scraper import EnhancedScientificScraper

def main():
    print("=" * 80)
    print("COLLECTING 2500+ REDDIT POSTS")
    print("Smart Farming + AI + Livestock Analysis")
    print("=" * 80)
    
    scraper = EnhancedScientificScraper()
    
    # Use the built-in comprehensive collection method
    # This will handle all queries, subreddits, and rate limiting
    posts = scraper.scrape_comprehensive_dataset(
        target_size=2500,
        time_window_months=180  # 15 years (2011-2026)
    )
    
    print(f"\n\n" + "=" * 80)
    print(f"COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total posts collected: {len(posts)}")
    print(f"Target: 2500 posts")
    
    if len(posts) >= 2500:
        print(f"✓✓✓ TARGET REACHED! ({len(posts)}/2500)")
    else:
        percentage = (len(posts) / 2500) * 100
        print(f"Progress: {percentage:.1f}% ({len(posts)}/2500)")
        print(f"Need {2500 - len(posts)} more posts")
    
    # Save with timestamp
    output = {
        'posts': posts,
        'metadata': {
            'total_posts': len(posts),
            'collection_date': '2026-02-18',
            'political_filters': 'removed',
            'time_range': '2011-2026'
        }
    }
    
    with open('enhanced_scraped_data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Data saved to: enhanced_scraped_data.json")
    print(f"✓ Total posts: {len(posts)}")
    print(f"✓ Ready for preprocessing!")
    print("=" * 80)

if __name__ == "__main__":
    main()
