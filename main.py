#!/usr/bin/env python3
"""
Main Runner Script
Orchestrates scraping, cleaning, and displaying data about smart farm technology and AI in livestock
"""

import json
from scraper import SmartFarmScraper
from text_cleaner import TextCleaner


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("SMART FARM TECHNOLOGY & AI IN LIVESTOCK - STEP 1")
    print("=" * 80)
    
    scraper = SmartFarmScraper()
    cleaner = TextCleaner()
    
    # PHASE 1: SCRAPING
    print("\n[PHASE 1] Scraping Reddit...")
    
    try:
        scraper.scrape_reddit_search("AI livestock monitoring", limit=25)
        scraper.scrape_reddit_search("smart farming sensors cattle dairy", limit=25)
        scraper.scrape_reddit_search("automated milking system robot", limit=20)
        scraper.scrape_reddit_search("precision livestock farming technology", limit=20)
        scraper.scrape_reddit_search("farm automation AI artificial intelligence", limit=20)
        
        for sub in ['AgTech', 'farming', 'agriculture', 'dairy']:
            try:
                scraper.scrape_top_posts_from_subreddit(sub, limit=30, time_filter='all')
            except Exception as e:
                print(f"  Warning: r/{sub}: {str(e)}")
        
        raw_data = scraper.get_data()
        print(f"\nCollected {len(raw_data)} items")
        
        if len(raw_data) == 0:
            print("No data scraped. Check network or try again.")
            return
        
        with open('raw_scraped_data.json', 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        print("Saved: raw_scraped_data.json")
        
    except Exception as e:
        print(f"\nError during scraping: {str(e)}")
        try:
            with open('raw_scraped_data.json', 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            print(f"Loaded {len(raw_data)} items from existing file")
        except:
            print("Could not load existing data. Exiting.")
            return
    
    # PHASE 2: CLEANING
    print("\n[PHASE 2] Cleaning text...")
    
    try:
        cleaned_data = cleaner.clean_all(raw_data)
        stats = cleaner.get_statistics()
        
        cleaner.save_cleaned_data('cleaned_data.json')
        
        with open('statistics.json', 'w', encoding='utf-8') as f:
            json.dump({
                'total_posts': stats['total_posts'],
                'unique_posts': stats['unique_posts'],
                'total_words': stats['total_words'],
                'unique_words': stats['unique_words'],
                'top_words': stats['top_words'],
                'top_bigrams': stats['top_bigrams']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Cleaned {stats['unique_posts']} unique posts")
        print(f"Total words: {stats['total_words']}, Unique: {stats['unique_words']}")
        print("\nTop 10 keywords:")
        for word, count in stats['top_words'][:10]:
            print(f"  {word}: {count}")
        
        print("\n" + "=" * 80)
        print("COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during cleaning: {str(e)}")
        return


if __name__ == "__main__":
    main()
