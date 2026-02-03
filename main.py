#!/usr/bin/env python3
"""
Main Runner Script
Orchestrates scraping, cleaning, and displaying data about smart farm technology and AI in livestock
"""

import json
import sys
from scraper import SmartFarmScraper
from text_cleaner import TextCleaner
from data_display import DataDisplay


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("SMART FARM TECHNOLOGY & AI IN LIVESTOCK ANALYSIS")
    print("   Social Media Opinion Scraper & Text Analyzer")
    print("=" * 80)
    
    scraper = SmartFarmScraper()
    cleaner = TextCleaner()
    display = DataDisplay()
    
    # ========== PHASE 1: SCRAPING ==========
    print("\n" + "=" * 80)
    print("[PHASE 1] SCRAPING SOCIAL MEDIA (REDDIT)")
    print("=" * 80)
    
    try:
        # Search for specific technology + livestock topics
        print("\n[SEARCH] Searching for TECHNOLOGY & AI opinions in farming...")
        scraper.scrape_reddit_search("AI livestock monitoring", limit=25)
        scraper.scrape_reddit_search("smart farming sensors cattle dairy", limit=25)
        scraper.scrape_reddit_search("automated milking system robot", limit=20)
        scraper.scrape_reddit_search("precision livestock farming technology", limit=20)
        scraper.scrape_reddit_search("farm automation AI artificial intelligence", limit=20)
        
        # Get technology posts from relevant subreddits
        print("\n[INFO] Getting technology-focused posts from farming subreddits...")
        target_subreddits = ['AgTech', 'farming', 'agriculture', 'dairy']
        for sub in target_subreddits:
            try:
                scraper.scrape_top_posts_from_subreddit(sub, limit=30, time_filter='all')
            except Exception as e:
                print(f"  [WARNING] Error with r/{sub}: {str(e)}")
        
        raw_data = scraper.get_data()
        
        print("\n" + "=" * 80)
        print(f"[SUCCESS] Scraping complete! Collected {len(raw_data)} items")
        print("=" * 80)
        
        if len(raw_data) == 0:
            print("\n[WARNING] No data was scraped. This could be due to:")
            print("   • Reddit blocking the requests")
            print("   • No relevant content found")
            print("   • Network connectivity issues")
            print("\n[TIP] Check raw_scraped_data.json if it exists")
            return
        
        # Save raw data
        with open('raw_scraped_data.json', 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        print("[SAVED] Raw data saved to: raw_scraped_data.json")
        
        # Display sample
        display.display_raw_data(raw_data, max_items=5)
        
    except Exception as e:
        print(f"\n[ERROR] Error during scraping: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Try to load existing data
        try:
            print("\n[RETRY] Attempting to load existing raw_scraped_data.json...")
            with open('raw_scraped_data.json', 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            print(f"[SUCCESS] Loaded {len(raw_data)} items from existing file")
        except:
            print("[ERROR] Could not load existing data. Exiting.")
            return
    
    # ========== PHASE 2: CLEANING ==========
    print("\n" + "=" * 80)
    print("[PHASE 2] CLEANING & PROCESSING TEXT")
    print("=" * 80)
    
    try:
        cleaned_data = cleaner.clean_all(raw_data)
        
        stats = cleaner.get_statistics()
        
        cleaner.save_cleaned_data('cleaned_data.json')

        with open('statistics.json', 'w', encoding='utf-8') as f:
            stats_json = stats.copy()
            stats_json['top_words'] = stats['top_words']
            stats_json['top_bigrams'] = stats['top_bigrams']
            json.dump(stats_json, f, indent=2, ensure_ascii=False)
        print("[SAVED] Statistics saved to: statistics.json")
        
    except Exception as e:
        print(f"\n[ERROR] Error during cleaning: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== PHASE 3: DISPLAY & ANALYSIS ==========
    print("\n" + "=" * 80)
    print("[PHASE 3] ANALYSIS & VISUALIZATION")
    print("=" * 80)
    
    try:
        display.display_cleaning_stats(stats)
        
        display.display_cleaned_sample(cleaned_data, max_items=3)
        
        display.display_topic_keywords(stats)
        
        display.create_summary_report(raw_data, cleaned_data, stats, 'summary_report.txt')
        
    except Exception as e:
        print(f"\n[ERROR] Error during display: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 80)
    print("[SUCCESS] ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n[FILES] Generated Files:")
    print("   1. raw_scraped_data.json    - Original scraped data")
    print("   2. cleaned_data.json        - Cleaned and processed text")
    print("   3. statistics.json          - Word frequencies and analysis")
    print("   4. summary_report.txt       - Human-readable summary")
    print("\n[TIPS] Next Steps:")
    print("   • Review the summary_report.txt for key insights")
    print("   • Analyze top words and bigrams for topic trends")
    print("   • Use cleaned_data.json for further sentiment analysis")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Process interrupted by user. Partial data may have been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
