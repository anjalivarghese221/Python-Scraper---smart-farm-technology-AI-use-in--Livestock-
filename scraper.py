#!/usr/bin/env python3
"""
Smart Farm Technology & AI in Livestock Scraper
Scrapes social media opinions (Reddit) about smart farming and AI use in livestock management
"""

import requests
import re
import time
from collections import Counter
import json


class SmartFarmScraper:
    """Scraper for smart farm technology and AI livestock social media opinions"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (Python Scraper for Academic Research)'
        }
        self.scraped_data = []
        self.subreddits = [
            'AgTech',           # Agricultural technology (most relevant)
            'farming',
            'agriculture', 
            'dairy',
            'livestock',
            'smartfarming',     # If it exists
            'precisionag'       # If it exists
        ]
    
    def scrape_reddit_search(self, query, limit=50):
        """
        Scrape Reddit search results for the given query across multiple subreddits
        Uses Reddit's JSON API (no authentication needed for public posts)
        """
        print(f"\n[SEARCH] Searching Reddit for: '{query}'")
        
        # More specific technology-focused keywords
        keywords = ['ai', 'smart', 'technology', 'sensor', 'automation', 'robot', 'machine learning', 
                   'precision', 'monitor', 'tracking', 'iot', 'drone', 'camera', 'software', 'app',
                   'device', 'system', 'digital', 'data']
        
        # Exclude political/economic posts
        exclude_keywords = ['trump', 'tariff', 'trade war', 'election', 'vote', 'biden', 'china trade']
        
        # Search across multiple relevant subreddits
        for subreddit in self.subreddits:
            try:
                # Reddit JSON API endpoint
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    'q': query,
                    'limit': limit,
                    'sort': 'relevance',
                    't': 'all',
                    'restrict_sr': 'on'  # Restrict to this subreddit
                }
                
                print(f"\n  [CHECKING] r/{subreddit}...")
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    found_relevant = 0
                    for post in posts:
                        post_data = post.get('data', {})
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        score = post_data.get('score', 0)
                        num_comments = post_data.get('num_comments', 0)
                        author = post_data.get('author', '[deleted]')
                        created = post_data.get('created_utc', 0)
                        permalink = post_data.get('permalink', '')
                        
                        # Check if post is relevant to TECHNOLOGY (not just farming)
                        combined_text = f"{title} {selftext}".lower()
                        
                        # Must contain at least one technology keyword
                        has_tech_keyword = any(keyword in combined_text for keyword in keywords)
                        
                        # Should NOT be primarily about politics/economics
                        has_exclude_keyword = any(keyword in combined_text for keyword in exclude_keywords)
                        
                        # Focus on technology opinions
                        if has_tech_keyword and not has_exclude_keyword:
                            found_relevant += 1
                            
                            self.scraped_data.append({
                                'source': 'Reddit',
                                'subreddit': subreddit,
                                'type': 'post',
                                'title': title,
                                'text': selftext if selftext else title,
                                'author': author,
                                'score': score,
                                'num_comments': num_comments,
                                'url': f"https://reddit.com{permalink}",
                                'created_date': time.strftime('%Y-%m-%d', time.localtime(created)),
                                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                            })
                    
                    if found_relevant > 0:
                        print(f"    [OK] Found {found_relevant} relevant posts")
                    else:
                        print(f"    [WARNING] No relevant posts found")
                        
                elif response.status_code == 429:
                    print(f"    [WARNING] Rate limited, waiting...")
                    time.sleep(5)
                    continue
                else:
                    print(f"    [ERROR] Status code {response.status_code}")
                
                time.sleep(2)  # Be respectful with API requests
                
            except Exception as e:
                print(f"    [ERROR] Error scraping r/{subreddit}: {str(e)}")
                continue
        
        return len(self.scraped_data)
    
    def scrape_reddit_comments(self, post_url, max_comments=20):
        """
        Scrape comments from a specific Reddit post to get user opinions
        """
        try:
            # Convert URL to JSON endpoint
            json_url = post_url.rstrip('/') + '.json'
            response = requests.get(json_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    comments = data[1].get('data', {}).get('children', [])
                    
                    for comment in comments[:max_comments]:
                        comment_data = comment.get('data', {})
                        if comment_data.get('body') and comment_data.get('body') != '[deleted]':
                            body = comment_data.get('body', '')
                            author = comment_data.get('author', '[deleted]')
                            score = comment_data.get('score', 0)
                            
                            self.scraped_data.append({
                                'source': 'Reddit',
                                'type': 'comment',
                                'text': body,
                                'author': author,
                                'score': score,
                                'post_url': post_url,
                                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                            })
                    
                    return len(comments)
            return 0
            
        except Exception as e:
            print(f"  [ERROR] Error scraping comments: {str(e)}")
            return 0
    
    def scrape_top_posts_from_subreddit(self, subreddit, limit=25, time_filter='month'):
        """
        Scrape top posts from a specific subreddit
        time_filter: hour, day, week, month, year, all
        """
        print(f"\n[INFO] Getting top posts from r/{subreddit} (past {time_filter})...")
        
        try:
            url = f"https://www.reddit.com/r/{subreddit}/top.json"
            params = {
                'limit': limit,
                't': time_filter
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                
                # Focus on TECHNOLOGY keywords
                keywords = ['ai', 'smart', 'technology', 'sensor', 'automation', 'robot', 'machine learning', 
                           'precision', 'monitor', 'tracking', 'iot', 'drone', 'camera', 'software', 'app',
                           'device', 'system', 'digital', 'data', 'livestock', 'cattle', 'dairy']
                
                # Exclude political/economic topics
                exclude_keywords = ['trump', 'tariff', 'trade war', 'election', 'vote', 'biden', 'china']
                
                found = 0
                
                for post in posts:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    
                    # Check relevance - must be about TECHNOLOGY
                    combined_text = f"{title} {selftext}".lower()
                    has_tech = any(keyword in combined_text for keyword in keywords)
                    has_politics = any(keyword in combined_text for keyword in exclude_keywords)
                    
                    # Only include technology-focused posts
                    if has_tech and not has_politics:
                        found += 1
                        score = post_data.get('score', 0)
                        num_comments = post_data.get('num_comments', 0)
                        author = post_data.get('author', '[deleted]')
                        permalink = post_data.get('permalink', '')
                        
                        self.scraped_data.append({
                            'source': 'Reddit',
                            'subreddit': subreddit,
                            'type': 'post',
                            'title': title,
                            'text': selftext if selftext else title,
                            'author': author,
                            'score': score,
                            'num_comments': num_comments,
                            'url': f"https://reddit.com{permalink}",
                            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        print(f"  [OK] '{title[:60]}...' ({score} upvotes, {num_comments} comments)")
                
                print(f"  [SUCCESS] Found {found} relevant posts from r/{subreddit}")
                time.sleep(2)
                return found
                
            else:
                print(f"  [ERROR] Status code {response.status_code}")
                return 0
                
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            return 0
    
    def get_data(self):
        """Return all scraped data"""
        return self.scraped_data


if __name__ == "__main__":
    print("=" * 80)
    print("SMART FARM TECHNOLOGY & AI IN LIVESTOCK - SOCIAL MEDIA SCRAPER")
    print("=" * 80)
    
    scraper = SmartFarmScraper()
    
    # Method 1: Search Reddit for specific terms
    print("\n[METHOD 1] Searching Reddit posts about smart farming & AI livestock")
    print("-" * 80)
    scraper.scrape_reddit_search("smart farm AI livestock", limit=20)
    scraper.scrape_reddit_search("artificial intelligence cattle dairy", limit=20)
    scraper.scrape_reddit_search("precision agriculture technology", limit=15)
    
    # Method 2: Get top posts from relevant subreddits
    print("\n" + "=" * 80)
    print("[METHOD 2] Getting top posts from farming/agriculture subreddits")
    print("-" * 80)
    
    target_subreddits = ['farming', 'agriculture', 'dairy']
    for sub in target_subreddits:
        try:
            scraper.scrape_top_posts_from_subreddit(sub, limit=30, time_filter='year')
        except Exception as e:
            print(f"  [WARNING] Skipping r/{sub}: {str(e)}")
    
    print("\n" + "=" * 80)
    print(f"[SUCCESS] SCRAPING COMPLETE!")
    print(f"[INFO] Total items scraped: {len(scraper.get_data())}")
    
    # Count by type
    posts = sum(1 for item in scraper.get_data() if item.get('type') == 'post')
    comments = sum(1 for item in scraper.get_data() if item.get('type') == 'comment')
    print(f"   - Posts: {posts}")
    print(f"   - Comments: {comments}")
    
    # Save raw data to file
    if scraper.get_data():
        with open('raw_scraped_data.json', 'w', encoding='utf-8') as f:
            json.dump(scraper.get_data(), f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Raw data saved to raw_scraped_data.json")
        print("=" * 80)
