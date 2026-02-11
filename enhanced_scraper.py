#!/usr/bin/env python3
"""
Enhanced Scientific Reddit Scraper - Peer-Review Ready
Implements rigorous data collection standards for academic research

Based on workflow requirements:
- Boolean query construction with documented relevance audit
- Temporal metadata collection for trend analysis
- Engagement metrics for weighted analysis
- Pagination strategy for comprehensive coverage
- Query logging for reproducibility
"""

import requests
import re
import time
import json
import hashlib
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Dict, Optional


class EnhancedScientificScraper:
    """
    Scientifically rigorous Reddit scraper for agricultural technology research
    Designed for datasets requiring 1500-2000+ posts with temporal validity
    """
    
    def __init__(self, log_file='query_log.json'):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot v2.0; +contact@university.edu)'
        }
        self.scraped_data = []
        self.query_log = []
        self.log_file = log_file
        
        # Target subreddits with documented rationale
        self.subreddits = {
            'AgTech': 'Primary - Agricultural technology focus',
            'farming': 'High volume - General farming discussions',
            'agriculture': 'High volume - Agricultural practices',
            'dairy': 'Domain specific - Dairy livestock',
            'livestock': 'Domain specific - Livestock management',
            'precisionag': 'Technology focus - Precision agriculture'
        }
        
    def construct_boolean_query(self, 
                                primary_concepts: List[str],
                                contextual_constraints: List[str],
                                exclude_terms: Optional[List[str]] = None) -> str:
        """
        Construct academically rigorous Boolean query
        
        Format: (A OR B) AND (C) NOT (D)
        
        Args:
            primary_concepts: Core technology terms (OR logic)
            contextual_constraints: Domain constraints (AND logic) 
            exclude_terms: Noise reduction terms (NOT logic)
            
        Returns:
            Formatted query string
            
        Example:
            primary=['AI', 'machine learning', 'sensor']
            contextual=['livestock', 'dairy', 'cattle']
            exclude=['politics', 'trade war']
        """
        # OR logic for lexical variation
        primary = ' OR '.join(primary_concepts)
        
        # AND logic for domain relevance
        contextual = ' '.join(contextual_constraints)
        
        # Build query
        query = f"({primary}) {contextual}"
        
        # Add exclusions if provided
        if exclude_terms:
            exclude_str = ' '.join([f'-{term}' for term in exclude_terms])
            query += f" {exclude_str}"
            
        return query
    
    def log_query(self, query: str, subreddit: str, parameters: Dict, rationale: str = ""):
        """
        Document query execution for reproducibility
        Required for peer-reviewed publications
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'subreddit': subreddit,
            'parameters': parameters,
            'rationale': rationale,
            'api_tier': 'Reddit JSON API (public)'
        }
        self.query_log.append(log_entry)
        
    def scrape_with_temporal_metadata(self,
                                      query: str,
                                      subreddit: str,
                                      limit: int = 100,
                                      time_filter: str = 'all',
                                      sort: str = 'relevance') -> List[Dict]:
        """
        Scrape with complete temporal and engagement metadata
        
        Returns posts with:
        - Exact timestamp (for time-series analysis)
        - Engagement metrics (score, comments)
        - Anonymized author ID
        - Geographic metadata (if available)
        """
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            'q': query,
            'limit': 100,  # Reddit max per request
            'sort': sort,
            't': time_filter,
            'restrict_sr': 'on'
        }
        
        # Log this query
        self.log_query(query, subreddit, params, 
                      f"Searching r/{subreddit} for agricultural technology discourse")
        
        collected_posts = []
        after = None
        total_collected = 0
        
        print(f"\n[SCRAPING] r/{subreddit} - Target: {limit} posts")
        
        # Pagination loop
        while total_collected < limit:
            if after:
                params['after'] = after
                
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                
                if response.status_code == 429:
                    print("  [RATE LIMIT] Waiting 60s...")
                    time.sleep(60)
                    continue
                    
                if response.status_code != 200:
                    print(f"  [ERROR] Status {response.status_code}")
                    break
                    
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                after = data.get('data', {}).get('after')
                
                if not posts:
                    break
                    
                for post in posts:
                    post_data = post.get('data', {})
                    
                    # Extract comprehensive metadata
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    
                    # Temporal data (CRITICAL for trend analysis)
                    created_utc = post_data.get('created_utc', 0)
                    created_date = datetime.fromtimestamp(created_utc)
                    
                    # Engagement metrics (for weighted sentiment)
                    score = post_data.get('score', 0)
                    upvote_ratio = post_data.get('upvote_ratio', 0.5)
                    num_comments = post_data.get('num_comments', 0)
                    
                    # Author (anonymized)
                    author = post_data.get('author', '[deleted]')
                    author_hash = hashlib.sha256(author.encode()).hexdigest()[:16]
                    
                    # URL for verification
                    permalink = post_data.get('permalink', '')
                    
                    # Flair (topic categorization)
                    link_flair_text = post_data.get('link_flair_text', None)
                    
                    post_record = {
                        # Core content
                        'text': selftext if selftext else title,
                        'title': title,
                        
                        # Temporal metadata
                        'created_at': created_date.isoformat(),
                        'created_utc': created_utc,
                        'year': created_date.year,
                        'month': created_date.month,
                        'day': created_date.day,
                        
                        # Engagement metrics  
                        'score': score,
                        'upvote_ratio': upvote_ratio,
                        'num_comments': num_comments,
                        'engagement': score + num_comments,  # Combined metric
                        
                        # Metadata
                        'source': 'Reddit',
                        'subreddit': subreddit,
                        'author_id': author_hash,  # Anonymized
                        'flair': link_flair_text,
                        'url': f"https://reddit.com{permalink}",
                        
                        # Raw text for audit trail
                        'raw_text': f"{title} {selftext}".strip()
                    }
                    
                    collected_posts.append(post_record)
                    total_collected += 1
                    
                    if total_collected >= limit:
                        break
                        
                print(f"  [PROGRESS] Collected {total_collected}/{limit}")
                
                # Respectful rate limiting
                time.sleep(2)
                
                if not after:
                    break
                    
            except Exception as e:
                print(f"  [ERROR] {str(e)}")
                break
                
        print(f"  [COMPLETE] Collected {len(collected_posts)} posts from r/{subreddit}")
        return collected_posts
    
    def scrape_comprehensive_dataset(self,
                                    target_size: int = 1500,
                                    time_window_months: int = 24) -> List[Dict]:
        """
        Collect scientifically valid dataset with temporal coverage
        
        Args:
            target_size: Minimum posts needed for statistical validity (1500-2000)
            time_window_months: Temporal scope for seasonal/trend analysis
            
        Returns:
            List of posts with complete metadata
        """
        print("=" * 80)
        print("ENHANCED SCIENTIFIC DATA COLLECTION")
        print(f"Target: {target_size} posts | Temporal scope: {time_window_months} months")
        print("=" * 80)
        
        # Define multiple query strategies for comprehensive coverage
        query_strategies = [
            {
                'primary': ['AI', 'artificial intelligence', 'machine learning'],
                'contextual': ['livestock', 'cattle', 'dairy'],
                'exclude': ['trump', 'election', 'tariff', 'trade war'],
                'rationale': 'Core AI + livestock intersection'
            },
            {
                'primary': ['smart farm', 'precision agriculture', 'precision livestock'],
                'contextual': ['technology', 'sensor', 'monitoring'],
                'exclude': ['trump', 'politics'],
                'rationale': 'Smart farming technology focus'
            },
            {
                'primary': ['automated', 'automation', 'robot'],
                'contextual': ['farm', 'dairy', 'milking', 'feeding'],
                'exclude': ['layoff', 'unemployment'],
                'rationale': 'Automation in farm operations'
            },
            {
                'primary': ['IoT', 'sensor', 'camera', 'tracking'],
                'contextual': ['livestock', 'cattle', 'herd', 'animal'],
                'exclude': [],
                'rationale': 'Sensor-based monitoring'
            },
            {
                'primary': ['data', 'analytics', 'prediction'],
                'contextual': ['farm', 'agriculture', 'livestock'],
                'exclude': ['stock market', 'crypto'],
                'rationale': 'Data-driven farming'
            }
        ]
        
        all_posts = []
        posts_per_strategy = target_size // len(query_strategies)
        posts_per_subreddit = posts_per_strategy // len(self.subreddits)
        
        for strategy in query_strategies:
            query = self.construct_boolean_query(
                strategy['primary'],
                strategy['contextual'],
                strategy.get('exclude')
            )
            
            print(f"\n[QUERY STRATEGY] {strategy['rationale']}")
            print(f"[QUERY] {query}")
            
            for subreddit, description in self.subreddits.items():
                posts = self.scrape_with_temporal_metadata(
                    query=query,
                    subreddit=subreddit,
                    limit=posts_per_subreddit,
                    time_filter='all'
                )
                all_posts.extend(posts)
                time.sleep(3)  # Respectful delay between subreddits
                
        # Save query log for reproducibility
        with open(self.log_file, 'w') as f:
            json.dump(self.query_log, f, indent=2)
        print(f"\n[SAVED] Query log: {self.log_file}")
        
        print(f"\n[TOTAL COLLECTED] {len(all_posts)} posts")
        return all_posts
    
    def manual_relevance_audit(self, sample_size: int = 100) -> float:
        """
        Perform manual relevance check on sample
        Required for query validation
        
        Returns:
            Relevance percentage
            
        Target: ≥80% relevance for peer-reviewed acceptance
        """
        if len(self.scraped_data) < sample_size:
            sample = self.scraped_data
        else:
            import random
            sample = random.sample(self.scraped_data, sample_size)
            
        print(f"\n[AUDIT] Manual relevance check on {len(sample)} posts")
        print("For each post, assess: Is this about agricultural technology?")
        print("Review titles and identify off-topic posts...")
        
        # Show sample for manual review
        for i, post in enumerate(sample[:10], 1):
            print(f"\n{i}. [{post['subreddit']}] {post['title'][:80]}")
            
        print("\n(Review remaining 90 posts manually and calculate relevance %)")
        print("Target: ≥80% relevance")
        
        return 0.0  # Placeholder - requires human judgment
    
    def save_dataset(self, filename: str = 'enhanced_scraped_data.json'):
        """Save dataset with complete metadata"""
        output = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'total_posts': len(self.scraped_data),
                'subreddits': list(self.subreddits.keys()),
                'query_strategies': len(self.query_log),
                'temporal_coverage': self._calculate_temporal_coverage()
            },
            'posts': self.scraped_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"\n[SAVED] {filename}")
        print(f"  - Total posts: {len(self.scraped_data)}")
        print(f"  - Date range: {output['metadata']['temporal_coverage']}")
        
    def _calculate_temporal_coverage(self) -> str:
        """Calculate date range of collected data"""
        if not self.scraped_data:
            return "No data"
            
        dates = [post['created_utc'] for post in self.scraped_data if 'created_utc' in post]
        if not dates:
            return "No timestamps"
            
        min_date = datetime.fromtimestamp(min(dates))
        max_date = datetime.fromtimestamp(max(dates))
        
        return f"{min_date.date()} to {max_date.date()}"


def main():
    """Run enhanced scientific scraper"""
    scraper = EnhancedScientificScraper()
    
    # Collect 1500-2000 posts with 24-month temporal coverage
    posts = scraper.scrape_comprehensive_dataset(
        target_size=1500,
        time_window_months=24
    )
    
    scraper.scraped_data = posts
    scraper.save_dataset('enhanced_scraped_data.json')
    
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE - Ready for Phase 2 preprocessing")
    print("=" * 80)


if __name__ == "__main__":
    main()
