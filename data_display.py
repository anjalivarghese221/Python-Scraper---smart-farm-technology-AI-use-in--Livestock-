#!/usr/bin/env python3
"""
Data Display & Visualization Module
Pretty printing and visualization for scraped and cleaned data
"""

import json
from collections import Counter
from datetime import datetime


class DataDisplay:
    """Display scraped and cleaned data in a user-friendly format"""
    
    def __init__(self):
        self.colors = {
            'HEADER': '\033[95m',
            'BLUE': '\033[94m',
            'CYAN': '\033[96m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'END': '\033[0m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m'
        }
    
    def color_text(self, text, color):
        """Add color to text"""
        return f"{self.colors.get(color, '')}{text}{self.colors['END']}"
    
    def print_header(self, title):
        """Print a formatted header"""
        width = 80
        print("\n" + "=" * width)
        print(self.color_text(title.center(width), 'BOLD'))
        print("=" * width)
    
    def print_section(self, title):
        """Print a section title"""
        print("\n" + self.color_text(f"[DATA] {title}", 'CYAN'))
        print("-" * 80)
    
    def display_raw_data(self, data, max_items=5):
        """Display raw scraped data"""
        self.print_header("RAW SCRAPED DATA SAMPLE")
        
        print(f"\nShowing {min(max_items, len(data))} of {len(data)} total items\n")
        
        for idx, item in enumerate(data[:max_items], 1):
            print(self.color_text(f"[{idx}] ", 'YELLOW'), end='')
            
            # Display based on type/source
            if item.get('source') == 'Reddit':
                if item.get('type') == 'post':
                    print(self.color_text(f"r/{item.get('subreddit', 'unknown')}", 'BLUE'), end=' ')
                    print(f"| {item.get('score', 0)} upvotes | {item.get('num_comments', 0)} comments")
                    print(f"    Title: {self.color_text(item.get('title', 'N/A')[:70], 'BOLD')}...")
                    
                    text = item.get('text', '')
                    if text and text != item.get('title'):
                        preview = text[:150].replace('\n', ' ')
                        print(f"    Text: {preview}...")
                elif item.get('type') == 'comment':
                    print(self.color_text("Comment", 'GREEN'), end=' ')
                    print(f"| {item.get('score', 0)} upvotes")
                    text = item.get('text', '')[:150].replace('\n', ' ')
                    print(f"    {text}...")
            else:
                print(f"{item.get('source', 'Unknown')}")
                print(f"    {item.get('title', 'N/A')[:70]}...")
            
            print()
    
    def display_cleaning_stats(self, stats):
        """Display statistics from the cleaning process"""
        self.print_header("TEXT CLEANING STATISTICS")
        
        print(f"\n[STATS] {self.color_text('Overall Statistics:', 'BOLD')}")
        print(f"   • Total items processed: {self.color_text(str(stats['total_items']), 'GREEN')}")
        print(f"   • Total tokens (words): {self.color_text(str(stats['total_tokens']), 'GREEN')}")
        print(f"   • Unique tokens: {self.color_text(str(stats['unique_tokens']), 'GREEN')}")
        avg_tokens = f"{stats['avg_tokens_per_item']:.1f}"
        print(f"   • Average tokens per item: {self.color_text(avg_tokens, 'GREEN')}")
        print(f"   • Total bigrams: {self.color_text(str(stats['total_bigrams']), 'GREEN')}")
        print(f"   • Unique bigrams: {self.color_text(str(stats['unique_bigrams']), 'GREEN')}")
        
        self.print_section("Top 20 Most Frequent Words")
        for idx, (word, count) in enumerate(stats['top_words'][:20], 1):
            bar_length = int((count / stats['top_words'][0][1]) * 30)
            bar = '█' * bar_length
            print(f"   {idx:2}. {word:20} {self.color_text(bar, 'BLUE')} {count:4}")
        
        self.print_section("Top 15 Most Frequent Word Pairs (Bigrams)")
        for idx, (bigram, count) in enumerate(stats['top_bigrams'][:15], 1):
            bigram_display = bigram.replace('_', ' → ')
            bar_length = int((count / stats['top_bigrams'][0][1]) * 25)
            bar = '▓' * bar_length
            print(f"   {idx:2}. {bigram_display:35} {self.color_text(bar, 'CYAN')} {count:3}")
    
    def display_cleaned_sample(self, cleaned_data, max_items=5):
        """Display sample of cleaned data"""
        self.print_header("CLEANED DATA SAMPLE")
        
        print(f"\nShowing {min(max_items, len(cleaned_data))} of {len(cleaned_data)} items\n")
        
        for idx, item in enumerate(cleaned_data[:max_items], 1):
            print(self.color_text(f"[{idx}] {item.get('source', 'Unknown')}", 'YELLOW'))
            
            if item.get('title'):
                print(f"    Title: {item.get('title')[:70]}...")
            
            print(f"    {self.color_text('Cleaned:', 'GREEN')} {item.get('cleaned_text', '')[:100]}...")
            print(f"    {self.color_text('Tokens:', 'CYAN')} {item.get('token_count', 0)} words")
            
            tokens = item.get('tokens', [])[:10]
            if tokens:
                print(f"    {self.color_text('Sample tokens:', 'BLUE')} {', '.join(tokens)}")
            
            bigrams = item.get('bigrams', [])[:5]
            if bigrams:
                bigram_display = ', '.join([b.replace('_', '→') for b in bigrams])
                print(f"    {self.color_text('Bigrams:', 'CYAN')} {bigram_display}")
            
            print()
    
    def display_topic_keywords(self, stats):
        """Display key topic-related words"""
        self.print_section("Key Topic Words (Smart Farming & AI Livestock)")
        
        # Filter for relevant topic words related to "smart farm technology/ AI use in -Livestock "
        topic_words = {
            'Technology': ['ai', 'technology', 'sensor', 'system', 'data', 'machine', 'automate', 'smart', 'precision', 'robot', 'software', 'app', 'device'],
            'Livestock': ['cattle', 'cow', 'dairy', 'livestock', 'animal', 'herd', 'calf', 'milk', 'beef'],
            'Farming': ['farm', 'agriculture', 'agricultural', 'crop', 'field', 'land', 'rural'],
            'Management': ['manage', 'monitor', 'track', 'feed', 'health', 'productive', 'efficiency']
        }
        
        word_freq = dict(stats['top_words'])
        
        for category, keywords in topic_words.items():
            print(f"\n   {self.color_text(category + ':', 'BOLD')}")
            found = []
            for word in keywords:
                if word in word_freq:
                    found.append((word, word_freq[word]))
            
            # Sort by frequency in descending order
            found.sort(key=lambda x: x[1], reverse=True)
            
            for word, count in found[:8]:
                print(f"      • {word:15} ({count} occurrences)")
            
            if not found:
                print(f"      {self.color_text('(no keywords found)', 'RED')}")
    
    def create_summary_report(self, raw_data, cleaned_data, stats, filename='summary_report.txt'):
        """Create a text summary report"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SMART FARM TECHNOLOGY & AI IN LIVESTOCK - SCRAPER REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SCRAPING SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total items scraped: {len(raw_data)}\n")
            f.write(f"Items after cleaning: {len(cleaned_data)}\n\n")
            
            # Count sources
            sources = {}
            for item in raw_data:
                source = item.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
            
            f.write("Sources breakdown:\n")
            for source, count in sources.items():
                f.write(f"  • {source}: {count}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("TEXT ANALYSIS STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total tokens (words): {stats['total_tokens']}\n")
            f.write(f"Unique tokens: {stats['unique_tokens']}\n")
            f.write(f"Average tokens per item: {stats['avg_tokens_per_item']:.1f}\n")
            f.write(f"Total bigrams: {stats['total_bigrams']}\n")
            f.write(f"Unique bigrams: {stats['unique_bigrams']}\n\n")
            
            f.write("TOP 20 WORDS\n")
            f.write("-" * 80 + "\n")
            for idx, (word, count) in enumerate(stats['top_words'][:20], 1):
                f.write(f"{idx:2}. {word:20} {count:4}\n")
            
            f.write("\nTOP 15 BIGRAMS\n")
            f.write("-" * 80 + "\n")
            for idx, (bigram, count) in enumerate(stats['top_bigrams'][:15], 1):
                bigram_display = bigram.replace('_', ' → ')
                f.write(f"{idx:2}. {bigram_display:35} {count:3}\n")
        
        print(f"\n[SAVED] Summary report saved to {filename}")


if __name__ == "__main__":
    sample_data = [
        {
            'source': 'Reddit',
            'subreddit': 'farming',
            'type': 'post',
            'title': 'AI technology revolutionizing dairy farm management',
            'text': 'Smart sensors are now tracking cattle health in real-time...',
            'score': 156,
            'num_comments': 45
        }
    ]
    
    display = DataDisplay()
    display.display_raw_data(sample_data)
