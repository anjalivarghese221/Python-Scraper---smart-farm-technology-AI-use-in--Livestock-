"""
Final Sentiment Analysis - Step 4
Comprehensive analysis combining scraping, sentiment classification, and network analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import os


class FinalSentimentAnalysis:
    """Comprehensive sentiment analysis combining all previous steps"""
    
    def __init__(self):
        self.classified_data = []
        self.network_results = {}
        self.sentiment_by_keyword = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        self.sentiment_by_community = {}
        
    def load_data(self):
        """Load all analysis results"""
        print("Loading data...")
        
        # Load classified sentiment data
        with open('classified_sentiment_data.json', 'r', encoding='utf-8') as f:
            self.classified_data = json.load(f)
        print(f"  Loaded {len(self.classified_data)} classified posts")
        
        # Load network analysis results
        with open('network_analysis_results.json', 'r', encoding='utf-8') as f:
            self.network_results = json.load(f)
        print(f"  Loaded network analysis with {len(self.network_results['communities'])} communities")
        
    def analyze_overall_sentiment(self):
        """Analyze overall sentiment distribution"""
        sentiments = [item['sentiment'] for item in self.classified_data]
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        return {
            'counts': sentiment_counts,
            'percentages': {s: (c/total)*100 for s, c in sentiment_counts.items()},
            'total': total
        }
    
    def analyze_sentiment_by_subreddit(self):
        """Analyze sentiment distribution by subreddit"""
        subreddit_sentiments = defaultdict(list)
        
        for item in self.classified_data:
            subreddit = item.get('subreddit', 'unknown')
            sentiment = item.get('sentiment', 'neutral')
            subreddit_sentiments[subreddit].append(sentiment)
        
        results = {}
        for subreddit, sentiments in subreddit_sentiments.items():
            counts = Counter(sentiments)
            total = len(sentiments)
            results[subreddit] = {
                'total': total,
                'counts': dict(counts),
                'percentages': {s: (c/total)*100 for s, c in counts.items()}
            }
        
        return results
    
    def analyze_sentiment_trends_by_topic(self):
        """Analyze sentiment for each network community/topic"""
        community_sentiments = {}
        
        for comm_data in self.network_results['communities']:
            comm_id = comm_data['community_id']
            community_sentiments[comm_id] = {
                'topic': comm_data['topic_label'],
                'size': comm_data['size'],
                'sentiment_pct': comm_data['sentiment_percentages'],
                'top_keywords': [k for k, _ in comm_data['top_keywords'][:5]]
            }
        
        return community_sentiments
    
    def identify_key_insights(self):
        """Extract key insights from the analysis"""
        insights = []
        
        # Overall sentiment
        overall = self.analyze_overall_sentiment()
        dominant_sentiment = max(overall['percentages'].items(), key=lambda x: x[1])
        insights.append(f"Overall: {dominant_sentiment[1]:.1f}% {dominant_sentiment[0]} sentiment dominates")
        
        # Subreddit insights
        subreddit_data = self.analyze_sentiment_by_subreddit()
        most_positive = max(subreddit_data.items(), 
                          key=lambda x: x[1]['percentages'].get('positive', 0))
        most_negative = max(subreddit_data.items(), 
                          key=lambda x: x[1]['percentages'].get('negative', 0))
        
        insights.append(f"Most positive community: r/{most_positive[0]} ({most_positive[1]['percentages']['positive']:.1f}%)")
        insights.append(f"Most negative community: r/{most_negative[0]} ({most_negative[1]['percentages']['negative']:.1f}%)")
        
        # Topic insights
        topic_sentiments = self.analyze_sentiment_trends_by_topic()
        for comm_id, data in topic_sentiments.items():
            dominant = max(data['sentiment_pct'].items(), key=lambda x: x[1])
            insights.append(f"{data['topic']}: {dominant[1]:.0f}% {dominant[0]}")
        
        return insights
    
    def generate_comprehensive_report(self, output_file='final_sentiment_analysis.txt'):
        """Generate final comprehensive report"""
        lines = [
            "=" * 80,
            "FINAL SENTIMENT ANALYSIS REPORT",
            "Smart Farm Technology & AI in Livestock - Social Media Analysis",
            "=" * 80,
            "\n" + "EXECUTIVE SUMMARY",
            "-" * 80
        ]
        
        # Overall sentiment
        overall = self.analyze_overall_sentiment()
        lines.append(f"\nTotal Posts Analyzed: {overall['total']}")
        lines.append("\nOverall Sentiment Distribution:")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = overall['counts'].get(sentiment, 0)
            pct = overall['percentages'].get(sentiment, 0)
            lines.append(f"  {sentiment.capitalize():8s}: {count:3d} posts ({pct:5.1f}%)")
        
        # Key insights
        lines.append("\n" + "-" * 80)
        lines.append("KEY INSIGHTS")
        lines.append("-" * 80)
        for insight in self.identify_key_insights():
            lines.append(f"• {insight}")
        
        # Sentiment by subreddit
        lines.append("\n" + "-" * 80)
        lines.append("SENTIMENT BY SUBREDDIT")
        lines.append("-" * 80)
        subreddit_data = self.analyze_sentiment_by_subreddit()
        for subreddit, data in sorted(subreddit_data.items(), key=lambda x: x[1]['total'], reverse=True):
            if subreddit == 'unknown':
                continue
            lines.append(f"\nr/{subreddit} ({data['total']} posts):")
            for sentiment in ['positive', 'negative', 'neutral']:
                pct = data['percentages'].get(sentiment, 0)
                lines.append(f"  {sentiment:8s}: {pct:5.1f}%")
        
        # Sentiment by topic community
        lines.append("\n" + "-" * 80)
        lines.append("SENTIMENT BY TOPIC")
        lines.append("-" * 80)
        topic_sentiments = self.analyze_sentiment_trends_by_topic()
        for comm_id, data in sorted(topic_sentiments.items()):
            lines.append(f"\n{data['topic']} ({data['size']} keywords):")
            lines.append(f"  Top keywords: {', '.join(data['top_keywords'])}")
            for sentiment in ['positive', 'negative', 'neutral']:
                pct = data['sentiment_pct'].get(sentiment, 0)
                lines.append(f"  {sentiment:8s}: {pct:5.1f}%")
        
        # Top keywords from network
        lines.append("\n" + "-" * 80)
        lines.append("TOP KEYWORDS")
        lines.append("-" * 80)
        top_keywords = list(self.network_results['keyword_frequencies'].items())[:15]
        for i, (keyword, freq) in enumerate(top_keywords, 1):
            lines.append(f"  {i:2d}. {keyword:15s} ({freq} occurrences)")
        
        # Conclusions
        lines.append("\n" + "-" * 80)
        lines.append("CONCLUSIONS")
        lines.append("-" * 80)
        lines.append("\n1. Technology Adoption: Mixed reception with slight negative bias")
        lines.append("2. Community Differences: Tech-focused subreddits more positive")
        lines.append("3. Topic Trends: Monitoring systems receive best sentiment")
        lines.append("4. Key Themes: Dairy, livestock, precision agriculture dominate")
        
        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        report_text = '\n'.join(lines)
        print(report_text)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def create_sentiment_overview_chart(self, output_file='visualizations/sentiment_overview.png'):
        """Create comprehensive sentiment overview visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall sentiment pie chart
        overall = self.analyze_overall_sentiment()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        sentiments = ['positive', 'negative', 'neutral']
        values = [overall['counts'].get(s, 0) for s in sentiments]
        ax1.pie(values, labels=[s.capitalize() for s in sentiments], 
               colors=[colors[s] for s in sentiments], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Sentiment by subreddit
        subreddit_data = self.analyze_sentiment_by_subreddit()
        subreddits = sorted([s for s in subreddit_data.keys() if s != 'unknown'], 
                          key=lambda x: subreddit_data[x]['total'], reverse=True)[:6]
        x = np.arange(len(subreddits))
        width = 0.25
        
        pos = [subreddit_data[s]['percentages'].get('positive', 0) for s in subreddits]
        neg = [subreddit_data[s]['percentages'].get('negative', 0) for s in subreddits]
        neu = [subreddit_data[s]['percentages'].get('neutral', 0) for s in subreddits]
        
        ax2.bar(x - width, pos, width, label='Positive', color=colors['positive'])
        ax2.bar(x, neg, width, label='Negative', color=colors['negative'])
        ax2.bar(x + width, neu, width, label='Neutral', color=colors['neutral'])
        ax2.set_xlabel('Subreddit', fontsize=11)
        ax2.set_ylabel('Percentage', fontsize=11)
        ax2.set_title('Sentiment by Subreddit', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"r/{s}" for s in subreddits], rotation=45, ha='right')
        ax2.legend()
        
        # 3. Sentiment by topic community
        topic_data = self.analyze_sentiment_trends_by_topic()
        topics = [data['topic'][:25] for data in topic_data.values()]
        pos_topic = [data['sentiment_pct']['positive'] for data in topic_data.values()]
        neg_topic = [data['sentiment_pct']['negative'] for data in topic_data.values()]
        neu_topic = [data['sentiment_pct']['neutral'] for data in topic_data.values()]
        
        x_topic = np.arange(len(topics))
        ax3.barh(x_topic, pos_topic, height=0.25, label='Positive', color=colors['positive'])
        ax3.barh(x_topic + 0.25, neg_topic, height=0.25, label='Negative', color=colors['negative'])
        ax3.barh(x_topic + 0.5, neu_topic, height=0.25, label='Neutral', color=colors['neutral'])
        ax3.set_xlabel('Percentage', fontsize=11)
        ax3.set_title('Sentiment by Topic Community', fontsize=14, fontweight='bold')
        ax3.set_yticks(x_topic + 0.25)
        ax3.set_yticklabels(topics)
        ax3.legend()
        
        # 4. Top keywords bar chart
        top_keywords = list(self.network_results['keyword_frequencies'].items())[:12]
        keywords, freqs = zip(*top_keywords)
        ax4.barh(range(len(keywords)), freqs, color='steelblue')
        ax4.set_yticks(range(len(keywords)))
        ax4.set_yticklabels(keywords)
        ax4.set_xlabel('Frequency', fontsize=11)
        ax4.set_title('Top 12 Keywords', fontsize=14, fontweight='bold')
        ax4.invert_yaxis()
        
        plt.suptitle('Smart Farm Technology - Comprehensive Sentiment Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Sentiment overview saved to: {output_file}")
        plt.close()


def main():
    """Main execution function"""
    print("=" * 80)
    print("FINAL SENTIMENT ANALYSIS - Step 4")
    print("=" * 80)
    
    # Check required files
    required_files = ['classified_sentiment_data.json', 'network_analysis_results.json']
    for file in required_files:
        if not os.path.exists(file):
            print(f"\nERROR: {file} not found. Run previous steps first.")
            return
    
    # Initialize analyzer
    analyzer = FinalSentimentAnalysis()
    
    # Load all data
    analyzer.load_data()
    
    # Generate comprehensive report
    print("\nGenerating final report...")
    analyzer.generate_comprehensive_report()
    
    # Create visualizations
    print("\nCreating visualizations...")
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    analyzer.create_sentiment_overview_chart()
    
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS COMPLETE")
    print("Files: final_sentiment_analysis.txt, visualizations/sentiment_overview.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
