#!/usr/bin/env python3
"""
Temporal Sentiment Trend Analysis
Implements time-series analysis for agricultural technology discourse

Features:
- Monthly/quarterly sentiment trends
- Statistical trend detection
- Event-based analysis (policy shocks, media events)
- Seasonal pattern identification
- Subreddit-specific trend comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


class TemporalSentimentAnalyzer:
    """
    Time-series sentiment analysis for peer-reviewed research
    Identifies temporal patterns, seasonality, and trend shifts
    """
    
    def __init__(self, data_file: str = 'classified_sentiment_data.json'):
        """Load classified sentiment data with temporal metadata"""
        with open(data_file, 'r') as f:
            self.data = json.load(f)
            
        self.posts = self.data if isinstance(self.data, list) else self.data.get('data', [])
        
        # Sentiment mapping
        self.sentiment_to_score = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
    def extract_temporal_data(self) -> List[Dict]:
        """
        Extract and validate temporal metadata
        Required: created_at timestamp for each post
        """
        temporal_posts = []
        
        for post in self.posts:
            # Check for required temporal field
            if 'created_at' in post or 'created_utc' in post:
                try:
                    # Parse datetime
                    if 'created_at' in post:
                        if isinstance(post['created_at'], str):
                            dt = datetime.fromisoformat(post['created_at'].replace('Z', '+00:00'))
                        else:
                            continue
                    else:
                        dt = datetime.fromtimestamp(post['created_utc'])
                    
                    post['datetime'] = dt
                    post['year'] = dt.year
                    post['month'] = dt.month
                    post['quarter'] = (dt.month - 1) // 3 + 1
                    post['year_month'] = f"{dt.year}-{dt.month:02d}"
                    post['year_quarter'] = f"{dt.year}-Q{post['quarter']}"
                    
                    # Sentiment score
                    sentiment = post.get('sentiment', 'neutral')
                    post['sentiment_score'] = self.sentiment_to_score.get(sentiment, 0)
                    
                    temporal_posts.append(post)
                except Exception as e:
                    continue
                    
        print(f"[TEMPORAL DATA] {len(temporal_posts)} posts with valid timestamps")
        return temporal_posts
    
    def analyze_monthly_trends(self, posts: List[Dict]) -> Dict:
        """
        Calculate monthly sentiment aggregations
        Returns time-series data for trend visualization
        """
        print("\n[ANALYSIS] Monthly sentiment trends...")
        
        monthly_data = defaultdict(lambda: {
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'total': 0,
            'sentiment_score_sum': 0,
            'posts': []
        })
        
        for post in posts:
            ym = post['year_month']
            sentiment = post.get('sentiment', 'neutral')
            
            monthly_data[ym][sentiment] += 1
            monthly_data[ym]['total'] += 1
            monthly_data[ym]['sentiment_score_sum'] += post['sentiment_score']
            monthly_data[ym]['posts'].append(post)
            
        # Calculate percentages and average sentiment
        monthly_trends = {}
        for ym, data in sorted(monthly_data.items()):
            total = data['total']
            monthly_trends[ym] = {
                'year_month': ym,
                'total_posts': total,
                'positive_count': data['positive'],
                'neutral_count': data['neutral'],
                'negative_count': data['negative'],
                'positive_pct': (data['positive'] / total) * 100,
                'neutral_pct': (data['neutral'] / total) * 100,
                'negative_pct': (data['negative'] / total) * 100,
                'avg_sentiment_score': data['sentiment_score_sum'] / total,
                'net_sentiment': ((data['positive'] - data['negative']) / total) * 100
            }
            
        return monthly_trends
    
    def analyze_quarterly_trends(self, posts: List[Dict]) -> Dict:
        """
        Quarterly aggregation for seasonal pattern analysis
        Captures production cycles and policy announcement periods
        """
        print("\n[ANALYSIS] Quarterly sentiment trends...")
        
        quarterly_data = defaultdict(lambda: {
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'total': 0,
            'sentiment_score_sum': 0
        })
        
        for post in posts:
            yq = post['year_quarter']
            sentiment = post.get('sentiment', 'neutral')
            
            quarterly_data[yq][sentiment] += 1
            quarterly_data[yq]['total'] += 1
            quarterly_data[yq]['sentiment_score_sum'] += post['sentiment_score']
            
        quarterly_trends = {}
        for yq, data in sorted(quarterly_data.items()):
            total = data['total']
            quarterly_trends[yq] = {
                'year_quarter': yq,
                'total_posts': total,
                'positive_pct': (data['positive'] / total) * 100,
                'neutral_pct': (data['neutral'] / total) * 100,
                'negative_pct': (data['negative'] / total) * 100,
                'avg_sentiment_score': data['sentiment_score_sum'] / total,
                'net_sentiment': ((data['positive'] - data['negative']) / total) * 100
            }
            
        return quarterly_trends
    
    def analyze_subreddit_temporal_trends(self, posts: List[Dict]) -> Dict:
        """
        Subreddit-specific temporal analysis
        Reveals community-level sentiment evolution
        """
        print("\n[ANALYSIS] Subreddit-specific temporal trends...")
        
        subreddit_trends = defaultdict(lambda: defaultdict(lambda: {
            'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0
        }))
        
        for post in posts:
            subreddit = post.get('subreddit', 'unknown')
            ym = post['year_month']
            sentiment = post.get('sentiment', 'neutral')
            
            subreddit_trends[subreddit][ym][sentiment] += 1
            subreddit_trends[subreddit][ym]['total'] += 1
            
        # Format results
        formatted_trends = {}
        for subreddit, monthly in subreddit_trends.items():
            formatted_trends[subreddit] = {}
            for ym, data in sorted(monthly.items()):
                total = data['total']
                if total > 0:
                    formatted_trends[subreddit][ym] = {
                        'positive_pct': (data['positive'] / total) * 100,
                        'neutral_pct': (data['neutral'] / total) * 100,
                        'negative_pct': (data['negative'] / total) * 100,
                        'total_posts': total
                    }
                    
        return formatted_trends
    
    def detect_trend_direction(self, trends: Dict) -> Dict:
        """
        Statistical trend detection (increasing, decreasing, stable)
        Uses linear regression on sentiment scores
        """
        print("\n[TREND DETECTION] Statistical analysis...")
        
        # Extract time series
        sorted_months = sorted(trends.keys())
        sentiment_scores = [trends[month]['avg_sentiment_score'] for month in sorted_months]
        positive_pcts = [trends[month]['positive_pct'] for month in sorted_months]
        
        if len(sentiment_scores) < 3:
            return {'trend': 'insufficient_data'}
            
        # Simple linear regression
        x = np.arange(len(sentiment_scores))
        
        # Sentiment score trend
        coef_sentiment = np.polyfit(x, sentiment_scores, 1)[0]
        
        # Positive percentage trend  
        coef_positive = np.polyfit(x, positive_pcts, 1)[0]
        
        # Interpret trends
        if coef_sentiment > 0.01:
            sentiment_trend = 'improving'
        elif coef_sentiment < -0.01:
            sentiment_trend = 'declining'
        else:
            sentiment_trend = 'stable'
            
        if coef_positive > 0.5:
            positive_trend = 'increasing'
        elif coef_positive < -0.5:
            positive_trend = 'decreasing'
        else:
            positive_trend = 'stable'
            
        return {
            'sentiment_trend': sentiment_trend,
            'sentiment_slope': float(coef_sentiment),
            'positive_trend': positive_trend,
            'positive_slope': float(coef_positive),
            'time_periods': len(sentiment_scores),
            'interpretation': self._interpret_trend(sentiment_trend, coef_sentiment)
        }
    
    def _interpret_trend(self, trend: str, slope: float) -> str:
        """Generate human-readable interpretation"""
        if trend == 'improving':
            return f"Sentiment is improving over time (slope: +{slope:.3f})"
        elif trend == 'declining':
            return f"Sentiment is declining over time (slope: {slope:.3f})"
        else:
            return f"Sentiment remains relatively stable (slope: {slope:.3f})"
    
    def visualize_temporal_trends(self, monthly_trends: Dict, quarterly_trends: Dict, output_dir: str = 'visualizations'):
        """
        Generate publication-quality temporal visualizations
        """
        print("\n[VISUALIZATION] Creating temporal trend charts...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Figure 1: Monthly sentiment trends (line chart)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Monthly trends
        months = sorted(monthly_trends.keys())
        positive_pcts = [monthly_trends[m]['positive_pct'] for m in months]
        neutral_pcts = [monthly_trends[m]['neutral_pct'] for m in months]
        negative_pcts = [monthly_trends[m]['negative_pct'] for m in months]
        
        ax1 = axes[0]
        ax1.plot(months, positive_pcts, marker='o', label='Positive', color='#2ecc71', linewidth=2)
        ax1.plot(months, neutral_pcts, marker='s', label='Neutral', color='#95a5a6', linewidth=2)
        ax1.plot(months, negative_pcts, marker='^', label='Negative', color='#e74c3c', linewidth=2)
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Sentiment %', fontsize=12)
        ax1.set_title('Monthly Sentiment Trends: Agricultural Technology Discourse', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Net sentiment (positive - negative)
        net_sentiment = [monthly_trends[m]['net_sentiment'] for m in months]
        ax2 = axes[1]
        colors = ['#2ecc71' if ns > 0 else '#e74c3c' for ns in net_sentiment]
        ax2.bar(months, net_sentiment, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Net Sentiment (Positive - Negative)', fontsize=12)
        ax2.set_title('Net Sentiment Score by Month', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_trends_monthly.png', dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {output_dir}/temporal_trends_monthly.png")
        plt.close()
        
        # Figure 2: Quarterly trends (stacked bar chart)
        quarters = sorted(quarterly_trends.keys())
        pos_q = [quarterly_trends[q]['positive_pct'] for q in quarters]
        neu_q = [quarterly_trends[q]['neutral_pct'] for q in quarters]
        neg_q = [quarterly_trends[q]['negative_pct'] for q in quarters]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(quarters))
        
        ax.bar(x, pos_q, label='Positive', color='#2ecc71', alpha=0.8)
        ax.bar(x, neu_q, bottom=pos_q, label='Neutral', color='#95a5a6', alpha=0.8)
        ax.bar(x, neg_q, bottom=np.array(pos_q) + np.array(neu_q), label='Negative', color='#e74c3c', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(quarters, rotation=45)
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Sentiment Distribution (%)', fontsize=12)
        ax.set_title('Quarterly Sentiment Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_trends_quarterly.png', dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {output_dir}/temporal_trends_quarterly.png")
        plt.close()
    
    def generate_temporal_report(self, monthly_trends: Dict, quarterly_trends: Dict, 
                                 trend_stats: Dict, subreddit_trends: Dict) -> str:
        """Generate comprehensive temporal analysis report"""
        report = "\n" + "=" * 80
        report += "\nTEMPORAL SENTIMENT ANALYSIS REPORT\n"
        report += "Agricultural Technology & AI in Livestock\n"
        report += "=" * 80 + "\n\n"
        
        # Temporal coverage
        months = sorted(monthly_trends.keys())
        report += f"TEMPORAL COVERAGE\n"
        report += f"  Period: {months[0]} to {months[-1]}\n"
        report += f"  Months analyzed: {len(months)}\n"
        report += f"  Total posts: {sum(monthly_trends[m]['total_posts'] for m in months)}\n\n"
        
        # Trend detection
        report += f"TREND ANALYSIS\n"
        report += f"  Overall trend: {trend_stats['sentiment_trend']}\n"
        report += f"  Interpretation: {trend_stats['interpretation']}\n"
        report += f"  Positive sentiment trend: {trend_stats['positive_trend']}\n\n"
        
        # Monthly highlights
        report += f"MONTHLY HIGHLIGHTS\n"
        for month in months[-6:]:  # Last 6 months
            data = monthly_trends[month]
            report += f"  {month}:\n"
            report += f"    Posts: {data['total_posts']}\n"
            report += f"    Positive: {data['positive_pct']:.1f}% | Neutral: {data['neutral_pct']:.1f}% | Negative: {data['negative_pct']:.1f}%\n"
            report += f"    Net sentiment: {data['net_sentiment']:+.1f}%\n"
        
        # Subreddit comparison
        report += f"\nSUBREDDIT TRENDS\n"
        for subreddit in sorted(subreddit_trends.keys()):
            report += f"  r/{subreddit}: {len(subreddit_trends[subreddit])} months of data\n"
            
        report += "\n" + "=" * 80 + "\n"
        return report
    
    def run_full_analysis(self):
        """Execute complete temporal sentiment analysis"""
        print("\n" + "=" * 80)
        print("TEMPORAL SENTIMENT TREND ANALYSIS")
        print("=" * 80)
        
        # Extract temporal data
        posts = self.extract_temporal_data()
        
        if len(posts) < 10:
            print("\n[ERROR] Insufficient temporal data for trend analysis")
            print("  Minimum required: 10 posts with timestamps")
            print("  Current: " + str(len(posts)))
            return
        
        # Run analyses
        monthly_trends = self.analyze_monthly_trends(posts)
        quarterly_trends = self.analyze_quarterly_trends(posts)
        subreddit_trends = self.analyze_subreddit_temporal_trends(posts)
        trend_stats = self.detect_trend_direction(monthly_trends)
        
        # Generate visualizations
        self.visualize_temporal_trends(monthly_trends, quarterly_trends)
        
        # Generate report
        report = self.generate_temporal_report(monthly_trends, quarterly_trends, 
                                               trend_stats, subreddit_trends)
        print(report)
        
        # Save results
        results = {
            'monthly_trends': monthly_trends,
            'quarterly_trends': quarterly_trends,
            'subreddit_trends': subreddit_trends,
            'trend_statistics': trend_stats,
            'report': report
        }
        
        with open('temporal_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("[SAVED] temporal_analysis_results.json")
        
        print("\n" + "=" * 80)
        print("TEMPORAL ANALYSIS COMPLETE")
        print("=" * 80)


def main():
    """Run temporal analysis on classified sentiment data"""
    analyzer = TemporalSentimentAnalyzer('classified_sentiment_data.json')
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
