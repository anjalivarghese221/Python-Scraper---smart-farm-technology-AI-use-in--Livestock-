"""
Network Analysis - Classify and Map Keywords
Step 3: Build keyword co-occurrence network and identify topic clusters

This script:
1. Extracts keywords from classified sentiment data
2. Builds co-occurrence network (which keywords appear together)
3. Performs community detection to identify topic clusters
4. Analyzes network properties and keyword relationships
"""

import json
import networkx as nx
from collections import Counter, defaultdict
from itertools import combinations
import pickle
import os


class KeywordNetworkAnalyzer:
    """Analyze keyword relationships and build co-occurrence networks"""
    
    def __init__(self, min_keyword_freq=3, min_cooccurrence=2):
        """
        Initialize network analyzer
        
        Args:
            min_keyword_freq: Minimum frequency for keyword to be included
            min_cooccurrence: Minimum co-occurrence count for edge creation
        """
        self.min_keyword_freq = min_keyword_freq
        self.min_cooccurrence = min_cooccurrence
        self.graph = nx.Graph()
        self.keyword_freq = Counter()
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        self.communities = []
        self.keyword_sentiment = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        
        # Comprehensive stop words list - remove all meaningless words
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
            'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
            'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
            'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
            # Additional common words to filter
            'get', 'got', 'getting', 'could', 'would', 'may', 'might', 'must', 'also',
            'one', 'two', 'first', 'second', 'new', 'make', 'made', 'making', 'use', 'used',
            'using', 'like', 'really', 'even', 'much', 'many', 'back', 'way', 'well',
            'still', 'though', 'see', 'seem', 'look', 'looking', 'think', 'thought',
            'know', 'said', 'say', 'says', 'saying', 'thing', 'things', 'work', 'works',
            'working', 'lot', 'lots', 'bit', 'quite', 'pretty', 'anything', 'something',
            'someone', 'anyone', 'everyone', 'everything', 'nothing', 'nobody', 'yes',
            'yeah', 'sure', 'maybe', 'probably', 'actually', 'basically', 'literally',
            'going', 'come', 'came', 'coming', 'take', 'took', 'taking', 'give', 'gave',
            'giving', 'need', 'needs', 'needed', 'want', 'wants', 'wanted', 'tell',
            'told', 'telling', 'put', 'puts', 'putting', 'keep', 'keeps', 'keeping',
            'let', 'lets', 'letting', 'ask', 'asked', 'asking', 'try', 'tried', 'trying',
            'find', 'found', 'finding', 'call', 'called', 'calling', 'feel', 'felt',
            'feeling', 'become', 'became', 'becoming', 'leave', 'left', 'leaving',
            'mean', 'means', 'meant', 'meaning', 'help', 'helps', 'helped', 'helping',
            'right', 'long', 'good', 'bad', 'high', 'low', 'old', 'young', 'little',
            'big', 'large', 'small', 'great', 'different', 'important', 'another',
            'last', 'next', 'best', 'better', 'worse', 'worst', 'less', 'least',
            'every', 'per', 'since', 'ago', 'yet', 'however', 'therefore', 'thus'
        ])
        
    def extract_keywords(self, text, min_length=4):
        """Extract meaningful keywords from text - filter out stop words and short words"""
        if not text:
            return []
        
        # Split and filter
        words = text.lower().split()
        keywords = [
            w for w in words 
            if len(w) >= min_length 
            and w.isalpha() 
            and w not in self.stop_words
        ]
        
        return keywords
    
    def load_classified_data(self, filepath):
        """Load classified sentiment data"""
        print(f"\nLoading classified data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} classified posts")
        return data
    
    def build_keyword_frequencies(self, data):
        """Count keyword frequencies across all posts"""
        print("\nAnalyzing keyword frequencies...")
        
        for item in data:
            # Get cleaned text
            text = item.get('cleaned_text', '')
            if not text:
                # Fallback to title + body
                text = item.get('title', '') + ' ' + item.get('selftext', '')
            
            # Extract keywords
            keywords = self.extract_keywords(text)
            
            # Track sentiment for each keyword
            sentiment = item.get('sentiment', 'neutral')
            for keyword in set(keywords):
                self.keyword_freq[keyword] += 1
                self.keyword_sentiment[keyword][sentiment] += 1
        
        # Filter by minimum frequency
        filtered_keywords = {k: v for k, v in self.keyword_freq.items() 
                           if v >= self.min_keyword_freq}
        
        print(f"Total unique keywords: {len(self.keyword_freq)}")
        print(f"Filtered keywords (freq >= {self.min_keyword_freq}): {len(filtered_keywords)}")
        
        self.keyword_freq = Counter(filtered_keywords)
        return self.keyword_freq
    
    def build_cooccurrence_network(self, data, window_size=None):
        """Build keyword co-occurrence network"""
        print("\nBuilding co-occurrence network...")
        
        # Get top keywords only
        top_keywords = set(self.keyword_freq.keys())
        
        for item in data:
            # Get text
            text = item.get('cleaned_text', '')
            if not text:
                text = item.get('title', '') + ' ' + item.get('selftext', '')
            
            # Extract keywords
            keywords = self.extract_keywords(text)
            
            # Filter to only top keywords
            keywords = [k for k in keywords if k in top_keywords]
            
            # Count co-occurrences
            if window_size:
                # Sliding window approach
                for i in range(len(keywords)):
                    for j in range(i+1, min(i+window_size, len(keywords))):
                        word1, word2 = sorted([keywords[i], keywords[j]])
                        self.cooccurrence_matrix[word1][word2] += 1
            else:
                # Document-level co-occurrence
                unique_keywords = list(set(keywords))
                for word1, word2 in combinations(sorted(unique_keywords), 2):
                    self.cooccurrence_matrix[word1][word2] += 1
        
        # Build graph
        edge_count = 0
        for word1, neighbors in self.cooccurrence_matrix.items():
            for word2, weight in neighbors.items():
                if weight >= self.min_cooccurrence:
                    self.graph.add_edge(word1, word2, weight=weight)
                    edge_count += 1
        
        # Add sentiment attributes to nodes
        for keyword in self.graph.nodes():
            self.graph.nodes[keyword]['sentiment'] = self.keyword_sentiment[keyword]
        
        print(f"Network nodes (keywords): {self.graph.number_of_nodes()}")
        print(f"Network edges (co-occurrences): {self.graph.number_of_edges()}")
        
        return self.graph
    
    def detect_communities(self):
        """Detect topic clusters using community detection"""
        print("\nDetecting topic communities...")
        
        if self.graph.number_of_nodes() == 0:
            print("No nodes in graph. Cannot detect communities.")
            return []
        
        # Use Louvain community detection
        try:
            from networkx.algorithms import community
            self.communities = list(community.greedy_modularity_communities(self.graph))
            print(f"Detected {len(self.communities)} communities")
            
            # Print community sizes
            for i, comm in enumerate(self.communities, 1):
                print(f"  Community {i}: {len(comm)} keywords")
            
        except Exception as e:
            print(f"Community detection error: {e}")
            print("Using connected components instead...")
            self.communities = list(nx.connected_components(self.graph))
            print(f"Found {len(self.communities)} connected components")
        
        return self.communities
    
    def analyze_communities(self, min_community_size=5):
        """Analyze and label detected communities - filter out small/noise communities"""
        community_analysis = []
        community_id = 1  # Track ID for meaningful communities only
        
        for i, community in enumerate(self.communities, 1):
            keywords = list(community)
            
            # Filter out small communities (noise)
            if len(keywords) < min_community_size:
                print(f"  Skipping community {i} (only {len(keywords)} keywords - too small)")
                continue
            
            total_freq = sum(self.keyword_freq.get(k, 0) for k in keywords)
            
            # Top keywords by frequency
            top_keywords = sorted(
                [(k, self.keyword_freq.get(k, 0)) for k in keywords],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            # Sentiment distribution
            sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 0}
            for keyword in keywords:
                for sent, count in self.keyword_sentiment[keyword].items():
                    sentiment_dist[sent] += count
            
            total_sent = sum(sentiment_dist.values())
            sentiment_pct = {
                sent: (count / total_sent * 100) if total_sent > 0 else 0
                for sent, count in sentiment_dist.items()
            }
            
            topic_label = self.infer_topic_label(top_keywords)
            
            community_analysis.append({
                'community_id': community_id,
                'size': len(keywords),
                'total_frequency': total_freq,
                'top_keywords': top_keywords,
                'topic_label': topic_label,
                'sentiment_distribution': sentiment_dist,
                'sentiment_percentages': sentiment_pct,
                'all_keywords': keywords
            })
            
            community_id += 1
        
        print(f"\nRetained {len(community_analysis)} meaningful communities (≥{min_community_size} keywords)")
        return community_analysis
    
    def infer_topic_label(self, top_keywords):
        """Infer topic label from top keywords"""
        keywords_list = [k for k, _ in top_keywords[:5]]
        keywords_str = ' '.join(keywords_list)
        
        # Simple heuristic-based labeling
        if any(word in keywords_str for word in ['dairy', 'milk', 'cow', 'cattle', 'herd']):
            return 'Dairy & Livestock Management'
        elif any(word in keywords_str for word in ['sensor', 'monitor', 'device', 'iot', 'technology']):
            return 'Sensors & Monitoring Technology'
        elif any(word in keywords_str for word in ['farm', 'agriculture', 'crop', 'field', 'land']):
            return 'General Farming & Agriculture'
        elif any(word in keywords_str for word in ['data', 'system', 'software', 'app', 'platform']):
            return 'Data Systems & Software'
        elif any(word in keywords_str for word in ['automation', 'robot', 'machine', 'automated']):
            return 'Automation & Robotics'
        elif any(word in keywords_str for word in ['precision', 'smart', 'digital', 'intelligent']):
            return 'Precision & Smart Farming'
        else:
            return 'Mixed Topics'
    
    def get_network_statistics(self):
        """Calculate network-level statistics"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'num_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph) if self.graph.number_of_edges() > 0 else 0,
            'top_degree_centrality': sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
            'top_betweenness_centrality': sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def save_results(self, community_analysis, stats, 
                    graph_file='keyword_network.pkl',
                    analysis_file='network_analysis_results.json'):
        """Save network analysis results"""
        print(f"\nSaving results...")
        
        # Save graph
        with open(graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to: {graph_file}")
        
        # Save analysis results
        results = {
            'network_statistics': stats,
            'communities': community_analysis,
            'keyword_frequencies': dict(self.keyword_freq.most_common(100)),
            'parameters': {
                'min_keyword_freq': self.min_keyword_freq,
                'min_cooccurrence': self.min_cooccurrence
            }
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Analysis results saved to: {analysis_file}")
        
        return results
    
    def generate_report(self, community_analysis, stats, output_file='network_analysis_report.txt'):
        """Generate concise network analysis report"""
        lines = [
            "=" * 70,
            "KEYWORD NETWORK ANALYSIS REPORT",
            "Smart Farm Technology - Topic Clustering & Relationships",
            "=" * 70,
            "\n" + "-" * 70,
            "NETWORK OVERVIEW",
            "-" * 70,
            f"Keywords: {stats['num_nodes']} | Connections: {stats['num_edges']}",
            f"Density: {stats['density']:.3f} | Clustering: {stats.get('avg_clustering', 0):.3f}",
            "\n" + "-" * 70,
            "TOP 10 KEYWORDS (by connections)",
            "-" * 70
        ]
        
        for i, (keyword, score) in enumerate(stats['top_degree_centrality'][:10], 1):
            lines.append(f"  {i:2d}. {keyword:20s} ({score:.3f})")
        
        lines.extend(["\n" + "-" * 70, "TOPIC COMMUNITIES", "-" * 70])
        
        for comm in community_analysis:
            lines.append(f"\nCommunity {comm['community_id']}: {comm['topic_label']}")
            lines.append(f"  Keywords: {comm['size']} | Frequency: {comm['total_frequency']}")
            lines.append(f"  Top: {', '.join([k for k, _ in comm['top_keywords'][:5]])}")
            
            sp = comm['sentiment_percentages']
            lines.append(f"  Sentiment: {sp['positive']:.0f}% pos | {sp['negative']:.0f}% neg | {sp['neutral']:.0f}% neutral")
        
        lines.extend(["\n" + "=" * 70, "END OF REPORT", "=" * 70])
        
        report_text = '\n'.join(lines)
        print(report_text)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


def main():
    """Main execution function"""
    print("=" * 70)
    print("KEYWORD NETWORK ANALYSIS - Step 3")
    print("=" * 70)
    
    # Lower thresholds for more connections (was 8, 4)
    analyzer = KeywordNetworkAnalyzer(min_keyword_freq=3, min_cooccurrence=2)
    
    input_file = 'classified_sentiment_data.json'
    if not os.path.exists(input_file):
        print(f"\nERROR: {input_file} not found. Run Step 2 first.")
        return
    
    print(f"\nLoading data...")
    data = analyzer.load_classified_data(input_file)
    
    print("Building network...")
    analyzer.build_keyword_frequencies(data)
    analyzer.build_cooccurrence_network(data)
    analyzer.detect_communities()
    
    print("Analyzing communities...")
    community_analysis = analyzer.analyze_communities()
    stats = analyzer.get_network_statistics()
    
    print("Saving results...")
    analyzer.save_results(community_analysis, stats)
    analyzer.generate_report(community_analysis, stats)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
