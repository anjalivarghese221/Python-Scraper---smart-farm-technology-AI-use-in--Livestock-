"""
Network Visualization Script
Generate visual representations of keyword networks and topic clusters
"""

import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import os


class NetworkVisualizer:
    """Visualize keyword networks and topic clusters"""
    
    def __init__(self):
        self.graph = None
        self.communities = []
        self.pos = None
        
    def load_network(self, graph_file='keyword_network.pkl'):
        """Load saved network graph"""
        print(f"\nLoading network from {graph_file}...")
        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def load_analysis_results(self, results_file='network_analysis_results.json'):
        """Load network analysis results"""
        print(f"Loading analysis results from {results_file}...")
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Extract communities
        self.communities = []
        for comm_data in results['communities']:
            self.communities.append(set(comm_data['all_keywords']))
        
        print(f"Loaded {len(self.communities)} communities")
        return results
    
    def visualize_full_network(self, output_file='network_full.png', figsize=(16, 12)):
        """Visualize filtered keyword network - only top nodes"""
        print("\nGenerating focused network visualization...")
        
        if self.graph.number_of_nodes() == 0:
            print("Empty graph. Cannot visualize.")
            return
        
        # Filter to top 50 most important nodes by degree
        degree_dict = dict(self.graph.degree())
        top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:50]
        top_node_names = [node for node, _ in top_nodes]
        
        # Create subgraph with only top nodes
        subgraph = self.graph.subgraph(top_node_names).copy()
        
        # Filter edges - keep only strong connections (top 30% by weight)
        if subgraph.number_of_edges() > 100:
            edges_with_weights = [(u, v, d.get('weight', 1)) for u, v, d in subgraph.edges(data=True)]
            edges_sorted = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)
            top_edges = edges_sorted[:int(len(edges_sorted) * 0.3)]
            
            # Rebuild filtered graph
            filtered_graph = nx.Graph()
            filtered_graph.add_nodes_from(subgraph.nodes())
            filtered_graph.add_weighted_edges_from(top_edges)
            subgraph = filtered_graph
        
        print(f"  Filtered to {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        
        plt.figure(figsize=figsize)
        
        # Better layout with more spacing
        self.pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
        
        # Node sizes based on importance
        node_sizes = [degree_dict[node] * 30 + 300 for node in subgraph.nodes()]
        
        # Draw edges with varying thickness
        edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 for w in edge_weights]
        
        nx.draw_networkx_edges(subgraph, self.pos, alpha=0.7, width=edge_widths, edge_color='#2c3e50')
        nx.draw_networkx_nodes(subgraph, self.pos, node_size=node_sizes, 
                              node_color='skyblue', alpha=0.8, edgecolors='darkblue', linewidths=1.5)
        nx.draw_networkx_labels(subgraph, self.pos, font_size=10, font_weight='bold')
        
        plt.title("Top 50 Keywords - Co-occurrence Network\nSmart Farm Technology & AI in Livestock", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Network visualization saved to: {output_file}")
        plt.close()
    
    def visualize_communities(self, results, output_file='network_communities.png', figsize=(16, 10)):
        """Visualize top keywords by community - much cleaner view"""
        print("\nGenerating community visualization...")
        
        if self.graph.number_of_nodes() == 0:
            print("Empty graph. Cannot visualize.")
            return
        
        # Get top 10 keywords from each community
        top_keywords_per_comm = []
        for comm_data in results['communities']:
            top_keys = [k for k, _ in comm_data['top_keywords'][:10]]
            top_keywords_per_comm.extend(top_keys)
        
        # Create subgraph with only these keywords
        filtered_nodes = [n for n in top_keywords_per_comm if n in self.graph.nodes()]
        subgraph = self.graph.subgraph(filtered_nodes).copy()
        
        print(f"  Showing {subgraph.number_of_nodes()} top keywords from communities")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout with more spacing
        pos = nx.spring_layout(subgraph, k=4, iterations=100, seed=42)
        
        # Color by community
        color_palette = plt.cm.Set3(range(len(self.communities)))
        community_map = {}
        for i, community in enumerate(self.communities):
            for node in community:
                community_map[node] = i
        
        node_colors = [color_palette[community_map.get(n, 0)] for n in subgraph.nodes()]
        
        # Node sizes based on frequency
        node_sizes = [results['keyword_frequencies'].get(node, 1) * 80 + 400 
                     for node in subgraph.nodes()]
        
        # Draw with better visibility
        nx.draw_networkx_edges(subgraph, pos, alpha=0.7, width=2.5, edge_color='#2c3e50', ax=ax)
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.85, 
                              edgecolors='black', linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(subgraph, pos, font_size=11, font_weight='bold', ax=ax)
        
        # Legend
        legend_elements = []
        for i, comm_data in enumerate(results['communities']):
            label = f"Community {i+1}: {comm_data['topic_label']}"
            legend_elements.append(mpatches.Patch(color=color_palette[i], label=label))
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        ax.set_title("Top Keywords by Topic Community", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Community visualization saved to: {output_file}")
        plt.close()
    
    def visualize_top_keywords(self, results, output_file='top_keywords.png', top_n=20):
        """Visualize top keywords by frequency"""
        print("\nGenerating top keywords visualization...")
        
        keyword_freq = results['keyword_frequencies']
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        keywords, frequencies = zip(*top_keywords)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(keywords)), frequencies, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(keywords)))
        ax.set_yticklabels(keywords)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(f'Top {top_n} Keywords in Smart Farm Technology Discussions', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (keyword, freq) in enumerate(top_keywords):
            ax.text(freq + 0.5, i, str(freq), va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Top keywords visualization saved to: {output_file}")
        plt.close()
    
    def visualize_centrality(self, results, output_file='keyword_centrality.png'):
        """Visualize keyword centrality measures"""
        print("\nGenerating centrality visualization...")
        
        top_degree = results['network_statistics']['top_degree_centrality'][:15]
        top_betweenness = results['network_statistics']['top_betweenness_centrality'][:15]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Degree centrality
        keywords1, scores1 = zip(*top_degree)
        ax1.barh(range(len(keywords1)), scores1, color='coral', alpha=0.8)
        ax1.set_yticks(range(len(keywords1)))
        ax1.set_yticklabels(keywords1)
        ax1.set_xlabel('Degree Centrality', fontsize=11)
        ax1.set_title('Most Connected Keywords', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        # Betweenness centrality
        keywords2, scores2 = zip(*top_betweenness)
        ax2.barh(range(len(keywords2)), scores2, color='mediumseagreen', alpha=0.8)
        ax2.set_yticks(range(len(keywords2)))
        ax2.set_yticklabels(keywords2)
        ax2.set_xlabel('Betweenness Centrality', fontsize=11)
        ax2.set_title('Most Bridging Keywords', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        
        plt.suptitle('Keyword Influence Measures', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Centrality visualization saved to: {output_file}")
        plt.close()
    
    def visualize_community_sentiments(self, results, output_file='community_sentiments.png'):
        """Visualize sentiment distribution across communities"""
        print("\nGenerating community sentiment visualization...")
        
        communities_data = results['communities']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        labels = [f"C{c['community_id']}\n{c['topic_label'][:20]}..." for c in communities_data]
        positive = [c['sentiment_percentages']['positive'] for c in communities_data]
        neutral = [c['sentiment_percentages']['neutral'] for c in communities_data]
        negative = [c['sentiment_percentages']['negative'] for c in communities_data]
        
        x = range(len(labels))
        width = 0.6
        
        # Stacked bar chart
        p1 = ax.bar(x, positive, width, label='Positive', color='#2ecc71', alpha=0.8)
        p2 = ax.bar(x, neutral, width, bottom=positive, label='Neutral', color='#95a5a6', alpha=0.8)
        p3 = ax.bar(x, negative, width, bottom=[i+j for i,j in zip(positive, neutral)], 
                   label='Negative', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title('Sentiment Distribution Across Topic Communities', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Community sentiment visualization saved to: {output_file}")
        plt.close()
    
    def visualize_individual_communities(self, results, output_prefix='community'):
        """Create separate clean visualizations for each community"""
        print("\nGenerating individual community visualizations...")
        
        viz_dir = 'visualizations'
        
        for i, comm_data in enumerate(results['communities'], 1):
            # Get top 15 keywords from this community
            top_keywords = [k for k, _ in comm_data['top_keywords'][:15]]
            
            # Filter to nodes that exist in graph
            comm_nodes = [n for n in top_keywords if n in self.graph.nodes()]
            
            if len(comm_nodes) < 3:
                continue
            
            # Create subgraph
            subgraph = self.graph.subgraph(comm_nodes).copy()
            
            # Layout
            pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 9))
            
            # Node sizes based on frequency
            node_sizes = [results['keyword_frequencies'].get(node, 1) * 100 + 500 
                         for node in subgraph.nodes()]
            
            # Draw
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=2, edge_color='gray', ax=ax)
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                                  node_color='lightcoral', alpha=0.85,
                                  edgecolors='darkred', linewidths=2, ax=ax)
            nx.draw_networkx_labels(subgraph, pos, font_size=12, font_weight='bold', ax=ax)
            
            # Title with topic info
            title = f"Community {i}: {comm_data['topic_label']}\n"
            title += f"{comm_data['size']} keywords (showing top {len(comm_nodes)}), {subgraph.number_of_edges()} connections"
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            output_file = f'{viz_dir}/{output_prefix}_{i}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  Community {i} saved to: {output_file}")
            plt.close()

    
    def visualize_sentiment_network(self, results, output_file='network_sentiment.png', figsize=(18, 14)):
        """
        Visualize network with nodes colored by dominant sentiment
        Standard in social media analysis - shows sentiment clustering
        """
        print("\nGenerating sentiment-colored network visualization...")
        
        if self.graph.number_of_nodes() == 0:
            print("Empty graph. Cannot visualize.")
            return
        
        # Get top nodes
        degree_dict = dict(self.graph.degree())
        top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:80]
        top_node_names = [node for node, _ in top_nodes]
        subgraph = self.graph.subgraph(top_node_names).copy()
        
        print(f"  Showing {subgraph.number_of_nodes()} nodes with sentiment colors")
        
        # Calculate dominant sentiment for each keyword
        keyword_sentiment = {}
        for node in subgraph.nodes():
            node_attrs = self.graph.nodes[node]
            if 'sentiment' in node_attrs:
                sent_dist = node_attrs['sentiment']
                # Find dominant sentiment
                if sent_dist['positive'] > sent_dist['negative'] and sent_dist['positive'] > sent_dist['neutral']:
                    keyword_sentiment[node] = 'positive'
                elif sent_dist['negative'] > sent_dist['positive'] and sent_dist['negative'] > sent_dist['neutral']:
                    keyword_sentiment[node] = 'negative'
                else:
                    keyword_sentiment[node] = 'neutral'
            else:
                keyword_sentiment[node] = 'neutral'
        
        # Color mapping
        sentiment_colors = {
            'positive': '#2ecc71',  # Green
            'negative': '#e74c3c',  # Red
            'neutral': '#95a5a6'    # Gray
        }
        
        node_colors = [sentiment_colors.get(keyword_sentiment.get(node, 'neutral'), '#95a5a6') 
                      for node in subgraph.nodes()]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(subgraph, k=2.5, iterations=100, seed=42)
        
        # Node sizes
        node_sizes = [degree_dict[node] * 40 + 400 for node in subgraph.nodes()]
        
        # Draw edges
        edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [w / max_weight * 3.5 for w in edge_weights]
        else:
            edge_widths = [1] * len(edge_weights)
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.7, width=edge_widths, 
                              edge_color='#2c3e50', ax=ax)
        
        # Draw nodes with sentiment colors
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.9, 
                              edgecolors='black', linewidths=1.5, ax=ax)
        
        nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold', ax=ax)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Positive Sentiment'),
            mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Negative Sentiment'),
            mpatches.Patch(facecolor='#95a5a6', edgecolor='black', label='Neutral Sentiment')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        
        ax.set_title('Keyword Network Colored by Dominant Sentiment\n' + 
                    'Green=Positive | Red=Negative | Gray=Neutral',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Sentiment network visualization saved to: {output_file}")
        plt.close()

    
    def generate_all_visualizations(self, results):
        """Generate all visualization outputs"""
        viz_dir = 'visualizations'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        print(f"\nGenerating {viz_dir}/...")
        self.visualize_full_network(output_file=f'{viz_dir}/network_full.png')
        self.visualize_sentiment_network(results, output_file=f'{viz_dir}/network_sentiment.png')
        self.visualize_communities(results, output_file=f'{viz_dir}/network_communities.png')
        self.visualize_top_keywords(results, output_file=f'{viz_dir}/top_keywords.png')
        self.visualize_centrality(results, output_file=f'{viz_dir}/keyword_centrality.png')
        self.visualize_community_sentiments(results, output_file=f'{viz_dir}/community_sentiments.png')
        self.visualize_individual_communities(results, output_prefix='community')
        print("Done!")


def main():
    """Main execution function"""
    print("=" * 70)
    print("NETWORK VISUALIZATION")
    print("=" * 70)
    
    if not os.path.exists('keyword_network.pkl') or not os.path.exists('network_analysis_results.json'):
        print("\nERROR: Network files not found. Run network_analysis.py first.")
        return
    
    visualizer = NetworkVisualizer()
    visualizer.load_network()
    results = visualizer.load_analysis_results()
    visualizer.generate_all_visualizations(results)
    
    print("\n" + "=" * 70)
    print("Check visualizations/ directory")
    print("=" * 70)


if __name__ == "__main__":
    main()
