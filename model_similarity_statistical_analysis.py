#!/usr/bin/env python3
"""
Statistical similarity analysis comparing VADER vs Logistic Regression TF-IDF vs RoBERTa.
Computes correlation, Cohen's kappa, confusion matrices, and confidence distributions.
"""

import json
import os
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
LOGISTIC_FILE = 'classified_sentiment_data.json'
ROBERTA_FILE = 'classified_sentiment_data_roberta.json'
VADER_FILE = 'classified_sentiment_data_vader.json'
OUTPUT_JSON = 'model_similarity_statistical_analysis.json'
OUTPUT_REPORT = 'model_similarity_statistical_report.txt'
VIZ_DIR = 'visualizations/model_comparison'


def ensure_viz_dir():
    """Create visualization directory if it doesn't exist."""
    os.makedirs(VIZ_DIR, exist_ok=True)


def load_json(path):
    """Load JSON file safely."""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def encode_sentiment(sentiment_str):
    """Convert sentiment string to numeric (0=negative, 1=neutral, 2=positive)."""
    s = (sentiment_str or '').lower().strip()
    if s == 'negative':
        return 0
    elif s == 'neutral':
        return 1
    elif s == 'positive':
        return 2
    return np.nan


def align_and_extract(data_a, data_b):
    """
    Align two datasets and extract sentiment + confidence for paired analysis.
    Returns lists aligned by URL (assumed unique identifier).
    """
    if data_a is None or data_b is None or len(data_a) == 0 or len(data_b) == 0:
        return None, None, None, None
    
    # Build URL -> index maps
    url_to_idx_a = {r.get('url'): i for i, r in enumerate(data_a)}
    url_to_idx_b = {r.get('url'): i for i, r in enumerate(data_b)}
    
    # Find common URLs
    common_urls = set(url_to_idx_a.keys()) & set(url_to_idx_b.keys())
    if len(common_urls) == 0:
        return None, None, None, None
    
    sentiments_a, sentiments_b = [], []
    confidences_a, confidences_b = [], []
    
    for url in common_urls:
        s_a = data_a[url_to_idx_a[url]].get('sentiment')
        c_a = data_a[url_to_idx_a[url]].get('sentiment_confidence', np.nan)
        s_b = data_b[url_to_idx_b[url]].get('sentiment')
        c_b = data_b[url_to_idx_b[url]].get('sentiment_confidence', np.nan)
        
        if s_a and s_b:
            sentiments_a.append(encode_sentiment(s_a))
            sentiments_b.append(encode_sentiment(s_b))
            confidences_a.append(float(c_a) if not np.isnan(c_a) else 0.5)
            confidences_b.append(float(c_b) if not np.isnan(c_b) else 0.5)
    
    return (np.array(sentiments_a), np.array(sentiments_b),
            np.array(confidences_a), np.array(confidences_b))


def compute_pairwise_metrics(sentiments_a, sentiments_b, label_a, label_b):
    """
    Compute comprehensive pairwise similarity metrics.
    Returns dict with correlation, kappa, matthews, agreement, and confusion matrix.
    """
    if sentiments_a is None or sentiments_b is None:
        return None
    
    n = len(sentiments_a)
    
    # Basic agreement
    exact_agreement = np.sum(sentiments_a == sentiments_b) / n
    
    # Cohen's Kappa (accounts for chance agreement)
    kappa = cohen_kappa_score(sentiments_a, sentiments_b)
    
    # Matthews Correlation Coefficient (balanced for all classes)
    mcc = matthews_corrcoef(sentiments_a, sentiments_b)
    
    # Spearman rank correlation (non-parametric)
    spearman_rho, spearman_p = spearmanr(sentiments_a, sentiments_b)
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(sentiments_a, sentiments_b)
    
    # Confusion matrix
    cm = confusion_matrix(sentiments_a, sentiments_b, labels=[0, 1, 2])
    
    # Per-class agreement (what % of each sentiment agrees)
    per_class_agreement = {}
    for sentiment_val, label in [(0, 'negative'), (1, 'neutral'), (2, 'positive')]:
        mask = sentiments_a == sentiment_val
        if mask.sum() > 0:
            agreement = np.sum((sentiments_a == sentiments_b) & mask) / mask.sum()
            per_class_agreement[label] = {
                'count_in_a': int(mask.sum()),
                'agreement_pct': float(agreement * 100)
            }
    
    return {
        'sample_size': int(n),
        'pair': f'{label_a} vs {label_b}',
        'exact_agreement_pct': float(exact_agreement * 100),
        'cohens_kappa': float(kappa),
        'matthews_corrcoef': float(mcc),
        'spearman_rho': float(spearman_rho),
        'spearman_p_value': float(spearman_p),
        'pearson_r': float(pearson_r),
        'pearson_p_value': float(pearson_p),
        'per_class_agreement': per_class_agreement,
        'confusion_matrix': {
            'labels': ['negative', 'neutral', 'positive'],
            'matrix': cm.tolist()
        }
    }


def compute_confidence_analysis(confidences_a, confidences_b, sentiments_a, sentiments_b, label_a, label_b):
    """
    Analyze confidence scores: do high-confidence predictions agree more?
    """
    if confidences_a is None or confidences_b is None:
        return None
    
    # Threshold-based agreement
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    confidence_analysis = {}
    
    for threshold in thresholds:
        high_conf_a = confidences_a >= threshold
        high_conf_b = confidences_b >= threshold
        both_high_conf = high_conf_a & high_conf_b
        
        if both_high_conf.sum() > 0:
            agreement = np.sum(sentiments_a[both_high_conf] == sentiments_b[both_high_conf]) / both_high_conf.sum()
            confidence_analysis[f'both_>={threshold}'] = {
                'sample_size': int(both_high_conf.sum()),
                'agreement_pct': float(agreement * 100)
            }
    
    # Mean confidences
    return {
        'mean_confidence_a': float(confidences_a.mean()),
        'mean_confidence_b': float(confidences_b.mean()),
        'std_confidence_a': float(confidences_a.std()),
        'std_confidence_b': float(confidences_b.std()),
        'confidence_threshold_analysis': confidence_analysis
    }


def create_comparison_visualization(data_a, data_b, data_c, label_a='Logistic', label_b='RoBERTa', label_c='VADER'):
    """Create heatmaps and agreement visualizations."""
    ensure_viz_dir()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Similarity: Confusion Matrices (Sentiment Agreement)', fontsize=14, fontweight='bold')
    
    pairs = [
        (data_a, data_b, label_a, label_b, axes[0]),
        (data_a, data_c, label_a, label_c, axes[1]),
        (data_b, data_c, label_b, label_c, axes[2])
    ]
    
    labels = ['negative', 'neutral', 'positive']
    
    for sentiments_x, sentiments_y, label_x, label_y, ax in pairs:
        if sentiments_x is not None and sentiments_y is not None:
            cm = confusion_matrix(sentiments_x, sentiments_y, labels=[0, 1, 2])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       xticklabels=labels, yticklabels=labels, cbar=True)
            ax.set_title(f'{label_x} vs {label_y}')
            ax.set_ylabel('Actual (True Model)')
            ax.set_xlabel('Predicted (Comparison Model)')
    
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR}/confusion_matrices_comparison.png")
    plt.close()


def create_confidence_visualization(confidences_a, confidences_b, confidences_c, 
                                    label_a='Logistic', label_b='RoBERTa', label_c='VADER'):
    """Create confidence distribution comparisons."""
    ensure_viz_dir()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confidence Score Distributions by Model', fontsize=14, fontweight='bold')
    
    data_tuples = [
        (confidences_a, label_a, axes[0]),
        (confidences_b, label_b, axes[1]),
        (confidences_c, label_c, axes[2])
    ]
    
    for confidences, label, ax in data_tuples:
        if confidences is not None:
            ax.hist(confidences, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(confidences.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidences.mean():.3f}')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title(label)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/confidence_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR}/confidence_distributions.png")
    plt.close()


def main():
    print("\n" + "="*70)
    print("MODEL SIMILARITY STATISTICAL ANALYSIS: VADER vs Logistic vs RoBERTa")
    print("="*70 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    logistic = load_json(LOGISTIC_FILE)
    roberta = load_json(ROBERTA_FILE)
    vader = load_json(VADER_FILE)
    
    if not all([logistic, roberta, vader]):
        print("ERROR: All three model files required for analysis.")
        return
    
    print(f"  ✓ Logistic: {len(logistic)} records")
    print(f"  ✓ RoBERTa: {len(roberta)} records")
    print(f"  ✓ VADER: {len(vader)} records\n")
    
    # Pairwise alignments
    print("Aligning datasets for pairwise comparison...")
    log_sent, rob_sent, log_conf, rob_conf = align_and_extract(logistic, roberta)
    log_sent_v, vader_sent, log_conf_v, vader_conf = align_and_extract(logistic, vader)
    rob_sent_v, vader_sent_v, rob_conf_v, vader_conf_v = align_and_extract(roberta, vader)
    
    print(f"  ✓ Logistic ↔ RoBERTa: {len(log_sent) if log_sent is not None else 0} aligned")
    print(f"  ✓ Logistic ↔ VADER: {len(log_sent_v) if log_sent_v is not None else 0} aligned")
    print(f"  ✓ RoBERTa ↔ VADER: {len(rob_sent_v) if rob_sent_v is not None else 0} aligned\n")
    
    # Compute pairwise metrics
    print("Computing pairwise similarity metrics...\n")
    metrics_log_rob = compute_pairwise_metrics(log_sent, rob_sent, 'Logistic', 'RoBERTa')
    metrics_log_vader = compute_pairwise_metrics(log_sent_v, vader_sent, 'Logistic', 'VADER')
    metrics_rob_vader = compute_pairwise_metrics(rob_sent_v, vader_sent_v, 'RoBERTa', 'VADER')
    
    # Confidence analysis
    print("Analyzing confidence scores...\n")
    conf_analysis_log_rob = compute_confidence_analysis(log_conf, rob_conf, log_sent, rob_sent, 'Logistic', 'RoBERTa')
    conf_analysis_log_vader = compute_confidence_analysis(log_conf_v, vader_conf, log_sent_v, vader_sent, 'Logistic', 'VADER')
    conf_analysis_rob_vader = compute_confidence_analysis(rob_conf_v, vader_conf_v, rob_sent_v, vader_sent_v, 'RoBERTa', 'VADER')
    
    # Compile results
    results = {
        'metadata': {
            'analysis_date': '2026-04-22',
            'models': ['Logistic Regression (TF-IDF)', 'RoBERTa (Transformer)', 'VADER (Lexicon)'],
            'description': 'Statistical similarity analysis comparing three sentiment classification models'
        },
        'pairwise_metrics': {
            'logistic_vs_roberta': metrics_log_rob,
            'logistic_vs_vader': metrics_log_vader,
            'roberta_vs_vader': metrics_rob_vader
        },
        'confidence_analysis': {
            'logistic_vs_roberta': conf_analysis_log_rob,
            'logistic_vs_vader': conf_analysis_log_vader,
            'roberta_vs_vader': conf_analysis_rob_vader
        }
    }
    
    # Save JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {OUTPUT_JSON}\n")
    
    # Generate report
    print("\n" + "="*70)
    print("GENERATING TEXT REPORT")
    print("="*70 + "\n")
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MODEL SIMILARITY STATISTICAL ANALYSIS\n")
        f.write("Comparing: Logistic Regression (TF-IDF) vs RoBERTa vs VADER\n")
        f.write("="*70 + "\n\n")
        
        f.write("INTERPRETATION GUIDE:\n")
        f.write("-" * 70 + "\n")
        f.write("• Cohen's Kappa: Measures agreement beyond chance (0-1 scale)\n")
        f.write("  - 0.0-0.2: Slight agreement\n")
        f.write("  - 0.2-0.4: Fair agreement\n")
        f.write("  - 0.4-0.6: Moderate agreement\n")
        f.write("  - 0.6-0.8: Substantial agreement\n")
        f.write("  - 0.8-1.0: Almost perfect agreement\n\n")
        
        f.write("• Pearson/Spearman Correlation: Relationship between predictions (-1 to +1)\n")
        f.write("  - High positive: Models move together\n")
        f.write("  - Near zero: Models operate independently\n\n")
        
        f.write("• Matthews Correlation Coefficient: Balanced measure for multi-class\n")
        f.write("  - Similar interpretation to Kappa\n\n")
        
        f.write("• Confusion Matrix: Shows misclassification patterns\n\n")
        
        f.write("="*70 + "\n")
        f.write("1. LOGISTIC REGRESSION (TF-IDF) vs RoBERTa\n")
        f.write("="*70 + "\n\n")
        
        if metrics_log_rob:
            m = metrics_log_rob
            f.write(f"Sample Size: {m['sample_size']} aligned posts\n\n")
            f.write(f"AGREEMENT METRICS:\n")
            f.write(f"  Exact Agreement: {m['exact_agreement_pct']:.2f}%\n")
            f.write(f"  Cohen's Kappa: {m['cohens_kappa']:.4f}\n")
            f.write(f"  Matthews Corrcoef: {m['matthews_corrcoef']:.4f}\n")
            f.write(f"  Pearson r: {m['pearson_r']:.4f} (p={m['pearson_p_value']:.4e})\n")
            f.write(f"  Spearman ρ: {m['spearman_rho']:.4f} (p={m['spearman_p_value']:.4e})\n\n")
            
            f.write(f"PER-CLASS AGREEMENT:\n")
            for sentiment, stats in m['per_class_agreement'].items():
                f.write(f"  {sentiment.capitalize()}: {stats['count_in_a']} posts, {stats['agreement_pct']:.2f}% agree\n")
            
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write(f"                 Predicted RoBERTa\n")
            f.write(f"                 Neg    Neu    Pos\n")
            cm = m['confusion_matrix']['matrix']
            labels_short = ['Neg', 'Neu', 'Pos']
            for i, label in enumerate(labels_short):
                f.write(f"  Logistic {label}  {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}\n")
            f.write("\n")
            
            if conf_analysis_log_rob:
                ca = conf_analysis_log_rob
                f.write(f"CONFIDENCE ANALYSIS:\n")
                f.write(f"  Logistic mean confidence: {ca['mean_confidence_a']:.4f}\n")
                f.write(f"  RoBERTa mean confidence: {ca['mean_confidence_b']:.4f}\n")
                f.write(f"\n  High-Confidence Agreement:\n")
                for threshold, stats in ca['confidence_threshold_analysis'].items():
                    f.write(f"    {threshold}: {stats['sample_size']} posts, {stats['agreement_pct']:.2f}% agree\n")
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("2. LOGISTIC REGRESSION (TF-IDF) vs VADER\n")
        f.write("="*70 + "\n\n")
        
        if metrics_log_vader:
            m = metrics_log_vader
            f.write(f"Sample Size: {m['sample_size']} aligned posts\n\n")
            f.write(f"AGREEMENT METRICS:\n")
            f.write(f"  Exact Agreement: {m['exact_agreement_pct']:.2f}%\n")
            f.write(f"  Cohen's Kappa: {m['cohens_kappa']:.4f}\n")
            f.write(f"  Matthews Corrcoef: {m['matthews_corrcoef']:.4f}\n")
            f.write(f"  Pearson r: {m['pearson_r']:.4f} (p={m['pearson_p_value']:.4e})\n")
            f.write(f"  Spearman ρ: {m['spearman_rho']:.4f} (p={m['spearman_p_value']:.4e})\n\n")
            
            f.write(f"PER-CLASS AGREEMENT:\n")
            for sentiment, stats in m['per_class_agreement'].items():
                f.write(f"  {sentiment.capitalize()}: {stats['count_in_a']} posts, {stats['agreement_pct']:.2f}% agree\n")
            
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write(f"                 Predicted VADER\n")
            f.write(f"                 Neg    Neu    Pos\n")
            cm = m['confusion_matrix']['matrix']
            labels_short = ['Neg', 'Neu', 'Pos']
            for i, label in enumerate(labels_short):
                f.write(f"  Logistic {label}  {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}\n")
            f.write("\n")
            
            if conf_analysis_log_vader:
                ca = conf_analysis_log_vader
                f.write(f"CONFIDENCE ANALYSIS:\n")
                f.write(f"  Logistic mean confidence: {ca['mean_confidence_a']:.4f}\n")
                f.write(f"  VADER mean confidence: {ca['mean_confidence_b']:.4f}\n")
                f.write(f"\n  High-Confidence Agreement:\n")
                for threshold, stats in ca['confidence_threshold_analysis'].items():
                    f.write(f"    {threshold}: {stats['sample_size']} posts, {stats['agreement_pct']:.2f}% agree\n")
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("3. RoBERTa vs VADER\n")
        f.write("="*70 + "\n\n")
        
        if metrics_rob_vader:
            m = metrics_rob_vader
            f.write(f"Sample Size: {m['sample_size']} aligned posts\n\n")
            f.write(f"AGREEMENT METRICS:\n")
            f.write(f"  Exact Agreement: {m['exact_agreement_pct']:.2f}%\n")
            f.write(f"  Cohen's Kappa: {m['cohens_kappa']:.4f}\n")
            f.write(f"  Matthews Corrcoef: {m['matthews_corrcoef']:.4f}\n")
            f.write(f"  Pearson r: {m['pearson_r']:.4f} (p={m['pearson_p_value']:.4e})\n")
            f.write(f"  Spearman ρ: {m['spearman_rho']:.4f} (p={m['spearman_p_value']:.4e})\n\n")
            
            f.write(f"PER-CLASS AGREEMENT:\n")
            for sentiment, stats in m['per_class_agreement'].items():
                f.write(f"  {sentiment.capitalize()}: {stats['count_in_a']} posts, {stats['agreement_pct']:.2f}% agree\n")
            
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write(f"                 Predicted VADER\n")
            f.write(f"                 Neg    Neu    Pos\n")
            cm = m['confusion_matrix']['matrix']
            labels_short = ['Neg', 'Neu', 'Pos']
            for i, label in enumerate(labels_short):
                f.write(f"  RoBERTa {label}   {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}\n")
            f.write("\n")
            
            if conf_analysis_rob_vader:
                ca = conf_analysis_rob_vader
                f.write(f"CONFIDENCE ANALYSIS:\n")
                f.write(f"  RoBERTa mean confidence: {ca['mean_confidence_a']:.4f}\n")
                f.write(f"  VADER mean confidence: {ca['mean_confidence_b']:.4f}\n")
                f.write(f"\n  High-Confidence Agreement:\n")
                for threshold, stats in ca['confidence_threshold_analysis'].items():
                    f.write(f"    {threshold}: {stats['sample_size']} posts, {stats['agreement_pct']:.2f}% agree\n")
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL SIMILARITY INTERPRETATION:\n\n")
        
        if all([metrics_log_rob, metrics_log_vader, metrics_rob_vader]):
            # Find strongest agreement
            agreements = [
                ('Logistic vs RoBERTa', metrics_log_rob['cohens_kappa']),
                ('Logistic vs VADER', metrics_log_vader['cohens_kappa']),
                ('RoBERTa vs VADER', metrics_rob_vader['cohens_kappa'])
            ]
            agreements_sorted = sorted(agreements, key=lambda x: x[1], reverse=True)
            
            f.write(f"1. Strongest Agreement: {agreements_sorted[0][0]} (κ={agreements_sorted[0][1]:.4f})\n")
            f.write(f"2. Middle Agreement: {agreements_sorted[1][0]} (κ={agreements_sorted[1][1]:.4f})\n")
            f.write(f"3. Lowest Agreement: {agreements_sorted[2][0]} (κ={agreements_sorted[2][1]:.4f})\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("• Model disagreement suggests each captures different aspects of sentiment\n")
            f.write("• Use ensemble methods (voting/averaging) for more robust predictions\n")
            f.write("• High-confidence predictions from all models provide highest reliability\n")
            f.write("• Consider using model disagreement as a signal of uncertain/nuanced posts\n")
    
    print(f"✓ Saved: {OUTPUT_REPORT}\n")
    
    # Create visualizations
    print("Creating visualizations...\n")
    create_comparison_visualization(log_sent, rob_sent, vader_sent, 'Logistic', 'RoBERTa', 'VADER')
    create_confidence_visualization(log_conf, rob_conf, vader_conf, 'Logistic', 'RoBERTa', 'VADER')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  • JSON Results: {OUTPUT_JSON}")
    print(f"  • Text Report: {OUTPUT_REPORT}")
    print(f"  • Visualizations: {VIZ_DIR}/")


if __name__ == '__main__':
    main()
