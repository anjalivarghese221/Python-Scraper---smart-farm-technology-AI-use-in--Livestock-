# Smart Farm Technology & AI in Livestock Analysis

Reddit sentiment analysis for smart farming technology and AI use in livestock management (2018-2026).

## Overview

**Peer-Reviewed Research Pipeline:**
1. **Data Collection** - Reddit posts from 2018+ (2,811 posts from 1,113 subreddits)
2. **Preprocessing** - Phase 2 compliant cleaning (86.4% retention)
3. **Sentiment Analysis** - 3-class classification (positive/negative/neutral)
4. **Network Analysis** - 4 agriculture-focused topic communities
5. **Temporal Analysis** - Trend detection and spike identification
6. **Statistical Analysis** - Odds ratios and probability distributions

## Features

### Phase 1: Data Extraction (2018+)
- Reddit JSON API with 2018+ temporal filtering
- Multiple query strategies (lexical expansion + targeted phrase queries + subreddit-restricted passes)
- Query optimization for livestock-emissions relevance (e.g., dairy/livestock + methane/emission)
- Reddit-only source collection (no BlueSky/news for this phase)
- Privacy compliant (no usernames/IDs collected)
- 8-year temporal coverage (2018-2026)

### Phase 2: Preprocessing
- Language detection and filtering
- MD5 hash deduplication (7% attrition)
- Length filtering (≥5 words)
- Lemmatization with SpaCy
- 86.4% retention rate (2,811 final posts)

### Sentiment Classification
- Logistic Regression with TF-IDF (150 training examples)
- 48.1% positive, 35.6% negative, 16.4% neutral
- Odds ratio: 1.35:1 (positive:negative)

### Topic Modeling
- 4 agriculture-focused communities (no mixed topics)
- Community-level sentiment analysis
- Keyword co-occurrence networks (905 keywords, 3,340 edges)

### Temporal Analysis
- Monthly trends (2018-2026, 96 months)
- Quarterly sentiment distribution (line graphs)
- Spike detection (2025: 1,179 posts, 472% growth)
- 22× volume increase over 8 years

## Installation

```bash
pip3 install -r requirements.txt
```

## Usage

**Complete Analysis Pipeline:**
```bash
# Step 1: Data Collection (Reddit-only, multi-query strategies)
python3 simple_collector.py

# Step 2: Preprocessing
python3 enhanced_preprocessing.py

# Step 3: Sentiment Classification
python3 sentiment_classifier.py

# Optional alternatives:
# - RoBERTa classifier output: classified_sentiment_data_roberta.json
python3 sentiment_classifier_roberta.py
# - VADER classifier output: classified_sentiment_data_vader.json
python3 sentiment_classifier_vader.py

# Step 4: Phase 1 quality checks
# - keywords by positive/negative/neutral labels
# - manual review sample (20 posts)
python3 phase1_quality_check.py

# Step 5: Network Analysis
python3 network_analysis.py

# Step 6: Temporal Analysis
python3 temporal_analysis.py

# Step 7: Generate Visualizations
python3 network_visualizer.py

# Step 8: Final Statistics
python3 generate_statistics.py

# Step 9: Policy Milestone Hypothesis Test (Before vs After 2024)
python3 policy_milestone_hypothesis.py

# Step 10: Topic-Specific Hypotheses (Livestock vs Non-livestock; Pre-surge vs Surge)
python3 smart_farming_hypothesis.py

# Step 11: Region-Proxy Hypothesis Test
python3 regional_hypothesis_proxy.py

# Step 12: Robustness & Sensitivity Checks
python3 robustness_sensitivity_analysis.py

# One-command variant pipelines:
python3 run_full_pipeline_roberta.py
python3 run_full_pipeline_vader.py
```

## Project Structure

```
├── simple_collector.py           # Data collection (2018+ Reddit API)
├── enhanced_preprocessing.py     # Phase 2 cleaning pipeline
├── sentiment_classifier.py       # 3-class sentiment classification
├── phase1_quality_check.py       # Keyword extraction + manual review sample (n=20)
├── network_analysis.py           # Community detection (4 topics)
├── temporal_analysis.py          # Trend and spike detection
├── network_visualizer.py         # Generate all visualizations
├── generate_statistics.py        # Final statistics report
├── requirements.txt              # Dependencies
├── PHASE1_DATA_EXTRACTION_DOCUMENTATION.md
├── PHASE2_PREPROCESSING_DOCUMENTATION.md
├── PRIVACY_COMPLIANCE_VERIFICATION.md
├── REQUIREMENT_3_ANALYSIS_SUMMARY.md
├── PROJECT_SUMMARY.md
└── visualizations/               # 23 output graphs
```

## Output Files

**Data:**
- `enhanced_scraped_data.json` - Collected posts (2018+)
- `preprocessed_data.json` - Cleaned text (Phase 2)
- `sentiment_training_data.json` - 150 labeled training examples
- `classified_sentiment_data.json` - Final analyzed dataset (2,811 posts)
- `temporal_analysis_results.json` - Monthly/quarterly trends
- `network_analysis_results.json` - 4 communities with sentiment
- `final_statistics.json` - Comprehensive statistics

**Models:**
- `sentiment_model.pkl` - Trained Logistic Regression classifier
- `vectorizer.pkl` - TF-IDF vectorizer
- `keyword_network.pkl` - NetworkX graph object
- `vectorizer.pkl` - TF-IDF vectorizer
- `keyword_network.pkl` - Network graph

**Reports:**
- `statistics.json` - Word frequencies
- `network_analysis_results.json` - Network statistics
- Text reports auto-generated by each step

**Visualizations:**
- `visualizations/full_network.png` - Complete keyword network
- `visualizations/communities.png` - Topic communities
- `visualizations/community_*.png` - Individual communities
- `visualizations/sentiment_overview.png` - 4-panel analysis

##  Requirements

- Python 3.9+
- requests
- scikit-learn
- pandas
- numpy
- networkx
- matplotlib

##  Notes

- Uses public Reddit JSON API (no auth needed)
- Includes rate limiting (2s delays)
- Stop words filtered for cleaner network analysis
- All generated reports/data excluded from git

---

**Educational project for analyzing agricultural technology discussions.**
