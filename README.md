# Smart Farm Technology & AI in Livestock Analysis

Reddit sentiment analysis for smart farming technology and AI use in livestock management (2018-2026).

## Overview

**Peer-Reviewed Research Pipeline:**
1. **Data Collection** - Reddit posts from 2018+ (Reddit-only workflow)
2. **Preprocessing** - Phase 2 compliant cleaning + attrition reporting
3. **Sentiment Analysis** - 3-class classification (positive/negative/neutral) with multiple model options
4. **Network Analysis** - 4 agriculture-focused topic communities
5. **Temporal Analysis** - Trend detection and spike identification
6. **Statistical Analysis** - Odds ratios and probability distributions
7. **Model Comparison** - Logistic vs RoBERTa vs VADER agreement and label shifts

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
- Logistic Regression with TF-IDF
- RoBERTa transformer classifier
- VADER lexicon-based classifier
- Pairwise agreement reporting across models

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

**Full baseline pipeline (logistic):**
```bash
python3 run_full_pipeline.py
```

**Full alternative pipelines:**
```bash
python3 run_full_pipeline_roberta.py
python3 run_full_pipeline_vader.py
```

**Step-by-step pipeline:**
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

# Optional: compare model outputs
python3 compare_sentiment_models.py

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

# Optional: generate model-specific temporal figures (logistic/roberta/vader)
python3 generate_model_temporal_visualizations.py
```

## Project Structure

```
├── simple_collector.py           # Data collection (2018+ Reddit API)
├── enhanced_preprocessing.py     # Phase 2 cleaning pipeline
├── sentiment_classifier.py       # 3-class sentiment classification
├── sentiment_classifier_roberta.py
├── sentiment_classifier_vader.py
├── compare_sentiment_models.py   # Logistic vs RoBERTa vs VADER comparison
├── phase1_quality_check.py       # Keyword extraction + manual review sample (n=20)
├── network_analysis.py           # Community detection (4 topics)
├── temporal_analysis.py          # Trend and spike detection
├── network_visualizer.py         # Generate all visualizations
├── run_full_pipeline.py          # End-to-end logistic pipeline
├── run_full_pipeline_roberta.py  # End-to-end RoBERTa pipeline
├── run_full_pipeline_vader.py    # End-to-end VADER pipeline
├── generate_model_temporal_visualizations.py
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
- `classified_sentiment_data.json` - Logistic classified dataset
- `classified_sentiment_data_roberta.json` - RoBERTa classified dataset
- `classified_sentiment_data_vader.json` - VADER classified dataset
- `temporal_analysis_results.json` - Monthly/quarterly trends
- `network_analysis_results.json` - 4 communities with sentiment
- `final_statistics.json` - Comprehensive statistics

**Models:**
- `sentiment_model.pkl` - Trained Logistic Regression classifier
- `vectorizer.pkl` - TF-IDF vectorizer
- `keyword_network.pkl` - NetworkX graph object

**Comparison:**
- `sentiment_model_comparison.json` - Structured comparison metrics across available models
- `sentiment_model_comparison.txt` - Human-readable comparison report

**Reports:**
- `network_analysis_results.json` - Network statistics
- Text reports auto-generated by each step

**Visualizations:**
- `visualizations/network_full.png` - Complete keyword network
- `visualizations/network_communities.png` - Topic communities
- `visualizations/community_*.png` - Individual communities
- `visualizations/sentiment_overview.png` - 4-panel analysis
- `visualizations/logistic/` - Snapshot visualizations for logistic runs
- `visualizations/roberta/` - Snapshot visualizations for RoBERTa runs
- `visualizations/vader/` - Snapshot visualizations for VADER runs

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
