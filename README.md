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

### Sentiment Classification (Multi-Model)
- Logistic Regression with TF-IDF (baseline)
- RoBERTa transformer classifier (primary)
- VADER lexicon-based classifier (fallback)
- ClimateBERT domain-specialized model (optional)
- Pairwise agreement reporting across models
- Confidence scores and model comparison metrics

### Topic Modeling & Validation
- Latent Dirichlet Allocation (LDA) with 10 topics
- Coherence metrics (Cv metric for topic quality)
- Stability analysis (cross-validation robustness)
- 4 agriculture-focused communities with sentiment analysis
- Keyword co-occurrence networks (905 keywords, 3,340 edges)

### Network Analysis
- Document similarity graph (cosine distance)
- Louvain community detection
- Centrality metrics (betweenness, closeness, degree)
- Community-level sentiment aggregation
- Interactive network visualization

### Temporal Analysis
- Monthly trends (2018-2026, 96 months)
- Quarterly sentiment distribution (line graphs)
- Spike detection (2025: 1,179 posts, 472% growth)
- Policy milestone correlation testing
- 22× volume increase over 8 years

### Hypothesis-Driven Validation
- Policy milestone hypothesis (policy changes correlate with sentiment shifts)
- Smart farming hypothesis (AI/automation topics drive positive sentiment)
- Regional pattern hypothesis (network community structure reflects topic patterns)
- Robustness & sensitivity analysis (resampling validation)

## Installation

```bash
pip3 install -r requirements.txt
```

## Usage

**Full automated pipelines:**
```bash
# Run complete RoBERTa pipeline (17 steps, default)
python3 run_full_pipeline_roberta.py

# Run complete Logistic pipeline (baseline comparison)
python3 run_full_pipeline.py

# Run complete VADER pipeline (alternative comparison)
python3 run_full_pipeline_vader.py
```

**Step-by-step manual execution:**
```bash
# Step 1: Data Collection (Reddit-only, multi-query strategies)
python3 simple_collector.py

# Step 2: Data Preprocessing
python3 enhanced_preprocessing.py

# Step 3: Sentiment Classification (RoBERTa primary model)
python3 sentiment_classifier_roberta.py

# Step 3 alternatives:
# - Logistic Regression: python3 sentiment_classifier.py
# - VADER: python3 sentiment_classifier_vader.py

# Step 4: Model Comparison & Selection
python3 compare_sentiment_models.py

# Step 5: Quality Validation
python3 phase1_quality_check.py

# Step 6: Network Analysis (Parallel branch 1)
python3 network_analysis.py

# Step 7: Temporal Analysis (Parallel branch 2)
python3 temporal_analysis.py

# Step 8: LDA Topic Modeling (Parallel branch 3)
python3 lda_topic_modeling.py

# Step 9: Topic Coherence Analysis
python3 lda_coherence_simple.py

# Step 10: Topic Stability Validation
python3 topic_stability_analysis_final.py

# Step 11: Network Visualization
python3 network_visualizer.py

# Step 12: Topic Modeling Visualization
python3 topic_modeling_visualizer.py

# Step 13: Semantic Drivers Analysis
python3 semantic_drivers_visualizer.py

# Step 14: Policy Milestone Hypothesis Test
python3 policy_milestone_hypothesis.py

# Step 15: Smart Farming Hypothesis Test
python3 smart_farming_hypothesis.py

# Step 16: Regional Pattern Hypothesis Test
python3 regional_hypothesis_proxy.py

# Step 17: Robustness & Sensitivity Analysis
python3 robustness_sensitivity_analysis.py

# Final: Generate Comprehensive Statistics
python3 generate_statistics.py
```

## Project Structure

```
# Orchestrator Scripts
├── run_full_pipeline.py              # End-to-end logistic pipeline
├── run_full_pipeline_roberta.py      # End-to-end RoBERTa pipeline (primary)
├── run_full_pipeline_vader.py        # End-to-end VADER pipeline

# Data Collection & Preprocessing
├── simple_collector.py               # Reddit API collection (2018+ data)
├── enhanced_preprocessing.py         # Phase 2 cleaning + lemmatization

# Sentiment Classification (4 models)
├── sentiment_classifier.py           # Logistic Regression + TF-IDF
├── sentiment_classifier_roberta.py   # RoBERTa transformer (primary)
├── sentiment_classifier_vader.py     # VADER lexicon-based
├── compare_sentiment_models.py       # Model comparison + selection

# Quality & Validation
├── phase1_quality_check.py           # Label distribution validation

# Structural Analysis
├── network_analysis.py               # Similarity graph + community detection
├── temporal_analysis.py              # Time-series sentiment + trend detection

# Topic Modeling & Validation
├── lda_topic_modeling.py             # LDA fitting (10 topics)
├── lda_coherence_simple.py           # Coherence metrics (Cv)
├── topic_stability_analysis_final.py # Cross-validation robustness

# Visualization & Reporting
├── network_visualizer.py             # Network graph + community visualization
├── topic_modeling_visualizer.py      # Interactive pyLDAvis output
├── semantic_drivers_visualizer.py    # Word-sentiment correlation analysis
├── generate_statistics.py            # Comprehensive statistics report

# Hypothesis Testing & Validation
├── policy_milestone_hypothesis.py    # Policy impact validation
├── smart_farming_hypothesis.py       # AI/automation driver analysis
├── regional_hypothesis_proxy.py      # Network pattern validation
├── robustness_sensitivity_analysis.py # Resampling & sensitivity checks

# Documentation & Config
├── requirements.txt
├── README.md
├── PHASE1_DATA_EXTRACTION_DOCUMENTATION.md
├── PHASE2_PREPROCESSING_DOCUMENTATION.md
├── PRIVACY_COMPLIANCE_VERIFICATION.md
├── REQUIREMENT_3_ANALYSIS_SUMMARY.md
├── PROJECT_SUMMARY.md
└── visualizations/                  # Output directory for HTML/PNG charts
```

## Output Files

**Data (JSON):**
- `enhanced_scraped_data.json` - Collected posts (2018+)
- `preprocessed_data.json` - Cleaned text with tokens & lemmas
- `classified_sentiment_data.json` - Active sentiment classifications (default: RoBERTa)
- `classified_sentiment_data_roberta.json` - RoBERTa model output
- `classified_sentiment_data_logistic_backup.json` - Logistic baseline
- `classified_sentiment_data_vader.json` - VADER alternative
- `temporal_analysis_results.json` - Monthly/quarterly trends
- `temporal_events.json` - Time-series sentiment aggregation
- `network_analysis_results.json` - Network statistics & communities
- `network_graph.json` - Graph structure with centrality scores
- `topic_distributions.json` - LDA doc-topic matrix
- `coherence_metrics.json` - Topic coherence scores
- `stability_scores.json` - Topic stability validation results
- `hypothesis_results.json` - Policy/semantic/regional test outcomes
- `robustness_report.json` - Sensitivity analysis results
- `final_statistics.json` - Comprehensive summary statistics

**Models (Serialized):**
- `sentiment_model.pkl` - Trained Logistic Regression classifier
- `vectorizer.pkl` - TF-IDF vectorizer
- `lda_model_output.pkl` - Fitted LDA model (10 topics)
- `keyword_network.pkl` - NetworkX graph object

**Model Comparison:**
- `sentiment_model_comparison.json` - Structured metrics across all models
- `sentiment_model_comparison.txt` - Human-readable comparison report

**Visualizations (HTML & PNG):**
- `visualizations/network_full.png` - Complete document similarity network
- `visualizations/network_communities.png` - Community structure
- `visualizations/sentiment_overview.png` - 4-panel sentiment summary
- `visualizations/lda_visualization.html` - Interactive pyLDAvis chart
- `visualizations/semantic_drivers_chart.html` - Word-sentiment correlation
- `visualizations/logistic/` - Model-specific visualizations (logistic)
- `visualizations/roberta/` - Model-specific visualizations (RoBERTa)
- `visualizations/vader/` - Model-specific visualizations (VADER)

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
