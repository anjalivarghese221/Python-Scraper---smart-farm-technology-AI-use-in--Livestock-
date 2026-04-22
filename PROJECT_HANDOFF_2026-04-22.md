# Project Handoff Document
## Smart Farm Technology & AI in Livestock (Reddit, 2018–2026)

**Handoff date:** 2026-04-22  
**Repository root:** `/Users/anjalivarghese/Python-Scraper---smart-farm-technology-AI-use-in--Livestock-`  
**Primary language:** Python 3.9+  
**Data source:** Reddit JSON API only

---

## 1) Executive Summary

This project is an end-to-end NLP and quantitative analysis pipeline for Reddit discussions about AI/smart farming in livestock systems.

In plain language: this project listens to thousands of public online discussions, cleans the text so it can be analyzed reliably, classifies each post as positive/negative/neutral, and then turns that into charts and evidence that can support research reporting and decision-making.

The work is not just “a script that makes a graph.” It is a complete research workflow with:
- traceable data collection,
- quality checks,
- multiple sentiment model options,
- repeatable analyses,
- and stakeholder-ready visual outputs.

It includes:
- Multi-strategy Reddit data collection
- Preprocessing with audit/attrition reporting
- Sentiment analysis with **3 model options**:
  - Logistic regression (`sentiment_classifier.py`)
  - RoBERTa (`sentiment_classifier_roberta.py`)
  - VADER (`sentiment_classifier_vader.py`)
- Network analysis and visualization
- Temporal analysis and trend/spike style charts
- Topic modeling coherence/stability reports
- Hypothesis testing + robustness checks
- Model comparison outputs across Logistic/RoBERTa/VADER

Recent work completed:
- Added VADER sentiment pipeline (`run_full_pipeline_vader.py`)
- Added model-specific temporal chart generation (`generate_model_temporal_visualizations.py`)
- Added per-model temporal visualization outputs in:
  - `visualizations/logistic/temporal_sentiment_trend_analysis.png`
  - `visualizations/roberta/temporal_sentiment_trend_analysis.png`
  - `visualizations/vader/temporal_sentiment_trend_analysis.png`

### 1.1 Why this project matters (non-technical)

This project answers a practical question: **how are people reacting over time to AI and smart-farming technologies in livestock contexts?**

It helps with:
- understanding whether sentiment is becoming more positive or negative,
- identifying which themes (cost, precision, risk, monitoring, etc.) drive reactions,
- comparing how conclusions change when different sentiment models are used,
- and creating a repeatable evidence trail for reports, presentations, or policy conversations.

### 1.2 How to read this handoff

- If you are a **non-coding stakeholder**, focus on Sections 1, 2, 3, and 12 first.
- If you are a **technical owner**, Sections 4 through 11 are your implementation map.
- If you are inheriting maintenance, Section 10 (caveats) and Section 11 (stabilization tasks) are the priority.

---

## 1A) Plain-English Project Walkthrough

Think of the workflow as a production line:

1. **Collect**: pull relevant Reddit posts from 2018 onward.
2. **Clean**: remove noise and duplicates while preserving meaning.
3. **Label sentiment**: tag each post as positive/negative/neutral.
4. **Analyze patterns**: find topic clusters, trends over time, and statistical differences.
5. **Visualize**: generate figures for reports and stakeholder communication.

At the end, you get both:
- machine-readable outputs (JSON) for reproducibility, and
- visual/report outputs (PNG/TXT/MD) for human interpretation.

---

## 1B) What “good output” looks like

A healthy run should produce:
- a classified dataset file for the selected model,
- updated comparison outputs (`sentiment_model_comparison.json/.txt`),
- refreshed analysis results (`network_analysis_results.json`, `temporal_analysis_results.json`, etc.),
- and visual folders populated for the model mode (logistic/roberta/vader).

If those outputs are present and counts are consistent, the pipeline is operational.

---

## 2) Current State Snapshot (Important)

### 2.1 Key current outputs
- `final_statistics.json` currently points to dataset: `classified_sentiment_data_clean_high_coverage.json`
- Current final dataset size in stats: **2640 posts**
- Date range in stats: **2018-01-03 to 2026-03-16**
- Current sentiment counts in `final_statistics.json` (high-coverage dataset):
  - positive: 1500
  - negative: 785
  - neutral: 355

### 2.2 Current model comparison snapshot
From `sentiment_model_comparison.json`:
- Logistic: 2684 rows
- RoBERTa: 2684 rows
- VADER: 2684 rows
- Agreement:
  - Logistic vs RoBERTa: 26.34%
  - Logistic vs VADER: 43.11%
  - RoBERTa vs VADER: 46.24%

### 2.3 Current network snapshot
From `network_analysis_results.json`:
- Nodes: 1080
- Edges: 5508
- Meaningful communities retained: 4

### 2.4 Interpretation of the current snapshot (non-technical)

- The project currently has a **large enough corpus** to support trend and topic analysis.
- Sentiment is currently **majority positive** in the high-coverage dataset, with substantial negative discourse still present.
- Model disagreement is non-trivial (especially Logistic vs RoBERTa), which is expected in real-world language data and reinforces why multi-model comparison is valuable.
- The network graph size (1080 keywords, 5508 connections) indicates rich thematic structure, not a sparse or underpowered dataset.

### 2.5 What this means for decision-makers

For non-technical consumers of the output:
- Use **directional conclusions** (trend up/down, key concerns, key opportunities) with confidence.
- Treat exact percentages as **model-sensitive estimates** rather than single absolute truth.
- For external reporting, include model-comparison context to show methodological rigor.

### 2.6 Key analysis findings (data-focused, non-technical)

Below is a concise interpretation of the most important evidence currently in the repository.

1. **Post-2024 sentiment is statistically more positive than pre-2024**  
  In the policy milestone test, average sentiment after 2024 is higher than before 2024, and the test result is statistically significant (`p < 0.001`).
  - Practical takeaway: discourse appears to have shifted in a more positive direction after the policy-era boundary used in the analysis.
  - Caveat: effect size is small (not a dramatic shift), so this should be communicated as “meaningful but modest.”

2. **The surge window (from 2025-07 onward) is more positive than the pre-surge period**  
  The smart-farming hypothesis test shows a significant difference between pre-surge and surge-window sentiment (`p < 0.001`).
  - Practical takeaway: growth in conversation volume was accompanied by slightly stronger positive sentiment.
  - Caveat: as above, this is a moderate/small practical effect, not a complete reversal of sentiment.

3. **Livestock-focused vs non-livestock discourse does not show strong separation**  
  The livestock-vs-non-livestock comparison did not reach significance at `alpha = 0.05`.
  - Practical takeaway: in this current labeling strategy, the sentiment difference by that specific topic split is limited.
  - Caveat: this can be sensitive to how “livestock-focused” is defined and filtered.

4. **Region-proxy result is inconclusive**  
  Region analysis (using subreddit-name proxy only) did not find a significant difference in the tested top-two groups.
  - Practical takeaway: no strong geographic-proxy claim should be made from this run.
  - Caveat: region here is inferred indirectly and sample sizes are small.

5. **Robustness checks generally support the surge finding**  
  Time-window shifts (`±7 days`) still show significance, and subsampling remains significant in most runs.
  - Practical takeaway: the main temporal signal is not likely a one-day artifact.
  - Important caveat: removing top 5% highest-volume days weakens the effect to non-significant, indicating part of the signal is concentrated in high-activity events.

6. **Semantic driver patterns are interpretable and policy-relevant**  
  Negative drivers include cost/inflation/funding terms; positive drivers include precision/accuracy/wireless/optimization terms.
  - Practical takeaway: concerns cluster around economics and risk; optimism clusters around technical capability and operational benefit.

### 2.7 Business/policy interpretation guidance

For non-technical reporting, the safest claims are:
- “Overall sentiment is more positive than negative in the current high-coverage corpus.”
- “There is statistically significant evidence of more positive sentiment in the post-2024 / surge period analyses.”
- “Economic and cost-related language is a recurring source of negative tone.”
- “Model choice matters, so conclusions should be triangulated rather than tied to one classifier.”

Avoid overclaiming:
- Do **not** frame findings as causal proof that policy caused sentiment change.
- Do **not** make strong geographic claims from the current region-proxy analysis.
- Do **not** present one model’s percentages as the single definitive truth without comparison context.

### 2.8 Data quality and representativeness notes

- Data is Reddit-only: high discourse richness, but not a full representation of all farmers or stakeholders.
- Community mix is broad, including many non-core subreddits, which increases coverage but can dilute domain purity.
- The project includes filtering and quality checks, but this remains observational social data with platform bias.
- Temporal analysis spans many months and is strong for trend detection, but event-driven spikes can influence aggregate outcomes.

---

## 3) High-Level Architecture

### 3.1 Data flow (conceptual)
1. Collect Reddit posts/comments → `enhanced_scraped_data.json`  
2. Preprocess and normalize → `preprocessed_data.json`  
3. Sentiment classification (model-dependent) → `classified_sentiment_data*.json`  
4. Analysis modules consume classified data:
   - quality checks
   - network analysis
   - temporal analysis
   - topic/coherence/stability
   - hypotheses + robustness
5. Visualizers generate PNG outputs under `visualizations/`

### 3.2 Pipeline orchestrators
- Baseline logistic pipeline: `run_full_pipeline.py`
- RoBERTa pipeline: `run_full_pipeline_roberta.py`
- VADER pipeline: `run_full_pipeline_vader.py`

### 3.3 Architecture summary in plain language

There are two layers:

- **Core layer**: collection → cleaning → classification → analysis.
- **Presentation layer**: charts and report files generated from the analysis outputs.

The reason this separation matters is maintainability: you can improve one stage (for example, sentiment labeling) without rewriting every chart.

---

## 4) Core Files and Responsibilities

## 4.1 Collection + preprocessing
- `simple_collector.py`
  - Query plan generation via lexical expansion/phrase-targeted strategies
  - Reddit post scraping + optional comment enrichment
  - Domain relevance scoring and filtering
  - Writes:
    - `enhanced_scraped_data.json`
    - `query_log.json`

- `enhanced_preprocessing.py`
  - Language detection heuristic
  - Exact deduplication via MD5 hash
  - URL/mention cleanup, hashtag handling, whitespace normalization
  - Keeps multiple text representations (`raw_text`, `clean_text`, `tokens`)
  - Writes `preprocessed_data.json` with attrition metadata/report

## 4.2 Sentiment models
- `sentiment_classifier.py`
  - Uses pre-trained `sentiment_model.pkl` + `vectorizer.pkl`
  - Outputs:
    - `classified_sentiment_data.json`
    - `sentiment_report.txt`

- `sentiment_classifier_roberta.py`
  - Uses Hugging Face model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Outputs:
    - `classified_sentiment_data_roberta.json`
    - `sentiment_report_roberta.txt`

- `sentiment_classifier_vader.py`
  - Lexicon-based sentiment (`vaderSentiment`)
  - Thresholds:
    - positive: `compound >= 0.05`
    - negative: `compound <= -0.05`
    - neutral: otherwise
  - Outputs:
    - `classified_sentiment_data_vader.json`
    - `sentiment_report_vader.txt`

- `compare_sentiment_models.py`
  - Compares Logistic vs RoBERTa vs VADER (if files exist)
  - Outputs:
    - `sentiment_model_comparison.json`
    - `sentiment_model_comparison.txt`

## 4.3 Main analysis
- `phase1_quality_check.py`
  - Keyword extraction by sentiment class
  - Creates manual review sample (n=20)
  - Outputs:
    - `phase1_keyword_check_by_sentiment.json`
    - `phase1_manual_review_sample_20.json`

- `network_analysis.py`
  - Keyword co-occurrence graph construction
  - Community detection and community sentiment summary
  - Outputs:
    - `keyword_network.pkl`
    - `network_analysis_results.json`
    - `network_analysis_report.txt`

- `temporal_analysis.py`
  - Monthly/quarterly trend aggregation
  - Trend slope inference and report generation
  - Outputs:
    - `temporal_analysis_results.json`
    - `visualizations/temporal_trends_monthly.png`
    - `visualizations/temporal_trends_quarterly.png`

- `topic_stability_analysis_final.py`
  - Topic coherence/stability and semantic driver extraction
  - Outputs:
    - `topic_stability_coherence_report.json`
    - `classified_sentiment_data_domain_smart_farming_livestock.json`

- `lda_topic_modeling.py` + `lda_coherence_simple.py`
  - LDA, coherence sensitivity, model selection reporting
  - Key output: `lda_coherence_results.json`

- `generate_statistics.py`
  - Consolidated summary stats for current selected dataset candidate
  - Output: `final_statistics.json`

## 4.4 Visualization generators
- `network_visualizer.py`
  - Produces top-level network + community + sentiment overview PNGs

- `topic_modeling_visualizer.py`
  - Produces topic modeling visuals under `visualizations/topic_modeling/`

- `semantic_drivers_visualizer.py`
  - Produces `visualizations/semantic_drivers_top10.png`

- `generate_model_temporal_visualizations.py`
  - Produces model-specific temporal figure per model folder:
    - `visualizations/logistic/temporal_sentiment_trend_analysis.png`
    - `visualizations/roberta/temporal_sentiment_trend_analysis.png`
    - `visualizations/vader/temporal_sentiment_trend_analysis.png`

## 4.5 Hypothesis & robustness modules
- `policy_milestone_hypothesis.py`
- `smart_farming_hypothesis.py`
- `regional_hypothesis_proxy.py`
- `robustness_sensitivity_analysis.py`

Each writes JSON result files and associated visuals.

---

## 5) Dataset and Artifact Conventions

### 5.1 Primary dataset files
- Raw: `enhanced_scraped_data.json`
- Preprocessed: `preprocessed_data.json`
- Model outputs:
  - Logistic: `classified_sentiment_data.json`
  - RoBERTa: `classified_sentiment_data_roberta.json`
  - VADER: `classified_sentiment_data_vader.json`
- Curated/high-coverage corpus: `classified_sentiment_data_clean_high_coverage.json`

### 5.2 Analysis outputs
- `network_analysis_results.json`
- `temporal_analysis_results.json`
- `topic_stability_coherence_report.json`
- `lda_coherence_results.json`
- `final_statistics.json`
- `sentiment_model_comparison.json`

### 5.3 Visualization layout
- General/current run: `visualizations/*.png`
- Topic modeling: `visualizations/topic_modeling/*.png`
- Model snapshots:
  - `visualizations/logistic/`
  - `visualizations/roberta/`
  - `visualizations/vader/`

---

## 6) How the 3 Model Pipelines Work

### 6.1 Logistic (baseline)
`run_full_pipeline.py`
- Uses `sentiment_classifier.py`
- Continues downstream on default files

### 6.2 RoBERTa variant
`run_full_pipeline_roberta.py`
- Runs `sentiment_classifier_roberta.py`
- Runs model comparison
- Temporarily maps `classified_sentiment_data.json` to RoBERTa output for downstream scripts
- Restores original logistic file backup at end

### 6.3 VADER variant
`run_full_pipeline_vader.py`
- Runs `sentiment_classifier_vader.py`
- Runs model comparison
- Temporarily maps `classified_sentiment_data.json` to VADER output for downstream scripts
- Copies visual outputs into `visualizations/vader/`
- Restores original logistic file backup at end

---

## 7) Reproducible Runbook

This section is the “how to rerun everything” checklist. A non-technical operator can treat each command as a step in order. A technical user can automate it via tasks.

### 7.1 Environment setup
1. Python 3.9+
2. Install dependencies:
   - `pip3 install -r requirements.txt`

### 7.2 Run baseline pipeline
- `python3 run_full_pipeline.py`

### 7.3 Run RoBERTa pipeline
- `python3 run_full_pipeline_roberta.py`

### 7.4 Run VADER pipeline
- `python3 run_full_pipeline_vader.py`

### 7.5 Generate per-model temporal visualizations
- `python3 generate_model_temporal_visualizations.py`

### 7.6 Practical run strategy (recommended)

For handoff continuity, use this operating pattern:
1. Run one full model pipeline (e.g., VADER or RoBERTa).
2. Verify comparison file updated.
3. Verify model-specific visualization folder updated.
4. Archive/share only validated outputs.

This avoids mixing partial outputs from multiple runs.

---

## 8) VS Code Tasks Available

The workspace includes task definitions in `.vscode/tasks.json`, including:
- `run-full-pipeline`
- `run-full-pipeline-vader`
- high-coverage rebuild/verify tasks
- utility tasks moved to `legacy/tmp` paths

---

## 9) Testing Coverage

Current automated tests are limited to the logistic classifier module:
- `tests/test_sentiment_classifier.py`
- `tests/test_sentiment_classifier_additional.py`

No dedicated tests currently exist for:
- `sentiment_classifier_roberta.py`
- `sentiment_classifier_vader.py`
- `run_full_pipeline_roberta.py`
- `run_full_pipeline_vader.py`
- topic/temporal/hypothesis modules

---

## 10) Known Caveats / Technical Debt (Read Carefully)

This section is critical for both technical and non-technical stakeholders. It explains why two runs can produce slightly different numbers even when no “bug” is present.

1. **Model pipeline consistency caveat**  
   Several downstream scripts prefer curated files (e.g., `classified_sentiment_data_clean_high_coverage.json`) over `classified_sentiment_data.json`.  
   This means a RoBERTa/VADER run may still produce downstream analyses from high-coverage logistic-curated data in some modules.

2. **Dataset candidate ordering differs by script**  
   Each module has its own `input_candidates` order. Results can differ depending on what files exist.

3. **Requirements file incompleteness risk**  
   Some scripts rely on packages not clearly pinned for every path (e.g., `scipy`, `gensim` install fallback in-script).  
   `lda_topic_modeling.py` may self-install `gensim` at runtime if missing.

4. **Mixed historical stats in docs**  
   Markdown summaries contain older values (e.g., 2811 rows) while current outputs may reflect 2640/2684 depending on file used.

5. **High number of non-domain subreddits remains**  
   Temporal results show broad subreddit spread; additional domain filtering may be desired for stricter inference.

6. **Legacy archive contains exploratory scripts/artifacts**  
   Old temporary/ad-hoc scripts are in `legacy/` and are not part of active pipeline.

### 10.1 Risk framing (non-technical)

- **Low risk**: visual generation failures (easy to detect and rerun).
- **Medium risk**: dependency drift (package versions causing environment differences).
- **High risk**: inconsistent input file selection across scripts (can silently change conclusions).

The top stabilization priority is therefore input standardization and run manifests.

### 10.2 Analysis interpretation caveats (important for non-coders)

1. **Statistical significance is not the same as practical impact**  
  Some tests are highly significant due to sample size, while effect sizes remain small.

2. **Classifier disagreement is expected**  
  Different sentiment models use different logic; disagreement should be treated as uncertainty information, not necessarily model failure.

3. **Multiple valid datasets exist in-repo**  
  Some reports use clean-expanded files, others use clean-high-coverage files; results can change because the analyzed population changes.

4. **Proxy variables have limits**  
  Region is inferred from subreddit naming conventions, which is useful for exploration but weak for hard geographic inference.

5. **Volume spikes can dominate trend signals**  
  Robustness checks show that high-volume days can carry a disproportionate share of temporal effect.

---

## 11) Recommended Immediate Stabilization Tasks

1. **Standardize one canonical input file per run mode**
   - Add a shared config file (e.g., `analysis_config.json`) with a single `classified_input_path`
   - Make all analysis scripts honor it

2. **Pin all required dependencies explicitly**
   - Include `scipy`, `gensim`, and any missing packages in `requirements.txt`
   - Avoid runtime install side effects

3. **Add model-specific integration tests**
   - Smoke tests for RoBERTa and VADER classifiers
   - Pipeline-level test that verifies outputs exist and file counts are consistent

4. **Add run manifest output**
   - At pipeline start/end, write a `run_manifest.json` summarizing:
     - exact input files used per step
     - git commit hash
     - timestamp
     - model mode

5. **Separate exploratory and production docs**
   - Keep this handoff as production source of truth
   - Flag legacy reports as historical

---

## 12) Key Output Files to Hand to Stakeholders

When sharing externally, pair each data-heavy file with a simple explanation slide or memo. Non-technical audiences should not be expected to interpret raw JSON directly.

For a clean external handoff package, prioritize:
- `PROJECT_HANDOFF_2026-04-22.md` (this file)
- `README.md`
- `final_statistics.json`
- `sentiment_model_comparison.json`
- `network_analysis_results.json`
- `temporal_analysis_results.json`
- `topic_stability_coherence_report.json`
- `visualizations/logistic/`
- `visualizations/roberta/`
- `visualizations/vader/`

### 12.1 Suggested stakeholder packet (non-technical friendly)

For executive/policy audiences, provide these in one folder:

1. This handoff document (context + caveats)
2. One-page summary slide with:
  - corpus size,
  - sentiment split,
  - post-2024 change statement,
  - model comparison headline
3. Three visuals:
  - model-specific temporal trend figure,
  - community sentiment figure,
  - semantic drivers figure
4. One methods note page:
  - data source,
  - key assumptions,
  - what conclusions are supported vs not supported

This package format minimizes misinterpretation while staying understandable to non-technical readers.

---

## 13) Ownership Notes for Next Engineer

If you are taking over this project:
- Start by deciding what the **canonical dataset file** should be for each pipeline mode.
- Run one full mode end-to-end and confirm every module used the intended file.
- Lock dependencies and add integration tests before changing methodology.
- Preserve `legacy/` as historical provenance but keep active workflow rooted in top-level scripts.

### 13.1 Ownership notes for non-engineering leads

If you are managing this project without coding daily:
- ask for a run manifest with every major refresh,
- require a “data file used per step” table in status updates,
- and request side-by-side model comparison whenever sentiment claims are presented.

This keeps the project transparent and audit-friendly.

---

## 14) Provenance

This handoff was compiled by reading active pipeline scripts, docs, and generated outputs in the workspace on 2026-04-22, including:
- orchestrators (`run_full_pipeline*.py`)
- sentiment/model comparison modules
- network/temporal/topic/hypothesis modules
- current result artifacts (`final_statistics.json`, `sentiment_model_comparison.json`, etc.)

---

## 15) Plain-English Glossary (for non-coding readers)

- **Sentiment model:** A method that labels text as positive, negative, or neutral.
- **Classifier agreement:** How often two models give the same label on the same post.
- **Statistically significant:** The observed difference is unlikely to be random under the test assumptions.
- **Effect size:** How large the difference is in practical terms (small/medium/large).
- **Robustness check:** Repeating analysis with altered assumptions to see if the conclusion holds.
- **Coherence (topic modeling):** How semantically consistent the discovered topics are.
- **Proxy variable:** An indirect substitute for a missing direct measure (e.g., subreddit name as region hint).
- **Attrition:** The number of records removed during cleaning/filtering stages.

---

**End of handoff.**
