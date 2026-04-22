#!/usr/bin/env python3
"""
Run the full analysis pipeline with RoBERTa sentiment classification.
This mirrors run_full_pipeline.py but replaces logistic sentiment step.
"""

import os
import subprocess
import time

BASE = '/Users/anjalivarghese/Python-Scraper---smart-farm-technology-AI-use-in--Livestock-'

STEPS = [
    'enhanced_preprocessing.py',
    'sentiment_classifier_roberta.py',
    'compare_sentiment_models.py',
    'phase1_quality_check.py',
    'network_analysis.py',
    'temporal_analysis.py',
    'network_visualizer.py',
    'generate_statistics.py',
    'lda_topic_modeling.py',
    'lda_coherence_simple.py',
    'topic_stability_analysis_final.py',
    'topic_modeling_visualizer.py',
    'semantic_drivers_visualizer.py',
    'policy_milestone_hypothesis.py',
    'smart_farming_hypothesis.py',
    'regional_hypothesis_proxy.py',
    'robustness_sensitivity_analysis.py',
]


def run(cmd):
    start = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    duration = time.time() - start
    return proc, duration


def main():
    os.chdir(BASE)
    results = []

    # Keep existing logistic output if present, then switch default classified file to RoBERTa output.
    logistic_backup = None
    if os.path.exists('classified_sentiment_data.json'):
        logistic_backup = 'classified_sentiment_data_logistic_backup.json'
        subprocess.run(['cp', 'classified_sentiment_data.json', logistic_backup], check=False)

    for i, step in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] RUN {step}")
        proc, duration = run(['python3', step])

        if proc.stdout:
            print(proc.stdout[-3000:])
        if proc.returncode != 0 and proc.stderr:
            print('--- STDERR ---')
            print(proc.stderr[-3000:])

        # After model comparison, point downstream scripts at RoBERTa file by copying it to default path.
        if step == 'compare_sentiment_models.py' and proc.returncode == 0:
            if os.path.exists('classified_sentiment_data_roberta.json'):
                subprocess.run(['cp', 'classified_sentiment_data_roberta.json', 'classified_sentiment_data.json'], check=False)
                print('Mapped classified_sentiment_data.json -> RoBERTa output for downstream steps')

        print(f"EXIT {proc.returncode} in {duration:.1f}s")
        results.append((step, proc.returncode, duration))

    # Restore original logistic output if there was one.
    if logistic_backup and os.path.exists(logistic_backup):
        subprocess.run(['cp', logistic_backup, 'classified_sentiment_data.json'], check=False)
        print('\nRestored original classified_sentiment_data.json from logistic backup')

    failed = [r for r in results if r[1] != 0]
    print('\n=== ROBERTA PIPELINE SUMMARY ===')
    for step, rc, duration in results:
        print(f"{step}: rc={rc}, {duration:.1f}s")
    print(f"FAILED_COUNT {len(failed)}")
    if failed:
        print('FAILED_STEPS', [x[0] for x in failed])


if __name__ == '__main__':
    main()
