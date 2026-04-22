#!/usr/bin/env python3
"""
Run the full analysis pipeline with VADER sentiment classification.
Mirrors run_full_pipeline.py but swaps the sentiment step to VADER.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

BASE = '/Users/anjalivarghese/Python-Scraper---smart-farm-technology-AI-use-in--Livestock-'

STEPS = [
    'enhanced_preprocessing.py',
    'sentiment_classifier_vader.py',
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


def snapshot_vader_visualizations():
    src = Path('visualizations')
    dst = src / 'vader'
    dst_topic = dst / 'topic_modeling'

    dst.mkdir(parents=True, exist_ok=True)
    dst_topic.mkdir(parents=True, exist_ok=True)

    copied = 0

    for p in src.glob('*.png'):
        if p.parent == dst:
            continue
        shutil.copy2(p, dst / p.name)
        copied += 1

    src_topic = src / 'topic_modeling'
    if src_topic.exists():
        for p in src_topic.glob('*.png'):
            shutil.copy2(p, dst_topic / p.name)
            copied += 1

    print(f'Copied {copied} visualization files to {dst}')


def main():
    os.chdir(BASE)
    results = []

    # Preserve existing logistic output if present, then map downstream steps to VADER.
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

        # After model comparison, route downstream steps to VADER labels.
        if step == 'compare_sentiment_models.py' and proc.returncode == 0:
            if os.path.exists('classified_sentiment_data_vader.json'):
                subprocess.run(['cp', 'classified_sentiment_data_vader.json', 'classified_sentiment_data.json'], check=False)
                print('Mapped classified_sentiment_data.json -> VADER output for downstream steps')

        print(f"EXIT {proc.returncode} in {duration:.1f}s")
        results.append((step, proc.returncode, duration))

    # Save a VADER visualization snapshot.
    try:
        snapshot_vader_visualizations()
    except Exception as e:
        print(f'WARNING: could not snapshot VADER visualizations: {e}')

    # Restore original logistic output if there was one.
    if logistic_backup and os.path.exists(logistic_backup):
        subprocess.run(['cp', logistic_backup, 'classified_sentiment_data.json'], check=False)
        print('\nRestored original classified_sentiment_data.json from logistic backup')

    failed = [r for r in results if r[1] != 0]
    print('\n=== VADER PIPELINE SUMMARY ===')
    for step, rc, duration in results:
        print(f"{step}: rc={rc}, {duration:.1f}s")
    print(f"FAILED_COUNT {len(failed)}")
    if failed:
        print('FAILED_STEPS', [x[0] for x in failed])


if __name__ == '__main__':
    main()
