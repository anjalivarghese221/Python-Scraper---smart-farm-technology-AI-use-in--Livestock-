#!/usr/bin/env python3
import os
import subprocess
import time

BASE = '/Users/anjalivarghese/Python-Scraper---smart-farm-technology-AI-use-in--Livestock-'

STEPS = [
    'enhanced_preprocessing.py',
    'sentiment_classifier.py',
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


def main():
    os.chdir(BASE)
    results = []

    for i, step in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] RUN {step}")
        start = time.time()
        proc = subprocess.run(['python3', step], text=True, capture_output=True)
        duration = time.time() - start

        if proc.stdout:
            print(proc.stdout[-3000:])
        if proc.returncode != 0 and proc.stderr:
            print('--- STDERR ---')
            print(proc.stderr[-3000:])

        print(f"EXIT {proc.returncode} in {duration:.1f}s")
        results.append((step, proc.returncode, duration))

    failed = [r for r in results if r[1] != 0]
    print('\n=== PIPELINE SUMMARY ===')
    for step, rc, duration in results:
        print(f"{step}: rc={rc}, {duration:.1f}s")
    print(f"FAILED_COUNT {len(failed)}")
    if failed:
        print('FAILED_STEPS', [x[0] for x in failed])


if __name__ == '__main__':
    main()
