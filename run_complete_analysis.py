#!/usr/bin/env python3
"""
Complete Scientific Workflow Executor
Runs all phases in sequence with progress tracking

Usage:
    python3 run_complete_analysis.py [--target-posts 1500] [--skip-scraping]
"""

import sys
import os
import json
from datetime import datetime


def print_header(text):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def check_file_exists(filename):
    """Check if file exists"""
    return os.path.exists(filename)


def run_phase(phase_num, phase_name, script_name, required_input=None, output_file=None):
    """Execute a workflow phase"""
    print_header(f"PHASE {phase_num}: {phase_name}")
    
    # Check prerequisites
    if required_input and not check_file_exists(required_input):
        print(f"⚠️  ERROR: Required input file not found: {required_input}")
        print(f"   Please run previous phases first.")
        return False
    
    # Check if output already exists
    if output_file and check_file_exists(output_file):
        response = input(f"\n⚠️  {output_file} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print(f"   Skipping Phase {phase_num}")
            return True
    
    # Execute script
    print(f"🔄 Executing: python3 {script_name}")
    print("-" * 80)
    
    exit_code = os.system(f"python3 {script_name}")
    
    if exit_code != 0:
        print(f"\n❌ Phase {phase_num} failed with exit code {exit_code}")
        return False
    
    # Verify output
    if output_file and not check_file_exists(output_file):
        print(f"\n❌ Phase {phase_num} did not generate expected output: {output_file}")
        return False
    
    print(f"\n✅ Phase {phase_num} complete")
    return True


def print_summary(results):
    """Print workflow completion summary"""
    print_header("WORKFLOW COMPLETE")
    
    print("Phase Completion Status:")
    phases = [
        ("Phase 1", "Data Collection", results.get('phase1', False)),
        ("Phase 2", "Preprocessing", results.get('phase2', False)),
        ("Phase 3", "Sentiment Classification", results.get('phase3', False)),
        ("Phase 4", "Temporal Analysis", results.get('phase4', False))
    ]
    
    for phase, name, status in phases:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {phase}: {name}")
    
    print("\nGenerated Files:")
    files = [
        ('enhanced_scraped_data.json', 'Raw dataset with temporal metadata'),
        ('query_log.json', 'Query documentation for reproducibility'),
        ('preprocessed_data.json', 'Cleaned dataset with attrition report'),
        ('sentiment_model.pkl', 'Trained sentiment classifier'),
        ('classified_sentiment_data.json', 'Posts with sentiment labels'),
        ('temporal_analysis_results.json', 'Time-series trend data'),
        ('visualizations/temporal_trends_monthly.png', 'Monthly sentiment chart'),
        ('visualizations/temporal_trends_quarterly.png', 'Quarterly sentiment chart')
    ]
    
    for filename, description in files:
        if check_file_exists(filename):
            print(f"  ✅ {filename} - {description}")
        else:
            print(f"  ⏭️  {filename} - {description}")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Review temporal_analysis_results.json for trends")
    print("  2. Check visualizations/ for charts")
    print("  3. Read SCIENTIFIC_WORKFLOW.md for interpretation")
    print("  4. Use attrition table for methods section")
    print("=" * 80 + "\n")


def main():
    """Execute complete scientific workflow"""
    print_header("SCIENTIFIC SOCIAL MEDIA ANALYSIS WORKFLOW")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: Peer-reviewed publication standards")
    
    # Parse arguments
    skip_scraping = '--skip-scraping' in sys.argv
    target_posts = 1500
    if '--target-posts' in sys.argv:
        idx = sys.argv.index('--target-posts')
        if idx + 1 < len(sys.argv):
            target_posts = int(sys.argv[idx + 1])
    
    results = {}
    
    # Phase 1: Data Collection (can be skipped if data exists)
    if skip_scraping and check_file_exists('enhanced_scraped_data.json'):
        print_header("PHASE 1: DATA COLLECTION (SKIPPED)")
        print("Using existing enhanced_scraped_data.json")
        results['phase1'] = True
    else:
        print(f"\n📊 Target dataset size: {target_posts} posts")
        print("⏱️  Estimated time: 20-40 minutes (depends on rate limiting)")
        response = input("\n▶️  Proceed with data collection? (y/n): ")
        
        if response.lower() != 'y':
            print("❌ Data collection cancelled")
            return
        
        results['phase1'] = run_phase(
            1, 
            "DATA COLLECTION",
            "enhanced_scraper.py",
            output_file="enhanced_scraped_data.json"
        )
        
        if not results['phase1']:
            print("\n❌ Workflow aborted: Phase 1 failed")
            return
    
    # Phase 2: Preprocessing
    results['phase2'] = run_phase(
        2,
        "PREPROCESSING WITH AUDIT TRAIL",
        "enhanced_preprocessing.py",
        required_input="enhanced_scraped_data.json",
        output_file="preprocessed_data.json"
    )
    
    if not results['phase2']:
        print("\n❌ Workflow aborted: Phase 2 failed")
        return
    
    # Phase 3: Sentiment Classification
    # Check if model exists
    if not check_file_exists('sentiment_model.pkl'):
        print_header("PREPARING SENTIMENT MODEL")
        
        # Check if training data exists
        if not check_file_exists('sentiment_training_data.json'):
            print("📝 Generating training data...")
            os.system("python3 create_training_data.py")
        
        print("🧠 Training sentiment model...")
        os.system("python3 sentiment_model.py")
    
    results['phase3'] = run_phase(
        3,
        "SENTIMENT CLASSIFICATION",
        "sentiment_classifier.py",
        required_input="preprocessed_data.json",
        output_file="classified_sentiment_data.json"
    )
    
    if not results['phase3']:
        print("\n❌ Workflow aborted: Phase 3 failed")
        return
    
    # Phase 4: Temporal Analysis
    results['phase4'] = run_phase(
        4,
        "TEMPORAL TREND ANALYSIS",
        "temporal_analysis.py",
        required_input="classified_sentiment_data.json",
        output_file="temporal_analysis_results.json"
    )
    
    # Print summary
    print_summary(results)
    
    print(f"\n🎉 Workflow completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Workflow error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
