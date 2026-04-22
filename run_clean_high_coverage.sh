#!/bin/zsh
set -e
cd /Users/anjalivarghese/Python-Scraper---smart-farm-technology-AI-use-in--Livestock-

python3 build_clean_high_coverage_corpus.py
python3 topic_stability_analysis_final.py
python3 network_analysis.py
python3 temporal_analysis.py
python3 network_visualizer.py
python3 semantic_drivers_visualizer.py
