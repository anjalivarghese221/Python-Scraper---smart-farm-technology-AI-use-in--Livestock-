#!/usr/bin/env python3
"""Copy current visualization artifacts into visualizations/<target_folder>."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, help='Folder name under visualizations/, e.g., logistic or roberta')
    parser.add_argument('--clean', action='store_true', help='Clear target folder before copying')
    args = parser.parse_args()

    base = Path('visualizations')
    target_dir = base / args.target
    target_topic = target_dir / 'topic_modeling'

    target_topic.mkdir(parents=True, exist_ok=True)

    if args.clean and target_dir.exists():
        for p in target_dir.rglob('*'):
            if p.is_file():
                p.unlink()

    copied = 0

    # Copy top-level pngs
    for p in base.glob('*.png'):
        shutil.copy2(p, target_dir / p.name)
        copied += 1

    # Copy topic_modeling pngs
    src_topic = base / 'topic_modeling'
    if src_topic.exists():
        for p in src_topic.glob('*.png'):
            shutil.copy2(p, target_topic / p.name)
            copied += 1

    print(f'Copied {copied} visualization files to {target_dir}')


if __name__ == '__main__':
    main()
