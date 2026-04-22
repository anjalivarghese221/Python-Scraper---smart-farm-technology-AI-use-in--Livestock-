#!/usr/bin/env python3
"""
Build a high-coverage cleaned corpus from classified_sentiment_data.json.
Goal: keep dataset size > 2500 while removing obvious noise/spam terms
(e.g., "island" artifacts, exam-cheating spam).
"""

import json
import re
import argparse

INPUT = 'classified_sentiment_data.json'
OUTPUT = 'classified_sentiment_data_clean_high_coverage.json'

TEXT_BLACKLIST_PATTERNS = [
    r'\bisland\b',
    r'clep\s+exam',
    r'proctortrack',
    r'hiraedu',
    r'whatsapp\s*:\s*\+?\d+',
    r'take\s+my\s+exam',
    r'pay\s+someone\s+to\s+take',
    r'exam\s+helper',
    r'\bstate\s+farm\b',
    r'oppo\s+find\s+x5\s+pro',
    r'stormwoes',
    r'clash\s+of\s+clans',
    r'\btotk\b',
    r'\blast\s*fm\b|\blastfm\b',
    r'\bscrobbl(?:e|es|ing)\b',
    r'\bk\-?pop\b',
    r'\bjimin\b',
    r'\bj\-?hope\b',
    r'fitness\s+app\s+development',
    r'\bmyfitnesspal\b',
    r'\bpeloton\b',
    r'\bstrava\b',
    r'\bfitbit\b',
]

SUBREDDIT_BLACKLIST = {
    'clepexam',
    'stormworks',
    'lastfm',
}


def row_text(row):
    return (
        (row.get('title', '') or '') + ' ' +
        (row.get('text', '') or '') + ' ' +
        (row.get('raw_text', '') or '') + ' ' +
        (row.get('subreddit', '') or '')
    ).lower()


def is_noise(row):
    sub = (row.get('subreddit', '') or '').lower()
    if sub in SUBREDDIT_BLACKLIST:
        return True

    txt = row_text(row)
    for pat in TEXT_BLACKLIST_PATTERNS:
        if re.search(pat, txt):
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=INPUT, help='Input classified sentiment JSON')
    parser.add_argument('--output', default=OUTPUT, help='Output cleaned JSON')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = [row for row in data if not is_noise(row)]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f'input_file={args.input}')
    print(f'output_file={args.output}')
    print(f'input_count={len(data)}')
    print(f'output_count={len(cleaned)}')
    print(f'removed={len(data)-len(cleaned)}')


if __name__ == '__main__':
    main()
