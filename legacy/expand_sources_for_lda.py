#!/usr/bin/env python3
"""
Expand LDA corpus with Bluesky + news data (2018 onwards),
clean using existing preprocessing logic, and merge with current clean dataset.

Output:
- classified_sentiment_data_clean_expanded.json
"""

import json
import re
import hashlib
from datetime import datetime
from collections import Counter
from urllib.parse import quote
import requests


BASE_FILE = 'classified_sentiment_data_clean.json'
OUTPUT_FILE = 'classified_sentiment_data_clean_expanded.json'
TARGET_MIN = 2500
TARGET_MAX = 3000

DOMAIN_KEYWORDS = {
    'smart', 'farming', 'farm', 'agriculture', 'agricultural', 'agtech',
    'livestock', 'dairy', 'cattle', 'cow', 'sensor', 'monitoring', 'automation',
    'robot', 'precision', 'ai', 'machine', 'learning', 'computer', 'vision'
}

BLUESKY_QUERIES = [
    '"smart farming" livestock',
    '"precision agriculture" ai',
    '"livestock monitoring" ai',
    '"dairy farming" automation',
    '"agtech" "machine learning"',
    '"farm automation" livestock',
]

NEWS_QUERIES = [
    '"smart farming" OR "precision agriculture" OR "agtech"',
    '"livestock" AND ("AI" OR "automation" OR "monitoring")',
    '"dairy" AND ("AI" OR "smart" OR "sensor")',
    '"farm automation" OR "digital agriculture"',
    '"machine learning" agriculture',
    '"computer vision" livestock',
    '"predictive analytics" farming',
    '"iot" farm sensors',
    '"robotics" agriculture',
    '"precision dairy" technology',
]


def parse_date_to_yyyy_mm_dd(value: str):
    if not value:
        return None
    candidates = [
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
        '%a, %d %b %Y %H:%M:%S %Z',
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            pass
    # Best-effort fallback
    m = re.search(r'(\d{4}-\d{2}-\d{2})', value)
    return m.group(1) if m else None


def is_2018_or_later(date_str: str):
    if not date_str:
        return False
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').year >= 2018
    except Exception:
        return False


def domain_relevant(text: str, min_hits: int = 2):
    text_l = (text or '').lower()
    hits = sum(1 for kw in DOMAIN_KEYWORDS if kw in text_l)
    return hits >= min_hits


def normalize_spaces(text: str):
    return re.sub(r'\s+', ' ', (text or '').strip())


def clean_text(text: str):
    text = normalize_spaces(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def stable_hash(title: str, raw_text: str):
    key = f"{(title or '').lower().strip()}||{(raw_text or '').lower().strip()}"
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def tokenize_for_word_count(cleaned: str):
    tokens = [t for t in cleaned.split() if t.isalpha() and len(t) > 2]
    return tokens


def load_base_data(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    return data


def collect_bluesky_posts():
    print('\nCollecting Bluesky posts (public API)...')
    base_url = 'https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts'
    rows = []

    for q in BLUESKY_QUERIES:
        cursor = None
        pages = 0
        while pages < 3:  # keep conservative; avoid aggressive scraping
            params = {
                'q': q,
                'limit': 100,
                'sort': 'latest'
            }
            if cursor:
                params['cursor'] = cursor
            try:
                r = requests.get(base_url, params=params, timeout=20)
                if r.status_code != 200:
                    break
                payload = r.json()
                posts = payload.get('posts', [])
                if not posts:
                    break

                for p in posts:
                    record = (p.get('record') or {})
                    text = normalize_spaces(record.get('text', ''))
                    indexed = p.get('indexedAt') or p.get('createdAt') or ''
                    created = parse_date_to_yyyy_mm_dd(indexed)
                    if not created or not is_2018_or_later(created):
                        continue
                    if not domain_relevant(text):
                        continue

                    url = p.get('uri', '')
                    handle = (p.get('author') or {}).get('handle', '')
                    title = text[:140]
                    rows.append({
                        'source': 'bluesky',
                        'subreddit': f'@{handle}' if handle else 'bluesky',
                        'title': title,
                        'text': text,
                        'raw_text': f"{title} {text}".strip(),
                        'score': int((p.get('likeCount') or 0) + (p.get('repostCount') or 0)),
                        'num_comments': int(p.get('replyCount') or 0),
                        'created_date': created,
                        'url': url,
                    })

                cursor = payload.get('cursor')
                pages += 1
                if not cursor:
                    break
            except Exception:
                break

    print(f"  Bluesky candidates: {len(rows)}")
    return rows


def collect_news_posts():
    print('\nCollecting news posts (Google News RSS + GDELT)...')
    rows = []

    # Google News RSS (no key)
    for q in NEWS_QUERIES:
        try:
            rss_url = f"https://news.google.com/rss/search?q={quote(q + ' after:2018-01-01')}&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(rss_url, timeout=20)
            if r.status_code != 200:
                continue

            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            for item in root.findall('.//item'):
                title = normalize_spaces((item.findtext('title') or '').replace(' - Google News', ''))
                link = normalize_spaces(item.findtext('link') or '')
                pub = parse_date_to_yyyy_mm_dd(item.findtext('pubDate') or '')
                descr = normalize_spaces(item.findtext('description') or '')
                text = f"{title} {descr}".strip()

                if not pub or not is_2018_or_later(pub):
                    continue
                if not domain_relevant(text, min_hits=1):
                    continue

                rows.append({
                    'source': 'news',
                    'subreddit': 'news',
                    'title': title,
                    'text': text,
                    'raw_text': text,
                    'score': 0,
                    'num_comments': 0,
                    'created_date': pub,
                    'url': link,
                })
        except Exception:
            pass

    # GDELT DOC API (no key) - yearly windows to broaden coverage
    current_year = datetime.utcnow().year
    for q in NEWS_QUERIES:
        for year in range(2018, current_year + 1):
            try:
                start_dt = f"{year}0101000000"
                end_dt = f"{year}1231235959"
                gdelt_url = (
                    'https://api.gdeltproject.org/api/v2/doc/doc'
                    f'?query={quote(q)}&mode=ArtList&maxrecords=180&format=json'
                    f'&startdatetime={start_dt}&enddatetime={end_dt}'
                    '&sort=DateDesc'
                )
                r = requests.get(gdelt_url, timeout=25)
                if r.status_code != 200:
                    continue
                articles = r.json().get('articles', [])
                for a in articles:
                    title = normalize_spaces(a.get('title', ''))
                    url = normalize_spaces(a.get('url', ''))
                    seen = parse_date_to_yyyy_mm_dd(a.get('seendate', ''))
                    domain = normalize_spaces(a.get('domain', ''))
                    text = f"{title} {domain}".strip()

                    if not seen or not is_2018_or_later(seen):
                        continue
                    if not domain_relevant(text, min_hits=1):
                        continue

                    rows.append({
                        'source': 'news',
                        'subreddit': domain or 'news',
                        'title': title,
                        'text': text,
                        'raw_text': text,
                        'score': 0,
                        'num_comments': 0,
                        'created_date': seen,
                        'url': url,
                    })
            except Exception:
                pass

    print(f"  News candidates: {len(rows)}")
    return rows


def classify_sentiment_if_possible(rows):
    """Optional: label new rows with existing sentiment model for compatibility."""
    try:
        import pickle
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        for r in rows:
            txt = r.get('clean_text', '')
            vec = vectorizer.transform([txt])
            pred = model.predict(vec)[0]
            conf = float(model.predict_proba(vec)[0].max())
            r['sentiment'] = pred
            r['sentiment_confidence'] = conf
        return True
    except Exception:
        for r in rows:
            r['sentiment'] = r.get('sentiment', 'neutral')
            r['sentiment_confidence'] = float(r.get('sentiment_confidence', 0.0))
        return False


def main():
    print('=' * 80)
    print('EXPAND LDA CORPUS WITH BLUESKY + NEWS (2018+)')
    print('=' * 80)

    base = load_base_data(BASE_FILE)
    print(f"Loaded base clean dataset: {len(base)} posts from {BASE_FILE}")

    # Build dedupe index from existing data
    existing_urls = set((x.get('url') or '').strip() for x in base if x.get('url'))
    existing_hashes = set(stable_hash(x.get('title', ''), x.get('raw_text', '') or x.get('text', '')) for x in base)

    bluesky = collect_bluesky_posts()
    news = collect_news_posts()

    incoming = bluesky + news
    print(f"\nTotal incoming candidates: {len(incoming)}")

    # Normalize + clean + filter
    kept = []
    for r in incoming:
        created = parse_date_to_yyyy_mm_dd(r.get('created_date', ''))
        if not created or not is_2018_or_later(created):
            continue

        raw = normalize_spaces(r.get('raw_text') or f"{r.get('title', '')} {r.get('text', '')}")
        cleaned = clean_text(raw)
        tokens = tokenize_for_word_count(cleaned)
        if len(tokens) < 5:
            continue

        row_hash = stable_hash(r.get('title', ''), raw)
        if (r.get('url') and r.get('url') in existing_urls) or row_hash in existing_hashes:
            continue

        # extend in current schema
        item = {
            'source': r.get('source', 'external'),
            'subreddit': r.get('subreddit', 'external'),
            'title': r.get('title', '')[:500],
            'text': r.get('text', ''),
            'raw_text': raw,
            'score': int(r.get('score', 0) or 0),
            'num_comments': int(r.get('num_comments', 0) or 0),
            'created_date': created,
            'url': r.get('url', ''),
            'clean_text': cleaned,
            'tokens': ' '.join(tokens),
            'word_count': len(tokens),
        }
        kept.append(item)
        existing_hashes.add(row_hash)
        if item.get('url'):
            existing_urls.add(item['url'])

    print(f"Usable new clean posts after filtering/dedup: {len(kept)}")

    # Add sentiment labels if model exists
    classified = classify_sentiment_if_possible(kept)
    print('Sentiment model applied to new posts.' if classified else 'Sentiment model unavailable; default sentiment fields used.')

    merged = base + kept

    # keep within requested usable range if needed
    if len(merged) > TARGET_MAX:
        # Preserve all original base posts, trim additions by recency descending
        kept_sorted = sorted(kept, key=lambda x: x.get('created_date', ''), reverse=True)
        allowed_new = max(0, TARGET_MAX - len(base))
        kept_sorted = kept_sorted[:allowed_new]
        merged = base + kept_sorted
        kept = kept_sorted

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    by_source = Counter(x.get('source', 'unknown') for x in merged)
    years = Counter()
    for x in merged:
        d = x.get('created_date', '')
        if re.match(r'^\d{4}-\d{2}-\d{2}$', d):
            years[d[:4]] += 1

    print('\n' + '=' * 80)
    print('EXPANSION SUMMARY')
    print('=' * 80)
    print(f"Base clean posts: {len(base)}")
    print(f"Added clean posts: {len(kept)}")
    print(f"Final usable posts: {len(merged)}")
    print(f"Target met (2500-3000): {TARGET_MIN <= len(merged) <= TARGET_MAX}")
    print(f"Saved: {OUTPUT_FILE}")
    print('\nBy source:')
    for s, n in by_source.items():
        print(f"  - {s}: {n}")
    if years:
        print(f"\nYear range: {min(years.keys())} to {max(years.keys())}")


if __name__ == '__main__':
    main()
