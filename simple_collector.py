#!/usr/bin/env python3
"""
Phase 1 Reddit-only data collection with multiple query strategies.

Implements the query-design guidance:
- Use multiple strategies equivalent to Boolean logic (via lexical expansion)
- Keep source restricted to Reddit only
- Focus on AI/smart-farming discourse in livestock systems
"""

import json
import os
import re
import time
from datetime import datetime
from itertools import product

import requests


USER_AGENT = 'SmartFarmResearch/1.0'
OUTPUT_FILE = 'enhanced_scraped_data.json'
QUERY_LOG_FILE = 'query_log.json'
MIN_YEAR = 2018
MAX_QUERIES_PER_STRATEGY = 12
TARGET_MIN_ITEMS = 2600
MAX_COMMENT_THREADS = 450
MAX_COMMENTS_PER_THREAD = 5
REUSE_EXISTING_BASE = True
RELEVANT_SUBREDDITS = {'agtech', 'farming', 'agriculture', 'livestock', 'dairyfarming', 'precisionagriculture'}


def build_query_plan():
    """Build multiple query strategies for relevance-focused scraping."""
    livestock_terms = ['livestock', 'dairy', 'cattle', 'beef', 'poultry']
    technology_terms = [
        'ai',
        'artificial intelligence',
        'smart farming',
        'precision livestock farming',
        'automation',
        'sensor',
        'machine learning',
        'computer vision',
    ]
    practice_terms = ['monitoring', 'health tracking', 'feeding optimization', 'farm management']

    # Boolean logic examples used in this pipeline:
    # 1) (livestock OR dairy OR cattle) AND (ai OR artificial intelligence OR machine learning)
    # 2) (livestock OR dairy) AND (smart farming OR precision livestock farming OR automation)
    # 3) (cattle OR dairy) AND (computer vision OR sensor OR iot) AND (monitoring OR health tracking)
    # 4) (beef OR poultry OR livestock) AND (ai OR smart farming) AND (farm management OR feeding optimization)

    # Strategy A: explicit lexical expansion for
    # (livestock OR dairy ...) AND (AI/smart-farming term)
    strategy_a = [f'{l} {t}' for l, t in product(livestock_terms, technology_terms)]

    # Strategy B: 3-way contextual expansion for tech + livestock + farm-practice context
    strategy_b = [f'{t} {l} {p}' for t, l, p in product(technology_terms[:4], livestock_terms[:4], practice_terms[:2])]

    # Strategy C: phrase-focused queries used in research discussions
    strategy_c = [
        'ai in livestock farming',
        'smart farming livestock',
        'precision livestock farming ai',
        'computer vision cattle monitoring',
        'iot sensors dairy farm',
        'robotics dairy automation',
        'machine learning livestock health',
        'smart dairy farm technology',
        'ai animal welfare monitoring',
        'digital livestock farming',
    ]

    # keep list unique while preserving order
    seen = set()
    all_queries = []
    per_strategy_count = {}
    for strategy_name, queries in [
        ('lexical_expansion_livestock_x_ai', strategy_a),
        ('tech_context_expansion', strategy_b),
        ('phrase_targeted_queries', strategy_c),
    ]:
        per_strategy_count[strategy_name] = 0
        for q in queries:
            if per_strategy_count[strategy_name] >= MAX_QUERIES_PER_STRATEGY:
                continue
            qn = q.strip().lower()
            if qn and qn not in seen:
                seen.add(qn)
                all_queries.append({'strategy': strategy_name, 'query': qn, 'limit': 75})
                per_strategy_count[strategy_name] += 1

    return all_queries


def scrape_reddit_search(query, limit=100, subreddit=None):
    """Reddit JSON API search with optional subreddit restriction."""
    url = 'https://www.reddit.com/search.json'
    params = {
        'q': query,
        'limit': limit,
        'sort': 'relevance',
        't': 'all',
    }
    if subreddit:
        params['restrict_sr'] = 'on'
        params['q'] = query
        url = f'https://www.reddit.com/r/{subreddit}/search.json'

    headers = {'User-Agent': USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            print(f'  HTTP {response.status_code} for query="{query}"')
            return [], response.status_code

        data = response.json()
        posts = []
        for child in data.get('data', {}).get('children', []):
            post_data = child.get('data', {})
            created_utc = post_data.get('created_utc', 0)
            post_date = datetime.fromtimestamp(created_utc)
            if post_date.year < MIN_YEAR:
                continue

            title = post_data.get('title', '')
            text = post_data.get('selftext', '')
            raw_text = f'{title} {text}'.strip()

            posts.append({
                'source': 'reddit',
                'item_type': 'post',
                'subreddit': post_data.get('subreddit', ''),
                'title': title,
                'text': text,
                'raw_text': raw_text,
                'score': post_data.get('score', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_date': post_date.strftime('%Y-%m-%d'),
                'url': post_data.get('url', ''),
            })
        return posts, 200
    except Exception as exc:
        print(f'  Error for query="{query}": {exc}')
        return [], None


def _extract_comment_items(comment_children, post, collected, seen_comment_ids, max_comments):
    for child in comment_children:
        if len(collected) >= max_comments:
            return

        if child.get('kind') != 't1':
            continue

        comment_data = child.get('data', {})
        comment_id = comment_data.get('id')
        if not comment_id or comment_id in seen_comment_ids:
            continue

        body = (comment_data.get('body') or '').strip()
        if not body or body in {'[deleted]', '[removed]'}:
            continue

        seen_comment_ids.add(comment_id)
        comment_url = comment_data.get('permalink') or post.get('url', '')
        if comment_url and comment_url.startswith('/'):
            comment_url = f'https://www.reddit.com{comment_url}'

        collected.append({
            'source': 'reddit',
            'item_type': 'comment',
            'subreddit': post.get('subreddit', ''),
            'title': post.get('title', ''),
            'text': body,
            'raw_text': f"{post.get('title', '')} {body}".strip(),
            'score': comment_data.get('score', 0),
            'num_comments': len(comment_data.get('replies', {}).get('data', {}).get('children', [])) if isinstance(comment_data.get('replies'), dict) else 0,
            'created_date': datetime.fromtimestamp(comment_data.get('created_utc', 0)).strftime('%Y-%m-%d') if comment_data.get('created_utc') else post.get('created_date', ''),
            'url': comment_url,
            'parent_post_url': post.get('url', ''),
            'parent_post_title': post.get('title', ''),
            'comment_id': comment_id,
            'parent_id': comment_data.get('parent_id', ''),
            'comment_depth': comment_data.get('depth', 0),
        })

        replies = comment_data.get('replies')
        if isinstance(replies, dict) and len(collected) < max_comments:
            reply_children = replies.get('data', {}).get('children', [])
            _extract_comment_items(reply_children, post, collected, seen_comment_ids, max_comments)


def scrape_reddit_comments(post, max_comments=MAX_COMMENTS_PER_THREAD):
    """Fetch top comments from a Reddit post permalink."""
    post_url = (post.get('url') or '').strip()
    if not post_url.startswith('https://www.reddit.com/'):
        return [], None

    comment_url = post_url.rstrip('/') + '.json'
    params = {
        'sort': 'top',
        'limit': max_comments,
        'depth': 2,
        'raw_json': 1,
    }
    headers = {'User-Agent': USER_AGENT}

    try:
        response = None
        for attempt in range(2):
            response = requests.get(comment_url, params=params, headers=headers, timeout=20)
            if response.status_code != 429:
                break
            wait_seconds = 5 * (attempt + 1)
            print(f'  HTTP 429 for comments on "{post_url}"; retrying in {wait_seconds}s')
            time.sleep(wait_seconds)

        if response is None or response.status_code != 200:
            if response is not None and response.status_code != 200:
                print(f'  HTTP {response.status_code} for comments on "{post_url}"')
            return [], response.status_code if response is not None else None

        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            return [], 200

        comment_listing = payload[1].get('data', {}).get('children', [])
        collected = []
        seen_comment_ids = set()
        _extract_comment_items(comment_listing, post, collected, seen_comment_ids, max_comments)
        return collected, 200
    except Exception as exc:
        print(f'  Error fetching comments for "{post_url}": {exc}')
        return [], None


def load_existing_posts():
    if not REUSE_EXISTING_BASE or not os.path.exists(OUTPUT_FILE):
        return []

    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        posts = data.get('posts', []) if isinstance(data, dict) else data
        if isinstance(posts, list):
            return posts
    except Exception:
        return []

    return []


def is_relevant(post):
    """Keep posts tied to livestock context + AI/smart-farming context."""
    text = (
        f"{post.get('title', '')} "
        f"{post.get('text', '')} "
        f"{post.get('url', '')} "
        f"{post.get('subreddit', '')}"
    ).lower()

    livestock_terms = ['dairy', 'livestock', 'cattle', 'beef', 'ruminant', 'goat', 'sheep']
    tech_terms = [
        'ai',
        'artificial intelligence',
        'smart farming',
        'precision livestock',
        'automation',
        'robot',
        'sensor',
        'iot',
        'machine learning',
        'computer vision',
        'algorithm',
        'predictive',
    ]

    livestock_hits = sum(1 for term in livestock_terms if re.search(rf"\b{re.escape(term)}\b", text))
    tech_hits = sum(1 for term in tech_terms if re.search(rf"\b{re.escape(term)}\b", text))

    score = livestock_hits + tech_hits

    # Query/search context is often stronger than the post body on Reddit search.
    subreddit = (post.get('subreddit', '') or '').lower()
    if subreddit in {'agtech', 'farming', 'agriculture', 'livestock', 'dairyfarming', 'precisionagriculture'}:
        score += 1

    if 'livestock' in text or 'dairy' in text or 'cattle' in text:
        score += 1
    if 'ai' in text or 'artificial intelligence' in text or 'machine learning' in text or 'computer vision' in text:
        score += 1

    return score >= 2


def main():
    print('=' * 80)
    print('PHASE 1: REDDIT-ONLY COLLECTION (MULTI-QUERY STRATEGIES)')
    print('=' * 80)

    query_plan = build_query_plan()
    focused_subreddits = [
        'AgTech',
        'farming',
        'agriculture',
        'livestock',
        'dairyfarming',
        'precisionagriculture',
        'MachineLearning',
        'artificial',
    ]

    all_posts = load_existing_posts()
    seen_urls = {item.get('url') for item in all_posts if item.get('url')}
    query_log = []

    if all_posts:
        print(f"\nReusing existing base corpus: {len(all_posts)} items from {OUTPUT_FILE}")

    if not all_posts:
        print(f"\nSubreddit-restricted collection: {len(focused_subreddits)} subreddits × {len(query_plan)} queries")
        total_restricted = len(focused_subreddits) * len(query_plan)
        progress = 0

        for subreddit in focused_subreddits:
            for item in query_plan:
                query = item['query']
                strategy = item['strategy']
                limit = item['limit']
                progress += 1
                print(f"[{progress}/{total_restricted}] r/{subreddit} :: '{query}'")
                posts, status_code = scrape_reddit_search(query=query, limit=limit, subreddit=subreddit)

                kept = 0
                for post in posts:
                    if not is_relevant(post):
                        continue
                    if post['url'] and post['url'] in seen_urls:
                        continue
                    if post['url']:
                        seen_urls.add(post['url'])
                    all_posts.append(post)
                    kept += 1

                query_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'strategy': f'subreddit_restricted_{strategy}',
                    'subreddit': subreddit,
                    'parameters': {
                        'limit': limit,
                        'sort': 'relevance',
                        't': 'all',
                        'restrict_sr': 'on',
                    },
                    'http_status': status_code,
                    'retrieved': len(posts),
                    'kept_after_relevance_filter': kept,
                    'api_tier': 'Reddit JSON API (public)',
                })

                print(f'  retrieved={len(posts)} | kept={kept} | running_total={len(all_posts)}')
                time.sleep(1)

    # Comment expansion pass: use replies from the strongest threads to push the corpus over the target size.
    if len(all_posts) < TARGET_MIN_ITEMS:
        candidate_posts = [
            p for p in all_posts
            if p.get('item_type', 'post') == 'post'
            and p.get('url', '').startswith('https://www.reddit.com/')
            and p.get('num_comments', 0) > 0
            and is_relevant(p)
        ]

        if len(candidate_posts) < 150:
            candidate_posts = [
                p for p in all_posts
                if p.get('item_type', 'post') == 'post'
                and p.get('url', '').startswith('https://www.reddit.com/')
                and p.get('num_comments', 0) > 0
                and (p.get('subreddit', '') or '').lower() in RELEVANT_SUBREDDITS
            ]

        candidate_posts = sorted(
            candidate_posts,
            key=lambda x: (int(x.get('num_comments', 0)), int(x.get('score', 0))),
            reverse=True,
        )
        comment_threads = candidate_posts[:MAX_COMMENT_THREADS]
        print(f"\nComment expansion pass: {len(comment_threads)} threads (target total: {TARGET_MIN_ITEMS})")

        comment_count = 0
        for idx, post in enumerate(comment_threads, 1):
            if len(all_posts) + comment_count >= TARGET_MIN_ITEMS:
                break

            print(f"[{idx}/{len(comment_threads)}] comments from r/{post.get('subreddit', '')} :: '{post.get('title', '')[:80]}'")
            comments, status_code = scrape_reddit_comments(post, max_comments=MAX_COMMENTS_PER_THREAD)
            kept = 0
            for comment in comments:
                if comment.get('url') and comment.get('url') in seen_urls:
                    continue
                if comment.get('url'):
                    seen_urls.add(comment['url'])
                all_posts.append(comment)
                kept += 1
                comment_count += 1
                if len(all_posts) >= TARGET_MIN_ITEMS:
                    break

            query_log.append({
                'timestamp': datetime.now().isoformat(),
                'query': post.get('url', ''),
                'strategy': 'comment_expansion_top_threads',
                'subreddit': post.get('subreddit', ''),
                'parameters': {
                    'limit': MAX_COMMENTS_PER_THREAD,
                    'sort': 'top',
                    'depth': 2,
                },
                'http_status': status_code,
                'retrieved': len(comments),
                'kept_after_relevance_filter': kept,
                'api_tier': 'Reddit JSON API comments (public)',
            })
            print(f'  retrieved={len(comments)} | kept={kept} | running_total={len(all_posts)}')
            time.sleep(0.5)

    output = {
        'posts': all_posts,
        'metadata': {
            'source': 'reddit_only',
            'collection_date': datetime.now().strftime('%Y-%m-%d'),
            'min_year': MIN_YEAR,
            'total': len(all_posts),
            'post_count': sum(1 for item in all_posts if item.get('item_type', 'post') == 'post'),
            'comment_count': sum(1 for item in all_posts if item.get('item_type') == 'comment'),
            'query_strategies': sorted(list({f"subreddit_restricted_{item['strategy']}" for item in query_plan})),
            'relevance_rule': 'must include livestock-context term and AI/smart-farming term',
        },
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(QUERY_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(query_log, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 80)
    print('COLLECTION COMPLETE')
    print('=' * 80)
    print(f'Total kept posts: {len(all_posts)}')
    print(f'Saved dataset: {OUTPUT_FILE}')
    print(f'Saved query log: {QUERY_LOG_FILE}')


if __name__ == '__main__':
    main()
