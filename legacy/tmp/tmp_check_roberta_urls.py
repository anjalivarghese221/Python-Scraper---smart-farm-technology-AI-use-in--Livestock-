import json
from pathlib import Path

urls = [
    'https://www.reddit.com/r/news/comments/1dx4x6p/twelve_threats_of_precision_livestock_farming_plf/',
    'https://www.reddit.com/r/news/comments/1ksx2qk/rethink_language_learning_the_shortcomings_of_ai/',
    'https://www.reddit.com/r/climatechange/comments/1jzfw88/hypothetical_if_precision_fermentation_actually/',
    'https://www.reddit.com/r/FarmingUK/comments/1nb4vp2/farming_software_curiosity/',
    'https://www.reddit.com/r/PrecisionAg/comments/1kmzkkd/the_cattle_tech_adoption_paradox_why_is/',
]

rows = json.loads(Path('classified_sentiment_data_roberta.json').read_text(encoding='utf-8'))
by_url = {r.get('url'): r for r in rows}

for u in urls:
    r = by_url.get(u)
    if not r:
        print('\nMISSING', u)
        continue
    print(f"\nURL: {u}")
    print(f"sentiment={r.get('sentiment')} conf={r.get('sentiment_confidence')}")
    print(f"sub=r/{r.get('subreddit')} date={r.get('created_date')}")
    print(f"title={r.get('title')}")
