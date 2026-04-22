import json,re
from pathlib import Path

rows=json.loads(Path('classified_sentiment_data_roberta.json').read_text())

pat_pairs=[
    re.compile(r'precision\s+livestock',re.I),
    re.compile(r'(dairy|livestock|cattle|farm|farming|agriculture).*(ai|machine learning|automation|robot|iot|sensor|precision)|((ai|machine learning|automation|robot|iot|sensor|precision).*(dairy|livestock|cattle|farm|farming|agriculture))',re.I),
]

bad=re.compile(r'epstein|lottery|scam|uber|game|yogscast|gift ideas|worldbuilding|metaverse|gangstalking|cathelp|phasmophobia|mmo|conspiracy',re.I)

cand=[]
for r in rows:
    if (r.get('sentiment') or '').lower()!='negative':
        continue
    url=r.get('url') or ''
    if not url.startswith('https://www.reddit.com/'):
        continue
    text=' '.join([(r.get('title') or ''),(r.get('text') or ''),(r.get('raw_text') or ''),(r.get('subreddit') or '')])
    if bad.search(text):
        continue
    if any(p.search(text) for p in pat_pairs):
        cand.append(r)

cand=sorted(cand,key=lambda x:x.get('sentiment_confidence',0),reverse=True)
print('count',len(cand))
for i,r in enumerate(cand[:40],1):
    print(f"\n[{i}] conf={r.get('sentiment_confidence'):.4f} date={r.get('created_date')} sub=r/{r.get('subreddit')}")
    print('title:',(r.get('title') or '').replace('\n',' '))
    print('url:',r.get('url'))
