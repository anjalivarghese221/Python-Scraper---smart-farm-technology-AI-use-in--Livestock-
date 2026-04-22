import json,re
from pathlib import Path

data=json.loads(Path('classified_sentiment_data_roberta.json').read_text())

livestock=re.compile(r"\b(livestock|dairy|cattle|cow|cows|sheep|goat|goats|swine|pig|pigs|poultry|herd|beef|milk)\b",re.I)
ai=re.compile(r"\b(ai|artificial intelligence|machine learning|computer vision|iot|sensor|sensors|robot|robotics|automation|automated|precision)\b",re.I)

exclude=re.compile(r"\b(game|genshin|epstein|lottery|uber|reef|cathelp|phasmophobia|mmo|yogscast|codingjobs|job|hiring|openai store|transitindia)\b",re.I)

whitelist_subs={'agriculture','farming','livestock','dairy','dairyfarming','agtech','precisionag','meridairy','farminguk','iot','news','machinelearning'}

cand=[]
for r in data:
    if (r.get('sentiment') or '').lower()!='negative':
        continue
    text=' '.join([(r.get('title') or ''),(r.get('text') or ''),(r.get('raw_text') or ''),(r.get('subreddit') or '')])
    sub=(r.get('subreddit') or '').lower()
    if not (livestock.search(text) and ai.search(text)):
        continue
    if exclude.search(text):
        continue
    if sub not in whitelist_subs and not re.search(r'precision livestock|dairy|livestock|cattle',text,re.I):
        continue
    cand.append(r)

cand=sorted(cand,key=lambda x:x.get('sentiment_confidence',0),reverse=True)
print('count',len(cand))
for i,r in enumerate(cand[:20],1):
    t=(r.get('title') or '').replace('\n',' ')
    c=(r.get('text') or r.get('raw_text') or '').replace('\n',' ')
    if len(c)>220:c=c[:220]+'...'
    print(f"\n[{i}] conf={r.get('sentiment_confidence'):.4f} date={r.get('created_date')} sub=r/{r.get('subreddit')}")
    print('title:',t)
    print('content:',c)
    print('url:',r.get('url'))
