#!/usr/bin/env python3
"""Find all subreddits not yet categorised"""
import json
from collections import Counter

data = json.load(open('classified_sentiment_data.json'))
subs = Counter(item.get('subreddit', '') for item in data)

already_kept_lower = {
    'farming','agriculture','cattle','ranching','fishfarming','llamafarm',
    'cultivatenation','precisionag','regenerativeag','hydroponics','verticalfarming',
    'farminguk','agriculturearticles','aquaculture','sheep','agricultureaust',
    'agriminds','clarksonsfarm','vegetablegardening','agronomics_investors',
    'biochartoday','australiancattledog','equestrian','permaculture',
    'greenhouse','lawncare','mycology','conservation','animalwelfarescience',
    'askvet','leopardsatemyfarm','satechnews','stew_sctecengworld',
    'u_farmingthebestgreens','lora','u_ozrobotics','dairy','livestock',
    'homestead','homesteading','agtech','remotesensing','uavmapping',
    'droneinspection','allthingsdrones','geospatial_data','gis','askrobotics',
    'raspberry_pi_projects','plc','esp32','esphome','diyelectronics','embedded',
    'ros','digitaltwin','videosurveillance','homeautomation','iot','robotics',
    'deeplearning','localllama','genai4all','ai_india','n8n','machinelearningjobs',
    'machinelearningnews','ml_news','ai4civilengineering','artificialnteligence',
    'aiandrobotics','aitrailblazers','openai','claudeai','deepseek','ai_agents',
    'aiagents','aipromptprogramming','secourses','machinelearning',
    'artificialintelligence','learnmachinelearning','debateavegan','antivegan',
    'askvegans','veganfood','food','environmentalnews','climate','climatechange',
    'environment','climateoffensive','climateactionplan','environmentalism',
    'ecouplift','betterbioeconomy','carboncredits','bird_flu_now','microbiology',
    'parasitology','reeftank','biologypreprints','worldnews','news',
    'everythingscience','tech','technews','futurism','damnthatsinteresting',
    'interestingasfuck','dataisbeautiful','collapse','economiccollapse',
    'anticonsumption','autonewspaper','businesstodaynews','nairobitechies',
    'sino','andhra_pradesh','newzealand','futurology','askscience','technology',
    'entrepreneur','sideproject','startups_promotion','gostartupindia',
    'startupindia','saas','cofounderhunt','hwstartups','angelinvesting',
    'investors','agronomics_investors','startups','smallbusiness',
}

print("Subreddits NOT yet categorised:")
for sub, cnt in sorted(subs.items(), key=lambda x: -x[1]):
    if sub.lower() not in already_kept_lower:
        print(f"  r/{sub}: {cnt}")
