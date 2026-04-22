#!/usr/bin/env python3
"""
Clean dataset by removing known contamination sources.
Strategy: blocklist-only — exclude confirmed gaming/fiction/drama subreddits.
All other posts (found by the scraper searching ag/AI keywords) are kept.
Case-insensitive matching.
"""
import json
from collections import Counter

# -----------------------------------------------------------------------
# EXCLUDE: subreddits confirmed to be gaming, sci-fi fiction, drama,
#          meme-stocks, job-board spam, or pure off-topic noise.
#          Every post in these subs matched the scraper keywords only
#          by coincidence (e.g. "livestock" in a game, "farm" in sci-fi).
# -----------------------------------------------------------------------
EXCLUDE_SUBREDDITS_LOWER = {
    # ── Sci-fi / alien fiction ──────────────────────────────────────────
    'hfy',                        # "Humanity F*** Yeah" short stories
    'natureofpredators',          # alien-livestock sci-fi series
    'realms_of_omnarai',          # fantasy fiction
    'nosleep',                    # horror fiction
    'humansarespaceorcs',         # sci-fi fiction
    'spacecannibalism',           # sci-fi fiction
    'scifiwriting',               # sci-fi writing community
    'writingprompts',             # fiction writing prompts
    'worldbuilding',              # fiction worldbuilding
    'goodworldbuilding',          # fiction worldbuilding
    'militaryworldbuilding',      # fiction worldbuilding

    # ── Farming / base-building VIDEO GAMES ────────────────────────────
    'stellaris',                  # space grand-strategy game
    'farmingsimulator',           # farming game
    'farmingsimulator25',         # farming game
    'stardewvalley',
    'farmsofstardewvalley',
    'stardewvalleyexpanded',
    'stardewvalleymods',
    'manorlords',                 # medieval city builder
    'medievaldynasty',
    'coralisland',
    'farmmanager',
    'fieldsofmistria',
    'fieldsofmistriagame',
    'runefactory',
    'goingmedieval',
    'skyfactory',
    'skyblocky',
    'stoneblock4',
    'projectzomboid',
    'dwarffortress',
    'newworldgame',
    'plateup',
    'victoria3',
    'eu5',
    'factorio',
    'technicalminecraft',
    'minecraft',
    'minecrafthelp',
    'minecraftbuilds',
    'mcpe',
    'createmod',
    'feedthebeast',
    'allthemods',
    'cobblemon',
    'shittymcsuggestions',
    'technicalminecraft',
    'legofortnite',
    'rimworld',
    'theplanetcrafter',
    'astroneer',
    'satisfactorygame',
    'spaceengineers',
    'stationeers',
    'autofarmgames',
    'automationgames',
    'incremental_games',
    'vintagestory',
    'songsofsyx',
    'lookoutsidegame',
    'hytale',
    'hytaleinfo',
    'anno',
    'worldbox',
    'kaiserreich',
    'crusaderkings',
    'civ',
    'eu4',
    'alternatehistory',
    'alternativehistory',
    'imaginarymaps',
    'mapporncirclejerk',

    # ── Other VIDEO GAMES ───────────────────────────────────────────────
    'warframe',
    'leagueoflegends',
    'destinythegame',
    'destiny',
    'arcraiders',
    'arc_raiders',
    'pathofexile',
    'pathofexile2',
    'pathofexilebuilds',
    'deadbydaylight',
    'marvelrivals',
    'gachagaming',
    'honkaistarrail',
    'battlefield',
    'battlefield6',
    'classicwow',
    'nostupidd',
    'nomansskytheGame',
    'nomansskythegame',
    'fallout',
    'fallout4modsps4',
    'fo76',
    'antiwork',
    'gaming',
    'games',
    'pcgaming',
    'pcmasterrace',
    'truegaming',
    'globaloffensive',
    'cs2',
    'csgo',
    'apexlegends',
    'rocketleague',
    'valorantcompetitive',
    'overwatch',
    'deadlockthegame',
    'mortalcombat',
    'brawlstars',
    'dota2',
    'hearthstone',
    'mtg',
    'magicarena',
    'codzmobies',
    'callofdutymoble',
    'callofdutymobi',
    'callofdutyMobile',
    'modernwarfareii',
    'ghostrecon',
    'guildwars2',
    'blackmythwukong',
    'eldenring',
    'baldursgate3',
    'diablo',
    'diablo3',
    'diablo4',
    'metalgearsolid',
    'ffxv',
    'finalfantasy',
    'mariokart',
    'pokemonscarletviolet',
    'pokemontcg',
    'ptcgp',
    'cobblemon',
    'rbx',
    'dresstoimpressroblox',
    'maplestory',
    'silksong',
    'hollowknight',
    'terraria',
    'umamusume',
    'wutheringwaves',
    'loveanddeepspace',
    'corekeepergame',
    'palworld',
    'thefirstdescendant',
    'xenoblade',
    'xenobladechroniclesx',
    'frostpunk',
    '2007scape',
    'runescape',
    'summonerschool',
    'jungle_mains',
    'dndnext',
    'dnd',
    'dmacademy',
    'boardgames',
    'danganronpa',
    'danganandchaos',
    'rwby',
    'aosLore',
    'warhammer40k',
    '40klore',
    'aoe2',
    'classicwow',
    'hearthstone',
    'totalwar',
    'mountandblade',
    'mhwilds',
    'monsterhunter',
    'monsterhunterrage',
    'btd6',
    'zzz_discussion',
    'genshin_impact',
    'genshin_lore',
    'kokomi_mains',
    'yugiohmasterduel',
    'chainsaw',
    'chainsawman',
    'jujutsufolk',
    'onpiece',
    'dccomics',
    'dcleaks',
    'dceuLeaks',
    'marvelstudios',
    'marvelstudiosspoilers',
    'joeRogan',
    'redrising',
    'masseffect',
    'startrek',
    'starwarscantina',
    'legostarwarsleaks',
    'spongtheband',
    'Helldivers',
    'helldivers',
    'abioticfactor',

    # ── Drama compilations & social media drama ────────────────────────
    'bestofredditoru pdates',
    'bestofredditorupdates',
    'bor updates',
    'borupdates',
    'subreditdrama',
    'subredditdrama',
    'hobbydrama',
    'maliciouscompliance',
    'pettyrevenge',
    'tifu',
    'amitheasshole',
    'aitah',
    'amiovereacting',
    'relationship_advice',
    'mildlyinfuriating',
    'twoXchromosomes',
    'twoxpreppers',
    'notliketheothergirls',
    'entitledpeople',
    'storiesaboutkevin',
    'unpopularopinion',
    'childfree',
    'antimlm',
    'survivinginfidelity',
    'traumatizethemback',
    'trueoffmychest',
    'rant',
    'workmoms',
    'workingmoms',
    'neighborsfromhell',

    # ── Meme stocks / crypto (not ag-related) ──────────────────────────
    'superstonk',
    'wallstreetbets',
    'wallstreetbetsnew',
    'wallstreetbetselite',
    'cryptocurrency',
    'cryptomoonshots',
    'cryptocurrencyclassic',
    'cryptospread',
    'cryptoindex_io',
    'ethtrader',
    'ethereum',
    'bitcoin',
    'bitcoincashsv',
    'moonbets',
    'defi',
    'pennystocks',
    'pennystock',
    'pennystockwatch',
    'byndInvest',
    'weedstocks',
    'swingtrading',
    'options',
    'nsdq420',
    'mmat',
    'mvis',
    'rklb',
    'spacs',
    'spacstocks',
    'bttensor_',
    'safemoon',
    'loopringorg',
    'pinetwork',
    'banano',
    'cardano',
    'singularitynetwork',
    'chainlink',
    'linktrader',
    'origintrail',
    'heliumnetwork',
    'haratoken',
    'altcoinadvisor',
    'monsterhunterworld',
    'byfin',
    'bynd',
    'byndcentral',
    'nba',
    'cfb',
    'premierleague',
    'formula1',
    'motogp',
    'formuladank',
    'f1technical',
    'SquaredCircle',
    'squaredcircle',
    'tennis',
    'fantasyfootball',
    'worldofwarships',
    'mtb',

    # ── Pure job-board spam ─────────────────────────────────────────────
    'jobboardsearch',
    'onlinemarketresearch',
    'jobbit',
    'jobs4bitcoins',
    'recruitinghell',
    # ── Academic homework / exam help spam ──────────────────────────────
    'mathshelper',
    'statisticshelperz',
    'studentcorner',
    'eductionalpartner',
    'gm310509',
    'college_homework',
    'samplesize',
    'surveyCircle',
    'surveycircle',

    # ── Conspiracy / hard political ─────────────────────────────────────
    'conspiracy',
    'ufos',
    'ufo',
    'ufob',
    'ufos_archives',
    'highstrangeness',
    'aliens',
    'aliensdarkdescent',
    'alientechnology',
    'paralanormal',
    'paranormal',
    'interlimensionalNHI',
    'interdimensionalnhi',
    'mystery',
    'unresolvedmysteries',
    'lowstakesconspiracies',
    'observingtheanomaly',
    'politics',
    'politicalcompassmemes',
    'politicalreceipts',
    'political_revolution',
    'conservativeterrorism',
    'whattrumphasdone',
    'trumpvirus',
    'walkaway',
    'democrats',
    'capitalconsequences',
    'capitolconsequences',
    'welometogilead',
    'welcometogilead',
    'islampalestine',
    'israelpalestine',
    'ukrainerussiareportii',
    'endukrainiangenocide',
    'debatevaccines',
    'hermancainaward',
    'hermancaindebate',
    'parlerwatch',
    'againsthatesubreddits',
    'insanepeoplefacebook',
    'inceltears',
    'kotakuinaction',
    'boomersbeingfools',
    'debatereligion',
    'debateachristian',
    'exmormon',
    'exmuslim',
    'antinatalism',
    'keep_track',
    'fednews',
    'modfed',
    'modcoord',
    'fedjerk',
    'aboring',
    'aboling',
    'aboRingDystopia',
    'aboringdystopia',

    # ── Horror / creepypasta ────────────────────────────────────────────
    'scarystories',
    'creepcast',
    'creepcast_submissions',
    'creepsmcpasta',
    'talesFromtheCreeps',
    'talesfromthecreeps',
    'writersofhorror',
    'copypasta',

    # ── Drug game specifically ──────────────────────────────────────────
    'schedule_i',

    # ── Pure entertainment / social noise ──────────────────────────────
    'funny',
    'memes',
    'feedthememes',
    'oddlysatisfying',
    'oddlyspecific',
    'mildlyinteresting',
    'nextfuckinglevel',
    'therewasanattempt',
    'dadjokes',
    'mademeSmile',
    'madesemile',
    'wellthatsucks',
    'nextlevel',
    'pics',
    'videos',
    'aww',
    'tiktokcringe',
    'oddlysatisfying',
    'showerthoughts',
    'letsDisucssthis',
    'letsdiscussthis',
    'brightside',
    'damnthatsinteresting',   # (keep only if already in keep; scraper likely found legit ag posts here)
    'coolguides',
    'infographics',
    'charts',
    'dataisbeautiful',        # kept above but flagging it borderline
    'unnecessaryinventions',
    'specializedtools',

    # ── Anime / manga / fandom ─────────────────────────────────────────
    'hololive',
    'rwby',
    'jujutsufolk',
    'onpiece',
    'onepier',
    'onepiec',
    'manhua',
    'romancebooks',
    'romancenovels',
    'characterrant',
    'dccomics',
    'marvelstudios',
    'respectedthreads',
    'respectthreads',
}

# -----------------------------------------------------------------------
# A few subreddits that appear borderline but should be KEPT because
# their posts were found by agriculture/AI keyword search and are on-topic.
# (Listed for documentation; the logic just checks not-in-exclude.)
# -----------------------------------------------------------------------
# r/datascience, r/science, r/ChatGPT, r/AgriTech, r/H5N1_AvianFlu,
# r/dairyfarming, r/CellularAgriculture, r/drones, r/computervision,
# r/arduino, r/RFIDNews, r/singularity, r/tomorrowsworld, r/solarpunk,
# r/vegan, r/IAmA, r/changemyview, r/3Dprinting, r/Scholar ...


def main():
    print("=" * 70)
    print("SUBREDDIT-BASED DATA CLEANING  (blocklist strategy)")
    print("=" * 70)

    data = json.load(open('classified_sentiment_data.json'))
    print(f"Total posts loaded: {len(data)}")

    kept = []
    removed = []
    for item in data:
        sub_lower = item.get('subreddit', '').lower()
        if sub_lower in EXCLUDE_SUBREDDITS_LOWER:
            removed.append(item)
        else:
            kept.append(item)

    print(f"\nKept  (clean posts): {len(kept)}")
    print(f"Removed (confirmed off-topic): {len(removed)} ({len(removed)/len(data)*100:.1f}%)")

    print("\n--- Top kept subreddits ---")
    for sub, count in sorted(Counter(i.get('subreddit') for i in kept).items(),
                              key=lambda x: -x[1])[:30]:
        print(f"  r/{sub}: {count}")

    print("\n--- Top removed subreddits ---")
    for sub, count in sorted(Counter(i.get('subreddit') for i in removed).items(),
                              key=lambda x: -x[1])[:30]:
        print(f"  r/{sub}: {count}")

    with open('classified_sentiment_data_clean.json', 'w') as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {len(kept)} clean posts → classified_sentiment_data_clean.json")


if __name__ == '__main__':
    main()

