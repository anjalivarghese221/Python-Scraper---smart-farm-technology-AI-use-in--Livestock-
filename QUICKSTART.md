#  Quick Start Guide

## Running the Scraper

### Full Analysis (Recommended)
```bash
python3 main.py
```
This will:
1. Scrape Reddit for smart farming & AI livestock discussions
2. Apply the 9-step cleaning process
3. Display statistics and analysis
4. Generate 4 output files

### Run Individual Components

**Test scraping only:**
```bash
python3 scraper.py
```

**Test text cleaning:**
```bash
python3 text_cleaner.py
```

**Test display functions:**
```bash
python3 data_display.py
```

## Output Files

After running `main.py`, you'll get:

1. **raw_scraped_data.json** (220 KB)
   - Original scraped data from Reddit
   - Includes titles, text, scores, comments
   - 141 items collected

2. **cleaned_data.json** (805 KB)
   - Processed through 9-step cleaning
   - Includes tokens, bigrams, cleaned text
   - 116 unique items (duplicates removed)

3. **statistics.json** (2.2 KB)
   - Word frequency counts
   - Bigram statistics
   - Top 30 words and top 20 bigrams

4. **summary_report.txt** (2.2 KB)
   - Human-readable summary
   - Key statistics and trends
   - Top words and bigrams

## Understanding the Results

### Top Words Found
- **farm** (191 occurrences)
- **agriculture** (116 occurrences)
- **technology** (79 occurrences)
- **precision** (78 occurrences)
- **automate** (65 occurrences)
- **ai** (37 occurrences)

### Top Bigrams (Word Pairs)
- **precision → agriculture** (51 times)
- **agriculture → technology** (10 times)
- **smart → agriculture** (8 times)
- **labor → shortages** (8 times)

## Customization Tips

### 1. Change Search Terms
Edit [main.py](main.py) line 25:
```python
scraper.scrape_reddit_search("your search term here", limit=20)
```

### 2. Target Different Subreddits
Edit [scraper.py](scraper.py) line 18:
```python
self.subreddits = ['farming', 'agriculture', 'YourSubreddit']
```

### 3. Adjust Time Filter
Edit [main.py](main.py) line 36:
```python
scraper.scrape_top_posts_from_subreddit(sub, limit=25, time_filter='month')
# Options: 'hour', 'day', 'week', 'month', 'year', 'all'
```

### 4. Add Custom Stop Words
Edit [text_cleaner.py](text_cleaner.py) line 17 - add to the `stop_words` set:
```python
self.stop_words = set([
    'i', 'me', 'my', 'your_custom_word', ...
])
```

### 5. Add Lemmatization Rules
Edit [text_cleaner.py](text_cleaner.py) line 39 - add to `lemma_dict`:
```python
self.lemma_dict = {
    'farming': 'farm',
    'your_word': 'base_form',
    ...
}
```

## Common Issues & Solutions

### Issue: "No data was scraped"
**Causes:**
- Reddit might be blocking requests
- Network issues
- No relevant content found

**Solutions:**
- Wait a few minutes and try again
- Check internet connection
- Try different search terms
- Verify Reddit is accessible

### Issue: Rate limiting (HTTP 429)
**Solution:** The scraper automatically handles this with delays. Just wait.

### Issue: SSL Warning
The warning about OpenSSL/LibreSSL is harmless and doesn't affect functionality.

## Data Analysis Ideas

### Word Frequency Analysis
- Shows what topics are most discussed
- "precision", "automate", "technology" indicate tech focus
- "farm", "agriculture" confirm relevance

### Bigram Analysis
- Reveals common phrases and concepts
- "precision agriculture" is the most common (51 times)
- Shows relationships between concepts

### Further Analysis You Can Do:
1. **Sentiment Analysis** - Are discussions positive or negative?
2. **Trend Analysis** - Track topics over time
3. **Topic Modeling** - Group similar discussions
4. **Network Analysis** - See how concepts connect

## Performance Notes

**Scraping Time:** 2-5 minutes (depending on network and Reddit load)
**Cleaning Time:** A few seconds
**Total Data:** ~141 posts collected, 116 after deduplication

## Next Steps

1. Review [summary_report.txt](summary_report.txt) for insights
2. Explore [cleaned_data.json](cleaned_data.json) for detailed analysis
3. Use the cleaned data for:
   - Sentiment analysis
   - Machine learning models
   - Research papers
   - Trend identification

## Files Overview

```
├── main.py              # Main script - run this!
├── scraper.py           # Reddit scraping logic
├── text_cleaner.py      # 9-step cleaning process
├── data_display.py      # Visualization & reporting
├── requirements.txt     # Dependencies (beautifulsoup4, requests)
├── README.md           # Full documentation
└── QUICKSTART.md       # This file!
```

## Need Help?

Check the full [README.md](README.md) for:
- Detailed feature explanations
- Architecture details
- Advanced customization
- Ethical scraping guidelines
