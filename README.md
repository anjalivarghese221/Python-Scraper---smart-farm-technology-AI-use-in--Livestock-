#  Smart Farm Technology & AI in Livestock - Social Media Scraper

A Python-based web scraper that collects and analyzes social media opinions about smart farming technology and AI use in livestock management. The scraper focuses on Reddit discussions and implements a comprehensive 9-step text cleaning process for data analysis.

##  Features

### 1. **Reddit Social Media Scraping**
- Scrapes relevant posts from farming, agriculture, and technology subreddits
- Searches for specific keywords: "smart farm", "AI livestock", "precision agriculture"
- Collects post titles, content, scores, and comment counts
- No authentication required (uses public Reddit JSON API)

### 2. **9-Step Text Cleaning Process**
Based on the industry-standard text preprocessing pipeline:
1.  Remove duplicate tweets/posts to filter out bots
2.  Remove usernames and links
3.  Remove special characters and punctuation
4.  Exclude meaningless words ("stop" words)
5.  Save text for sentiment analysis
6.  Remove hashtagged words
7.  Tokenize the texts (break into words)
8.  Count word combinations (bigrams)
9.  Convert tokenized words to base form (lemmatization)

### 3. **Data Visualization & Display**
- Color-coded terminal output with statistics
- Top word frequency analysis
- Bigram (word pair) frequency analysis
- Topic-specific keyword extraction
- Beautiful data presentation with progress indicators

### 4. **Output Files**
- `raw_scraped_data.json` - Original scraped data
- `cleaned_data.json` - Processed and cleaned text data
- `statistics.json` - Word frequencies and analysis
- `summary_report.txt` - Human-readable summary report

##  Getting Started

### Prerequisites
- Python 3.9+ (already installed )
- pip (Python package manager)

### Installation

1. **Install required packages:**
```bash
pip3 install -r requirements.txt
```

### Usage

**Run the complete scraper with cleaning and analysis:**
```bash
python3 main.py
```

**Or run individual components:**
```bash
# Just scraping
python3 scraper.py

# Test text cleaning
python3 text_cleaner.py

# Test display
python3 data_display.py
```

##  Project Structure

```
├── main.py              # Main orchestration script
├── scraper.py           # Reddit scraping functionality
├── text_cleaner.py      # 9-step cleaning process
├── data_display.py      # Visualization and reporting
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

##  Sample Output

The scraper will:
1. Search Reddit for relevant posts about smart farming and AI livestock
2. Clean and process the text data
3. Display statistics including:
   - Total words and unique vocabulary
   - Most frequent words related to the topic
   - Common word pairs (bigrams)
   - Topic-specific keyword analysis
4. Generate multiple output files for further analysis

##  Target Topics

The scraper focuses on:
- **Smart Farm Technology**: Sensors, automation, IoT devices
- **AI & Machine Learning**: Predictive analytics, computer vision
- **Livestock Management**: Cattle monitoring, dairy farming, health tracking
- **Precision Agriculture**: Data-driven farming decisions

##  Customization

### Modify Target Subreddits
Edit the `subreddits` list in [scraper.py](scraper.py):
```python
self.subreddits = ['farming', 'agriculture', 'AgTech', 'dairy', 'livestock']
```

### Adjust Search Keywords
Modify search terms in [main.py](main.py):
```python
scraper.scrape_reddit_search("your custom search term", limit=20)
```

### Change Stop Words
Edit the `stop_words` set in [text_cleaner.py](text_cleaner.py) to customize word filtering.

##  Important Notes

### Ethical Scraping
- The scraper uses public Reddit APIs and respects rate limiting
- Includes polite delays between requests (2 seconds)
- Only collects publicly available data
- Always respect website terms of service

### Potential Issues & Solutions

**Issue**: Reddit blocks requests
- **Solution**: The scraper includes proper User-Agent headers and rate limiting
- **Alternative**: If blocked, wait a few minutes before retrying

**Issue**: No data collected
- **Solution**: Check your internet connection
- **Alternative**: Reddit's JSON API might be temporarily unavailable

**Issue**: Rate limiting (HTTP 429)
- **Solution**: The scraper automatically waits when rate limited
- **Note**: This is normal and handled automatically

##  Understanding the Results

### Top Words
Shows the most frequently mentioned terms in discussions about smart farming and AI livestock.

### Bigrams (Word Pairs)
Reveals common phrases and concepts, such as:
- "smart → farm"
- "cattle → health"
- "ai → system"

### Topic Keywords
Categorized analysis showing:
- **Technology**: AI, sensors, automation
- **Livestock**: Cattle, dairy, animals
- **Farming**: Agriculture, crops, fields
- **Management**: Monitoring, tracking, health

##  Future Enhancements

Potential additions:
- [ ] Sentiment analysis on cleaned text
- [ ] Export to CSV format
- [ ] Twitter/X scraping (requires API keys)
- [ ] Word cloud visualization
- [ ] Time-series analysis of discussions
- [ ] Machine learning topic modeling

##  License

This project is for educational and research purposes.

##  Contributing

---

**Note**: This scraper was created for analyzing public discussions about agricultural technology. Always ensure you comply with website terms of service and data privacy regulations.
