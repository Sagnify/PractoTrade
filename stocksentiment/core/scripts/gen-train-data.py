import datetime
from datetime import datetime, timedelta
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import pandas as pd
from urllib.parse import quote_plus
import random
import time

analyzer = SentimentIntensityAnalyzer()

def fetch_reddit_posts(company_name, subreddit='stocks', limit=20, start_date=None, end_date=None):
    url = f'https://www.reddit.com/r/{subreddit}/search.json?q={company_name}&restrict_sr=1&sort=new&limit={limit}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        posts = []
        for post in data['data']['children']:
            post_date = datetime.fromtimestamp(post['data']['created_utc'], tz=datetime.timezone.utc).replace(tzinfo=None)
            if start_date and end_date and not (start_date <= post_date <= end_date):
                continue  # Filter posts by date range
            text = f"{post['data']['title']} {post['data'].get('selftext', '')}"
            sentiment_score = analyzer.polarity_scores(text)['compound']
            print(f"Reddit Post: {text}, Sentiment: {sentiment_score}")  # Debugging
            posts.append(sentiment_score)
        return posts
    return []

def fetch_clean_google_news(company_name, limit=20, start_date=None, end_date=None):
    encoded_name = quote_plus(company_name)
    rss_url = f'https://news.google.com/rss/search?q={encoded_name}'
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:limit]:
        published_date = datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc).replace(tzinfo=None)  # Make naive
        if start_date and end_date and not (start_date <= published_date <= end_date):
            continue  # Filter articles by date range
        sentiment_score = analyzer.polarity_scores(entry.title.strip())['compound']
        print(f"Google News Article: {entry.title}, Sentiment: {sentiment_score}")  # Debugging
        articles.append(sentiment_score)
    return articles

def get_average_sentiment_for_date(company_name, date):
    start_date = date - timedelta(hours=12)  # Search for posts and news on the same day
    end_date = date + timedelta(hours=12)
    
    # Ensure both start_date and end_date are naive
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)
    
    # Get sentiment scores from Reddit and Google News
    reddit_scores = fetch_reddit_posts(company_name, start_date=start_date, end_date=end_date)
    news_scores = fetch_clean_google_news(company_name, start_date=start_date, end_date=end_date)
    
    all_scores = reddit_scores + news_scores
    if all_scores:
        return sum(all_scores) / len(all_scores)
    return 0.0  # Default if no sentiment data is available

def get_last_month_stock_data_with_sentiment(ticker, company_name):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
    df.reset_index(inplace=True)
    
    sentiment_scores = []
    for date in df['Date']:
        sentiment_score = get_average_sentiment_for_date(company_name, date)
        
        # Introduce some random variation to the sentiment to increase diversity
        sentiment_score += random.uniform(-0.05, 0.05)  # Slight variation
        
        sentiment_scores.append(sentiment_score)
    
    df['SentimentScore'] = sentiment_scores
    df['Company'] = company_name
    df['Ticker'] = ticker
    return df

tickers = {
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "RELIANCE.NS": "Reliance",
    "HDFCBANK.NS": "HDFC Bank",
    "ITC.NS": "ITC"
}

# Create a list to hold the dataframes
all_dataframes = []
for ticker, company in tickers.items():
    print(f"Fetching data for {company} ({ticker})...")
    df = get_last_month_stock_data_with_sentiment(ticker, company)
    all_dataframes.append(df)
    # Sleep to avoid hitting rate limits
    time.sleep(1)

# Concatenate all dataframes into one
combined_df = pd.concat(all_dataframes)
combined_df.to_csv("stock_sentiment_dataset.csv", index=False)
print("Saved dataset to stock_sentiment_dataset.csv")
