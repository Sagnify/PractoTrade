import datetime
from datetime import datetime, timedelta
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

analyzer = SentimentIntensityAnalyzer()

def fetch_reddit_posts(company_name, subreddit='stocks', limit=20):
    url = f'https://www.reddit.com/r/{subreddit}/search.json?q={company_name}&restrict_sr=1&sort=new&limit={limit}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()

        posts = []
        for post in data['data']['children']:
            title = post['data']['title']
            content = post['data'].get('selftext', '')
            full_text = f"{title} {content}"

            sentiment = analyzer.polarity_scores(full_text)
            compound = sentiment['compound']
            category = (
                "positive" if compound >= 0.05 else
                "negative" if compound <= -0.05 else
                "neutral"
            )

            posts.append({
                'title': title,
                'content': content,
                'sentiment_score': compound,
                'sentiment_category': category
            })
        
        return posts
    else:
        print(f"Error fetching posts: {response.status_code}")
        return []

def fetch_clean_google_news(company_name, limit=20):
    rss_url = f'https://news.google.com/rss/search?q={company_name}'
    feed = feedparser.parse(rss_url)
    articles = []

    for entry in feed.entries[:limit]:
        title = str(entry.title).strip()
        sentiment = analyzer.polarity_scores(title)
        compound = sentiment['compound']
        category = (
            "positive" if compound >= 0.05 else
            "negative" if compound <= -0.05 else
            "neutral"
        )

        articles.append({
            'title': title,
            'published': entry.published,
            'sentiment_score': compound,
            'sentiment_category': category
        })

    return articles

def get_stock_data_on_date(ticker_symbol, date_str):
    """
    Fetches stock data for a specific ticker on a particular date.
    
    Args:
        ticker_symbol (str): e.g., "TCS.NS"
        date_str (str): Date in "YYYY-MM-DD" format

    Returns:
        dict: Stock data including open, high, low, close, volume
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        next_day = date + timedelta(days=1)
        
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(start=date_str, end=next_day.isoformat())

        if hist.empty:
            return {"error": f"No data found for {ticker_symbol} on {date_str}"}

        row = hist.iloc[0]
        return {
            "date": date_str,
            "ticker": ticker_symbol,
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "adj_close": round(row["Close"], 2),  # or row["Adj Close"] if needed
            "volume": int(row["Volume"])
        }

    except Exception as e:
        return {"error": str(e)}

# Example usage
# if __name__ == "__main__":
#     data = get_stock_data_on_date("TCS.NS", "2024-04-12")
#     print(data)

#     company = "TCS"

#     reddit_posts = fetch_reddit_posts(company)
#     news_articles = fetch_clean_google_news(company)
#     reddit_scores = [post['sentiment_score'] for post in reddit_posts]
#     news_scores = [article['sentiment_score'] for article in news_articles]

#     reddit_avg = sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0
#     news_avg = sum(news_scores) / len(news_scores) if news_scores else 0

#     total_avg = (reddit_avg + news_avg) / 2 if reddit_scores and news_scores else 0
#     print(f"Average Sentiment Score for {company}: {total_avg:.2f}")


