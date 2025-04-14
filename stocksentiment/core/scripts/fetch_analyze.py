import datetime
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
        title = entry.title.strip()
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

# Example usage
# if __name__ == "__main__":
#     company = "TCS"

#     reddit_posts = fetch_reddit_posts(company)
#     news_articles = fetch_clean_google_news(company)
#     reddit_scores = [post['sentiment_score'] for post in reddit_posts]
#     news_scores = [article['sentiment_score'] for article in news_articles]

#     reddit_avg = sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0
#     news_avg = sum(news_scores) / len(news_scores) if news_scores else 0

#     total_avg = (reddit_avg + news_avg) / 2 if reddit_scores and news_scores else 0
#     print(f"Average Sentiment Score for {company}: {total_avg:.2f}")


