from celery import shared_task, Task
from django.shortcuts import render,HttpResponse
from .scripts import fetch_analyze as fa
from .models import CompanySentiment
from django.http import JsonResponse
from django.utils.timezone import now
from datetime import datetime


company_tickers = [
    'META',        # Meta (formerly Facebook)
    'TSLA',        # Tesla
    'MSFT',        # Microsoft
    'GOOGL',       # Google (Alphabet Inc.)
    'AAPL',        # Apple
    'TCS.NS',      # Tata Consultancy Services
    'INFY.NS',     # Infosys
    'HDFCBANK.NS', # HDFC Bank
    'RELIANCE.NS', # Reliance Industries
    'WIPRO.NS',     # Wipro
    'ITCLTD.NS',  # ITC Limited
    'HINDUNILVR.NS', # Hindustan Unilever
]

@shared_task(bind=True, name="core.tasks.sentiment_analysis")
def sentiment_analysis(self, company: str) -> dict: 

    # Fetch Reddit and News sentiments
    reddit_posts = fa.fetch_reddit_posts(company)
    news_articles = fa.fetch_clean_google_news(company)
    stock_data = fa.get_stock_data_on_date(company, datetime.now().strftime("%Y-%m-%d"))

    # Calculate average scores
    reddit_scores = [post['sentiment_score'] for post in reddit_posts]
    news_scores = [article['sentiment_score'] for article in news_articles]

    reddit_avg = sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0
    news_avg = sum(news_scores) / len(news_scores) if news_scores else 0

    if reddit_scores and news_scores:
        total_avg = (reddit_avg + news_avg) / 2
    else:
        total_avg = reddit_avg or news_avg  # Use whichever is available

    # Save result to DB
    print("Creating CompanySentiment entry...")

    CompanySentiment.objects.create(
        company_name=company,
        reddit_score=reddit_avg,
        news_score=news_avg,
        sentiment_score=total_avg,
        sentiment_category=(
            "positive" if total_avg >= 0.05 else
            "negative" if total_avg <= -0.05 else
            "neutral"
        ),
        stock_data = stock_data,
        timestamp=now()
    )
    print("Created!")


    # Return JSON response
    print("Sentiment analysis completed and saved to database.")
    print(f"Company: {company}")
    return {
        'company': company,
        'reddit_average': round(reddit_avg, 4),
        'news_average': round(news_avg, 4),
        'total_average': round(total_avg, 4),
        'reddit_posts_count': len(reddit_posts),
        'news_articles_count': len(news_articles),
        'sentiment_category': (
            "positive" if total_avg >= 0.05 else
            "negative" if total_avg <= -0.05 else
            "neutral"
        ),
        'stock_data' : stock_data,
        'status': 'Saved to database'
    }



@shared_task(bind=True, name="core.tasks.company_wise_sentiment_analysis")
def company_wise_sentiment_analysis(self):

    results = {}
    for company in company_tickers:
        # result = sentiment_analysis(company)
        result = sentiment_analysis.delay(company) # type: ignore
        results[company] = result

    return results



