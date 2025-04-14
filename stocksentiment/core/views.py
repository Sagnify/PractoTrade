from django.shortcuts import render
from .scripts import fetch_analyze as fa
from .models import CompanySentiment
from django.http import JsonResponse
from django.utils.timezone import now


def sentiment_analysis(request):
    company = "TCS"

    # Fetch Reddit and News sentiments
    reddit_posts = fa.fetch_reddit_posts(company)
    news_articles = fa.fetch_clean_google_news(company)

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
        timestamp=now()
    )

    # Return JSON response
    return JsonResponse({
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
        'status': 'Saved to database'
    })
