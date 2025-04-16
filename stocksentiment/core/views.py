from django.shortcuts import render,HttpResponse
from .scripts import fetch_analyze as fa
from .models import CompanySentiment, StockPrediction
from django.http import JsonResponse
from django.utils.timezone import now
from datetime import datetime
import joblib
import numpy as np
import os
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
from django.utils.timezone import now
from django.db.models import F


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



# def sentiment_analysis(request):
#     company = "TCS.NS"

#     # Fetch Reddit and News sentiments
#     reddit_posts = fa.fetch_reddit_posts(company)
#     news_articles = fa.fetch_clean_google_news(company)
#     stock_data = fa.get_stock_data_on_date(company, datetime.now().strftime("%Y-%m-%d"))

#     # Calculate average scores
#     reddit_scores = [post['sentiment_score'] for post in reddit_posts]
#     news_scores = [article['sentiment_score'] for article in news_articles]

#     reddit_avg = sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0
#     news_avg = sum(news_scores) / len(news_scores) if news_scores else 0

#     if reddit_scores and news_scores:
#         total_avg = (reddit_avg + news_avg) / 2
#     else:
#         total_avg = reddit_avg or news_avg  # Use whichever is available

#     # Save result to DB
#     CompanySentiment.objects.create(
#         company_name=company,
#         reddit_score=reddit_avg,
#         news_score=news_avg,
#         sentiment_score=total_avg,
#         sentiment_category=(
#             "positive" if total_avg >= 0.05 else
#             "negative" if total_avg <= -0.05 else
#             "neutral"
#         ),
#         stock_data = stock_data,
#         timestamp=now()
#     )

#     # Return JSON response
#     return JsonResponse({
#         'company': company,
#         'reddit_average': round(reddit_avg, 4),
#         'news_average': round(news_avg, 4),
#         'total_average': round(total_avg, 4),
#         'reddit_posts_count': len(reddit_posts),
#         'news_articles_count': len(news_articles),
#         'sentiment_category': (
#             "positive" if total_avg >= 0.05 else
#             "negative" if total_avg <= -0.05 else
#             "neutral"
#         ),
#         'stock_data' : stock_data,
#         'status': 'Saved to database'
#     })


# from .task import handle_sleep


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stock_model.pkl')
model = joblib.load(MODEL_PATH)


@csrf_exempt
def predict_stock_price(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            company_name = body.get('company_name')

            if not company_name:
                return JsonResponse({'error': 'Company name is required'}, status=400)

            # Fetch the latest 7 entries for the company
            entries = CompanySentiment.objects.filter(company_name=company_name).order_by('-timestamp')[:7]

            if len(entries) < 7:
                return JsonResponse({'error': 'Not enough data for the company'}, status=400)

            # Extract features
            data = []
            for entry in entries:
                stock_data = entry.stock_data
                data.append({
                    'Open': stock_data.get('open'),
                    'High': stock_data.get('high'),
                    'Low': stock_data.get('low'),
                    'Volume': stock_data.get('volume'),
                    'SentimentScore': entry.sentiment_score
                })

            df = pd.DataFrame(data)
            aggregated_features = df.mean().to_list()

            # Predict
            prediction = model.predict([aggregated_features])[0]

            obj, created = StockPrediction.objects.update_or_create(
                company_name=company_name,
                defaults={
                    'predicted_price': prediction,
                    'prediction_time': now()
                }
            )

            return JsonResponse({
                'company': company_name,
                'aggregated_input': dict(zip(df.columns, aggregated_features)),
                'predicted_Close': round(float(prediction), 2),
                'status': 'created' if created else 'updated'
            }, status=200)
        

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST method allowed'}, status=405)



