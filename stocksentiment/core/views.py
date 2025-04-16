from django.shortcuts import render,HttpResponse
from .scripts import fetch_analyze as fa
from .models import CompanySentiment, StockPrediction
from django.http import JsonResponse
from django.utils.timezone import now
from datetime import datetime, timedelta
import joblib
import numpy as np
import os
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
from django.utils.timezone import now
from django.db.models import F
import yfinance as yf


company_tickers = [
    'META',         # Meta
    'TSLA',         # Tesla
    'MSFT',         # Microsoft
    # 'GOOGL',        # Google
    # 'AAPL',         # Apple
    'TCS.NS',       # Tata Consultancy Services
    'INFY.NS',      # Infosys
    'HDFCBANK.NS',  # HDFC Bank
    'RELIANCE.NS',  # Reliance Industries
    'WIPRO.NS',     # Wipro
    'ITCLTD.NS',    # ITC
    'HINDUNILVR.NS' # Hindustan Unilever
]

def sentiment_analysis_manual(request):
    results = []
    errors = []

    for company in company_tickers:
        try:
            # Fetch Reddit and News sentiments
            reddit_posts = fa.fetch_reddit_posts(company)
            news_articles = fa.fetch_clean_google_news(company)
            stock_data = fa.get_stock_data_on_date(company, datetime.now().strftime("%Y-%m-%d"))

            # Calculate average scores
            reddit_scores = [post['sentiment_score'] for post in reddit_posts]
            news_scores = [article['sentiment_score'] for article in news_articles]

            reddit_avg = sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0
            news_avg = sum(news_scores) / len(news_scores) if news_scores else 0

            total_avg = (
                (reddit_avg + news_avg) / 2 if reddit_scores and news_scores else
                reddit_avg or news_avg
            )

            sentiment_category = (
                "positive" if total_avg >= 0.05 else
                "negative" if total_avg <= -0.05 else
                "neutral"
            )

            # Save result to DB
            CompanySentiment.objects.create(
                company_name=company,
                reddit_score=reddit_avg,
                news_score=news_avg,
                sentiment_score=total_avg,
                sentiment_category=sentiment_category,
                stock_data=stock_data,
                timestamp=now()
            )

            results.append({
                'company': company,
                'reddit_average': round(reddit_avg, 4),
                'news_average': round(news_avg, 4),
                'total_average': round(total_avg, 4),
                'reddit_posts_count': len(reddit_posts),
                'news_articles_count': len(news_articles),
                'sentiment_category': sentiment_category,
                'stock_data': stock_data,
                'status': 'Saved to database'
            })

        except Exception as e:
            errors.append({
                'company': company,
                'error': str(e)
            })

    return JsonResponse({
        'results': results,
        'errors': errors
    }, status=200)


# from .task import handle_sleep


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stock_model.pkl')
model = joblib.load(MODEL_PATH)


def get_last_close_price(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # In case of weekends/holidays

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if not data.empty:
        # Get the last available close price
        last_close = data['Close'].iloc[-1]
        return float(last_close)
    else:
        return None

@csrf_exempt
def predict_all_stock_prices(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET method allowed'}, status=405)

    predictions = []
    errors = []

    for company_name in company_tickers:
        try:
            entries = CompanySentiment.objects.filter(company_name=company_name).order_by('-timestamp')[:7]

            if len(entries) < 7:
                # Insert with predicted price 0 if data is insufficient
                StockPrediction.objects.update_or_create(
                    company_name=company_name,
                    defaults={
                        'predicted_price': 0,
                        'prediction_time': datetime.now(),
                        'predicted_percentage_change': 0,
                        'direction': 'neutral',
                    }
                )
                errors.append({'company': company_name, 'error': 'Insufficient data'})
                continue

            # Prepare input data
            data = []
            for entry in entries:
                stock_data = entry.stock_data
                data.append({
                    'Open': stock_data.get('open'),
                    'High': stock_data.get('high'),
                    'Low': stock_data.get('low'),
                    'Volume': stock_data.get('volume'),
                    'SentimentScore': entry.sentiment_score,
                })

            df = pd.DataFrame(data)
            aggregated_features = df.mean().to_list()

            # Predict price
            prediction = model.predict([aggregated_features])[0]

            # Get last close price using yfinance
            last_close_price = get_last_close_price(company_name)

            if last_close_price:
                percentage_change = ((prediction - last_close_price) / last_close_price) * 100
                direction = 'up' if percentage_change > 0 else 'down' if percentage_change < 0 else 'neutral'
            else:
                percentage_change = 0
                direction = 'neutral'

            # Save to DB
            StockPrediction.objects.update_or_create(
                company_name=company_name,
                defaults={
                    'predicted_price': round(float(prediction), 2),
                    'prediction_time': datetime.now(),
                    'predicted_percentage_change': round(float(percentage_change), 2),
                    'direction': direction,
                }
            )

            # Append to response
            predictions.append({
                'company': company_name,
                'predicted_Close': round(float(prediction), 2),
                'predicted_percentage_change': round(float(percentage_change), 2),
                'direction': direction,
            })

        except Exception as e:
            errors.append({'company': company_name, 'error': str(e)})

    return JsonResponse({'predictions': predictions, 'errors': errors}, status=200)



@csrf_exempt
def get_predicted_stock_price(request, company_name):
    if request.method == 'GET':
        try:
            # Get the latest prediction for the company
            prediction = StockPrediction.objects.filter(company_name=company_name).order_by('-prediction_time').first()
            
            if prediction:
                return JsonResponse({
                    'company': company_name,
                    'predicted_Close': round(float(prediction.predicted_price), 2),
                    'prediction_time': prediction.prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_percentage_change': prediction.predicted_percentage_change,
                    'direction': prediction.direction,
                }, status=200)
            else:
                return JsonResponse({'error': 'Prediction not found for this company'}, status=404)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only GET method allowed'}, status=405)



