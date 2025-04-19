import logging
import random
import time
from django.shortcuts import render,HttpResponse
import feedparser
import requests
from .scripts import fetch_analyze as fa
from .models import CompanySentiment, StockPrediction, StockPrediction, DailyPoll, PollOption, Vote, favourite
from django.http import JsonResponse
from django.utils.timezone import now
from datetime import datetime, timedelta
from django.utils import timezone
import joblib
import numpy as np
import pandas as pd
import os
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
from django.utils.timezone import now
from django.db.models import F
import yfinance as yf
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
import traceback
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_GET
from django.http import StreamingHttpResponse
from statsmodels.tsa.arima.model import ARIMA
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from stocksentiment.settings import CACHE_TIMEOUT_MEDIUM, CACHE_TIMEOUT_LONG, CACHE_TIMEOUT_SHORT
import uuid
import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth import login
from .models import Viewer
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import json
import traceback
from plotly.utils import PlotlyJSONEncoder
from django.http import JsonResponse
from django.shortcuts import render

company_tickers = [
    'META',         # Meta
    'TSLA',         # Tesla
    'MSFT',         # Microsoft

    'TCS.NS',       # Tata Consultancy Services
    'INFY.NS',      # Infosys
    'HDFCBANK.NS',  # HDFC Bank
    'RELIANCE.NS',  # Reliance Industries
    'WIPRO.NS',     # Wipro
    # 'ITCLTD.NS',    # ITC
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


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH_1 = os.path.abspath(os.path.join(BASE_DIR, 'stock_model.pkl'))
MODEL_PATH_2 = os.path.abspath(os.path.join(BASE_DIR, 'stock_model_v2.pkl'))
MODEL_PATH_3 = os.path.abspath(os.path.join(BASE_DIR, 'arima_model_combined.pkl'))

# Lazy loader for model 1
def get_model_1():
    if not hasattr(get_model_1, "_model"):
        print("Loading model 1...")
        get_model_1._model = joblib.load(MODEL_PATH_1)
    return get_model_1._model

# Lazy loader for model 2
def get_model_2():
    if not hasattr(get_model_2, "_model"):
        print("Loading model 2...")
        get_model_2._model = joblib.load(MODEL_PATH_2)
    return get_model_2._model

# Lazy loader for model 3 (ARIMA model)
def get_model_3():
    if not hasattr(get_model_3, "_model"):
        print("Loading ARIMA model...")
        get_model_3._model = joblib.load(MODEL_PATH_3)
    return get_model_3._model

# Usage example
# model = get_model_1()
# model_2 = get_model_2()

@csrf_exempt
def predict_all_stock_prices(request):  # sourcery skip: low-code-quality
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET method allowed'}, status=405)

    predictions = []
    errors = []

    for company_name in company_tickers:
        try:
            # Get more data points for ARIMA - fetch more historical data
            entries = CompanySentiment.objects.filter(company_name=company_name).order_by('-timestamp')[:14]  # Increased from 7 to 14

            if len(entries) < 7:
                StockPrediction.objects.update_or_create(
                    company_name=company_name,
                    defaults={
                        'predicted_price_with_sentiment': 0,
                        'predicted_price_without_sentiment': 0,
                        'avg_predicted_price': 0,
                        'prediction_time': timezone.now(),
                        'predicted_percentage_change': 0,
                        'direction': 'neutral',
                        'predicted_price_with_arima': 0,
                    }
                )
                errors.append({'company': company_name, 'error': 'Insufficient data'})
                continue

            # Prepare data for RandomForest models
            data = []
            data_no_sentiment = []

            # Extract closing prices and timestamps for ARIMA model
            closing_prices = []
            timestamps = []

            # Get the last close price directly from the first entry
            last_close_price = None
            if entries and entries[0].stock_data and 'close' in entries[0].stock_data:
                last_close_price = float(entries[0].stock_data['close'])
                print(f"Got last close price for {company_name} directly: {last_close_price}")

            for entry in entries:
                stock_data = entry.stock_data

                # Ensure we have valid closing price before adding to data
                close_price = stock_data.get('close')
                if close_price is not None:
                    # Convert to float to ensure we can do calculations
                    close_price = float(close_price)

                    # For RandomForest models
                    data.append([
                        stock_data.get('open'),
                        stock_data.get('high'),
                        stock_data.get('low'),
                        stock_data.get('volume'),
                        entry.sentiment_score
                    ])

                    data_no_sentiment.append([
                        stock_data.get('open'),
                        stock_data.get('high'),
                        stock_data.get('low'),
                    ])

                    # For ARIMA model
                    closing_prices.append(close_price)
                    timestamps.append(entry.timestamp)

            # Process RandomForest model inputs
            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'volume', 'sentiment_score'])
            df_no_sentiment = pd.DataFrame(data_no_sentiment, columns=['open', 'high', 'low'])

            # Debug logs for prediction
            print(f"Predicting for {company_name} with data: {df.head()}")

            try:
                # Load models for other predictions
                model_with_sentiment = get_model_1()
                model_without_sentiment = get_model_2()
                # Make predictions with sentiment and without sentiment
                features_with_sentiment = df.mean().values.tolist()
                features_without_sentiment = df_no_sentiment.mean().values.tolist()

                print(f"Features with sentiment: {features_with_sentiment}")
                print(f"Features without sentiment: {features_without_sentiment}")

                pred_with_sentiment = model_with_sentiment.predict([features_with_sentiment])[0]
                pred_without_sentiment = model_without_sentiment.predict([features_without_sentiment])[0]
            except Exception as e:
                print(f"Error predicting for {company_name} using RandomForest: {str(e)}")
                pred_with_sentiment = 0
                pred_without_sentiment = 0

            # Create and train a new ARIMA model for each company
            arima_pred = 0
            try:
                # Make sure we have at least 5 data points for ARIMA
                if len(closing_prices) >= 5:
                    # Convert to Series and ensure proper time order (oldest to newest)
                    ts = pd.Series(closing_prices[::-1], index=timestamps[::-1])  # Reverse order to have oldest first
                    ts = ts.sort_index()

                    print(f"ARIMA input for {company_name}: {len(ts)} data points")
                    print(f"Time series data: {ts}")

                    # Use a simpler ARIMA model if data is limited
                    order = (1, 1, 0) if len(ts) >= 7 else (1, 0, 0)

                    # Fit the ARIMA model
                    model = ARIMA(ts.values, order=order)
                    model_fit = model.fit()

                    # Forecast next value
                    forecast = model_fit.forecast(steps=1)
                    arima_pred = forecast[0]

                    print(f"ARIMA forecast for {company_name}: {arima_pred}")
                else:
                    print(f"Not enough closing prices for {company_name} ARIMA model: {len(closing_prices)} available")
                    # Use the last closing price as fallback
                    if closing_prices:
                        arima_pred = closing_prices[0]  # Use the most recent price
            except Exception as e:
                print(f"Error in ARIMA for {company_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Fallback to the last closing price if available
                if closing_prices:
                    arima_pred = closing_prices[0]

            # Calculate average prediction including ARIMA model
            # Only include ARIMA in average if it's not zero
            if arima_pred != 0:
                avg_pred = (pred_with_sentiment + pred_without_sentiment + arima_pred) / 3
            else:
                avg_pred = (pred_with_sentiment + pred_without_sentiment) / 2 if pred_with_sentiment or pred_without_sentiment else 0

            avg_pred = round(avg_pred, 2)

            # If we still don't have a last close price, try to get it from closing_prices
            if last_close_price is None and closing_prices:
                last_close_price = closing_prices[0]  # Use the most recent price
                print(f"Using most recent closing price for {company_name}: {last_close_price}")

            # If still no last_close_price, try the external function
            if last_close_price is None:
                last_close_price = get_last_close_price(company_name)
                print(f"Using external function for last close price for {company_name}: {last_close_price}")

            # Calculate percentage change
            percentage_change = 0
            direction = 'neutral'

            # Calculate only if we have valid data
            if last_close_price and last_close_price > 0 and avg_pred > 0:
                percentage_change = ((avg_pred - last_close_price) / last_close_price) * 100
                print(f"{company_name} - Last close: {last_close_price}, Avg pred: {avg_pred}, Change: {percentage_change}%")
                direction = 'up' if percentage_change > 0 else 'down' if percentage_change < 0 else 'neutral'
            elif arima_pred > 0 and closing_prices and closing_prices[0] > 0:
                percentage_change = ((avg_pred - closing_prices[0]) / closing_prices[0]) * 100
                print(f"{company_name} - Fallback: Using closing price: {closing_prices[0]}, Avg pred: {avg_pred}, Change: {percentage_change}%")
                direction = 'up' if percentage_change > 0 else 'down' if percentage_change < 0 else 'neutral'
            else:
                print(f"WARNING: Could not calculate percentage change for {company_name}")
                print(f"Last close price: {last_close_price}, closing_prices: {closing_prices[:1] if closing_prices else None}, Avg pred: {avg_pred}")

                # EMERGENCY FALLBACK: If we have a forecast and no last close price, use the arima prediction itself
                # This is not ideal but prevents having 0% changes
                if arima_pred > 0:
                    # Use the difference between the prediction and the arima value as a percentage change
                    percentage_change = ((avg_pred - arima_pred) / arima_pred) * 100
                    print(f"{company_name} - Emergency fallback: Using ARIMA pred: {arima_pred}, Avg pred: {avg_pred}, Change: {percentage_change}%")
                    direction = 'up' if percentage_change > 0 else 'down' if percentage_change < 0 else 'neutral'

            # Save to DB
            StockPrediction.objects.update_or_create(
                company_name=company_name,
                defaults={
                    'predicted_price_with_sentiment': round(float(pred_with_sentiment), 2),
                    'predicted_price_without_sentiment': round(float(pred_without_sentiment), 2),
                    'avg_predicted_price': round(float(avg_pred), 2),
                    'prediction_time': timezone.now(),
                    'predicted_percentage_change': round(float(percentage_change), 2),
                    'direction': direction,
                    'predicted_price_with_arima': round(float(arima_pred), 2),
                }
            )

            predictions.append({
                'company': company_name,
                'with_sentiment': round(float(pred_with_sentiment), 2),
                'without_sentiment': round(float(pred_without_sentiment), 2),
                'average': round(float(avg_pred), 2),
                'predicted_percentage_change': round(float(percentage_change), 2),
                'direction': direction,
                'arima_pred': round(float(arima_pred), 2),
            })

        except Exception as e:
            errors.append({'company': company_name, 'error': str(e)})
            import traceback
            traceback.print_exc()

    return JsonResponse({'predictions': predictions, 'errors': errors}, status=200)


# Make sure to fix the get_last_close_price function as well
def get_last_close_price(company_name):
    try:
        latest_entry = CompanySentiment.objects.filter(company_name=company_name).order_by('-timestamp').first()
        if latest_entry:
            stock_data = latest_entry.stock_data
            last_close = stock_data.get('close')
            # Fix for the float conversion warning
            return float(last_close) if isinstance(last_close, (int, float)) else None
        return None
    except Exception as e:
        print(f"Error getting last close price for {company_name}: {str(e)}")
        return None



# @csrf_exempt
# def get_predicted_stock_price(request, company_name):
#     if request.method != 'GET':
#         return

#     try:
#         # Get the latest prediction for the company
#         prediction = StockPrediction.objects.filter(company_name=company_name).order_by('-prediction_time').first()

#         if not prediction:
#             return JsonResponse({'error': 'Prediction not found for this company'}, status=404)

#         # Build the base URL
#         base_url = request.build_absolute_uri('/')[:-1]  # removes trailing slash

#         # API links for different periods
#         candle_urls = {
#             'realtime': f"{base_url}/api/stock-chart/?company={company_name}&interval=realtime",
#             '1d': f"{base_url}/api/stock-chart/?company={company_name}&interval=1d",
#             '7d': f"{base_url}/api/stock-chart/?company={company_name}&interval=7d",
#         }

#         # Return the predicted information along with the API links
#         return JsonResponse({
#             'company': company_name,
#             'predicted_with_sentiment': round(float(prediction.predicted_price_with_sentiment), 2),
#             'predicted_without_sentiment': round(float(prediction.predicted_price_without_sentiment), 2),
#             'avg_predicted_price': round(float(prediction.avg_predicted_price), 2),
#             'prediction_time': prediction.prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
#             'predicted_percentage_change': round(float(prediction.predicted_percentage_change), 2),
#             'direction': prediction.direction,
#             'api_links': candle_urls,
#         }
#         , status=200)

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)



@csrf_exempt
@cache_page(CACHE_TIMEOUT_MEDIUM)  # Cache for 1 hour
def get_predicted_stock_price(request, company_name):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        # Try to fetch prediction from DB
        prediction = StockPrediction.objects.filter(company_name=company_name).order_by('-prediction_time').first()

        # Build the base URL for API links
        base_url = request.build_absolute_uri('/')[:-1]  # removes trailing slash
        candle_urls = {
            'realtime': f"{base_url}/api/stock-chart/?company={company_name}&interval=realtime",
            '1d': f"{base_url}/api/stock-chart/?company={company_name}&interval=1d",
            '7d': f"{base_url}/api/stock-chart/?company={company_name}&interval=7d",
        }

        # Dummy companies list with necessary info only
        dummy_companies = {
            'AMZN': {'ticker': 'AMZN', 'name': 'Amazon', 'is_in': False},
            'GOOGL': {'ticker': 'GOOGL', 'name': 'Alphabet', 'is_in': False},
            'NVDA': {'ticker': 'NVDA', 'name': 'NVIDIA', 'is_in': False},
            'ITC': {'ticker': 'ITC.NS', 'name': 'ITC', 'is_in': True},
            'LT': {'ticker': 'LT.NS', 'name': 'Larsen & Toubro', 'is_in': True},
            'BAJFINANCE': {'ticker': 'BAJFINANCE.NS', 'name': 'Bajaj Finance', 'is_in': True},
        }


        # If prediction not found, but is a dummy company, return dummy data
        if not prediction and company_name in dummy_companies:
            dummy_price_1 = round(random.uniform(1000, 5000), 2)
            dummy_price_2 = round(dummy_price_1 + random.uniform(-50, 50), 2)
            avg_price = round((dummy_price_1 + dummy_price_2) / 2, 2)
            percentage_change = round((avg_price - dummy_price_1) / dummy_price_1 * 100, 2)
            direction = 'up' if percentage_change > 0 else 'down'

            return JsonResponse({
                'company': company_name,
                'isIn': dummy_companies[company_name]['is_in'],
                'predicted_with_sentiment': dummy_price_1,
                'predicted_without_sentiment': dummy_price_2,
                'avg_predicted_price': avg_price,
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_percentage_change': percentage_change,
                'direction': direction,
                'api_links': candle_urls,
            }, status=200)

        elif not prediction:
            return JsonResponse({'error': 'Prediction not found for this company'}, status=404)

        # If real prediction exists
        return JsonResponse({
            'company': company_name,
            'isIn': prediction.is_in,
            'predicted_with_sentiment': round(float(prediction.predicted_price_with_sentiment), 2),
            'predicted_without_sentiment': round(float(prediction.predicted_price_without_sentiment), 2),
            'arima_pred': round(float(prediction.predicted_price_with_arima), 2),
            'avg_predicted_price': round(float(prediction.avg_predicted_price), 2),
            'prediction_time': prediction.prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_percentage_change': round(float(prediction.predicted_percentage_change), 2),
            'direction': prediction.direction,
            'api_links': candle_urls,
        }, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def stock_chart_api(request):  # sourcery skip: low-code-quality
    """
    API endpoint to generate stock candlestick charts.
    Query parameters:
    - company: Stock ticker symbol (e.g., 'TCS.NS', 'AAPL')
    - interval: Time interval ('realtime', '1d', '7d')
    
    Returns a JSON response with the plotly chart data.
    """

    
    # Get query parameters
    company = request.GET.get('company', 'TCS.NS')
    interval = request.GET.get('interval', 'realtime')
    
    print(f"Received request for {company} with interval {interval}")
    
    # Set yfinance parameters based on the requested interval
    if interval == 'realtime':
        period = '1d'
        yf_interval = '1m'
    elif interval == '1d':
        period = '1d'
        yf_interval = '5m'
    elif interval == '7d':
        period = '7d'
        yf_interval = '1h'
    else:
        return JsonResponse({'error': 'Invalid interval. Choose from: realtime, 1d, 7d'}, status=400)
    
    try:
        print(f"Fetching data with period={period}, interval={yf_interval}")
        # Fetch stock data with explicit progress=False to avoid tqdm issues
        data = yf.download(company, period=period, interval=yf_interval, progress=False)
        
        if data is None:
            print(f"Failed to fetch data for {company}")
            return JsonResponse({'error': f'Failed to fetch data for {company}'}, status=404)
        if data.empty:
            print(f"No data found for {company}")
            return JsonResponse({'error': f'No data found for {company}'}, status=404)
        
        # Debug information
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns}")
        print(f"Data index type: {type(data.index)}")
        
        # Print the first few rows of data
        print("\nFirst 5 rows of data:")
        print(data.head())
        
        # Convert the index to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
            print("Converted index to datetime")

        
        # Fix: Convert Series to list instead of using tolist() on DataFrame
        # Format datetime for JSON serialization
        datetime_values = data.index.strftime('%Y-%m-%d %H:%M:%S').tolist() # type: ignore
        
        # Check if columns are MultiIndex (happens with some yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            print("Detected MultiIndex columns")
            # Print the exact structure of columns
            print("Exact column structure:", data.columns.tolist())
            
            # Try to find the appropriate columns - usually ticker symbol is second level
            open_cols = [col for col in data.columns if col[0] == 'Open']
            high_cols = [col for col in data.columns if col[0] == 'High']
            low_cols = [col for col in data.columns if col[0] == 'Low']
            close_cols = [col for col in data.columns if col[0] == 'Close']
            
            print(f"Found columns - Open: {open_cols}, High: {high_cols}, Low: {low_cols}, Close: {close_cols}")
            
            if open_cols:
                open_values = data[open_cols[0]].values.tolist()
            else:
                open_values = data.iloc[:, 0].values.tolist()  # Fallback
                
            if high_cols:
                high_values = data[high_cols[0]].values.tolist()
            else:
                high_values = data.iloc[:, 1].values.tolist()  # Fallback
                
            if low_cols:
                low_values = data[low_cols[0]].values.tolist()
            else:
                low_values = data.iloc[:, 2].values.tolist()  # Fallback
                
            if close_cols:
                close_values = data[close_cols[0]].values.tolist()
            else:
                close_values = data.iloc[:, 3].values.tolist()  # Fallback
        else:
            print("Using standard columns")
            # Extract OHLC data directly from DataFrame as numpy arrays then convert to lists
            open_values = data['Open'].values.tolist()
            high_values = data['High'].values.tolist()
            low_values = data['Low'].values.tolist()
            close_values = data['Close'].values.tolist()
        
        print(f"Data points extracted: {len(datetime_values)}")
        
        # Debug the first few values to verify data
        print(f"First few datetime values: {datetime_values[:5]}")
        print(f"First few OHLC values: Open {open_values[:5]}, High {high_values[:5]}, Low {low_values[:5]}, Close {close_values[:5]}")
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=datetime_values,
            open=open_values,
            high=high_values,
            low=low_values,
            close=close_values,
            name=company
        )])

        fig.update_layout(
            title=f'{company} Candlestick Chart ({interval})',
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=500,
            width=900,
            margin=dict(l=50, r=50, t=50, b=50, pad=4)
        )
        
        # Convert the figure to JSON for the response
        chart_json = json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        
        # Final debug - check the size of the JSON data
        chart_data = json.loads(chart_json)
        print(f"JSON chart data length: {len(json.dumps(chart_data))}")
        print(f"Number of data points in chart: {len(chart_data.get('data', [{}])[0].get('x', []))}")
        
        return JsonResponse({
            'company': company,
            'interval': interval,
            'chart_data': chart_data,
            'data_points': len(datetime_values)
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'error': str(e),
            'details': traceback.format_exc()
        }, status=500)


def stock_chart_view(request):
    """
    View to render the stock chart page
    """
    return render(request, 'stock_chart.html') 


@csrf_exempt
def company_list(request):
    """
    View to return the list of companies with tickers and full names as JSON
    """
    if request.method == 'GET':
        try:
            # Get the viewer_id from the request (assuming it's passed as a query parameter)
            viewer_id = request.GET.get('viewer_id')
            if not viewer_id:
                return JsonResponse({'error': 'Viewer ID is required'}, status=400)

            # Fetch the viewer object
            viewer = Viewer.objects.get(viewer_id=viewer_id)

            # Fetch the user's favourite companies from Favourites model
            user_favourites = favourite.objects.filter(user=viewer).values_list('company_name', flat=True)

            companies = {
                'META': {'ticker': 'META', 'name': 'Meta', 'is_fav': False, 'is_in': False, 'description': 'Meta (formerly Facebook) is a global leader in social media and virtual reality.'},
                'TSLA': {'ticker': 'TSLA', 'name': 'Tesla', 'is_fav': False, 'is_in': False, 'description': 'Tesla is an electric vehicle and clean energy company, revolutionizing transportation.'},
                'MSFT': {'ticker': 'MSFT', 'name': 'Microsoft', 'is_fav': False, 'is_in': False, 'description': 'Microsoft is a global technology company known for software, hardware, and cloud services.'},
                'TCS': {'ticker': 'TCS.NS', 'name': 'Tata Consultancy Services', 'is_fav': False, 'is_in': True, 'description': 'TCS is a leading global IT services and consulting company from India.'},
                'INFY': {'ticker': 'INFY.NS', 'name': 'Infosys', 'is_fav': False, 'is_in': True, 'description': 'Infosys is an Indian multinational corporation that provides IT and consulting services.'},
                'HDFCBANK': {'ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank', 'is_fav': False, 'is_in': True, 'description': 'HDFC Bank is one of India’s largest private sector banks offering a wide range of financial services.'},
                'RELIANCE': {'ticker': 'RELIANCE.NS', 'name': 'Reliance Industries', 'is_fav': False, 'is_in': True, 'description': 'Reliance Industries is a conglomerate with businesses in petrochemicals, retail, and telecommunications.'},
                'WIPRO': {'ticker': 'WIPRO.NS', 'name': 'Wipro', 'is_fav': False, 'is_in': True, 'description': 'Wipro is an Indian multinational corporation providing IT services and consulting.'},
                'HINDUNILVR': {'ticker': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever', 'is_fav': False, 'is_in': True, 'description': 'Hindustan Unilever is a leading Indian consumer goods company offering products in health, beauty, and home care.'},

                # 🌍 Global Dummy Companies
                'AMZN': {'ticker': 'AMZN', 'name': 'Amazon', 'is_fav': False, 'is_in': False, 'description': 'Amazon is a multinational technology company focusing on e-commerce, cloud computing, and AI.'},
                'GOOGL': {'ticker': 'GOOGL', 'name': 'Alphabet', 'is_fav': False, 'is_in': False, 'description': 'Alphabet is the parent company of Google, focusing on internet services and products.'},
                'NVDA': {'ticker': 'NVDA', 'name': 'NVIDIA', 'is_fav': False, 'is_in': False, 'description': 'NVIDIA designs GPUs for gaming and professional markets, and is a key player in AI.'},

                # 🇮🇳 Indian Dummy Companies
                'ITC': {'ticker': 'ITC.NS', 'name': 'ITC', 'is_fav': False, 'is_in': True, 'description': 'ITC is an Indian conglomerate with businesses in FMCG, hotels, paperboards, and packaging.'},
                'LT': {'ticker': 'LT.NS', 'name': 'Larsen & Toubro', 'is_fav': False, 'is_in': True, 'description': 'L&T is a major Indian multinational in engineering, construction, and manufacturing.'},
                'BAJFINANCE': {'ticker': 'BAJFINANCE.NS', 'name': 'Bajaj Finance', 'is_fav': False, 'is_in': True, 'description': 'Bajaj Finance provides a range of financial services including loans, insurance, and investment products.'}
            }


            # Update 'is_fav' for each company based on user's favourites
            for company in companies.values():
                if company['ticker'] in user_favourites:
                    company['is_fav'] = True

            return JsonResponse({'companies': companies}, status=200)

        except Viewer.DoesNotExist:
            return JsonResponse({'error': 'Viewer not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)




def search(request):


    if request.method == 'GET':
        search_term = request.GET.get('search', '').strip()
        if not search_term:
            return JsonResponse({'error': 'Search term is required'}, status=400)

        companies = {
            'META': {'ticker': 'META', 'name': 'Meta', 'description': 'Meta (formerly Facebook) is a global leader in social media and virtual reality.', 'is_in': False},
            'TSLA': {'ticker': 'TSLA', 'name': 'Tesla', 'description': 'Tesla is an electric vehicle and clean energy company, revolutionizing transportation.', 'is_in': False},
            'MSFT': {'ticker': 'MSFT', 'name': 'Microsoft', 'description': 'Microsoft is a global technology company known for software, hardware, and cloud services.', 'is_in': False},
            'TCS': {'ticker': 'TCS.NS', 'name': 'Tata Consultancy Services', 'description': 'TCS is a leading global IT services and consulting company from India.', 'is_in': True},
            'INFY': {'ticker': 'INFY.NS', 'name': 'Infosys', 'description': 'Infosys is an Indian multinational corporation that provides IT and consulting services.', 'is_in': True},
            'HDFCBANK': {'ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank', 'description': 'HDFC Bank is one of India’s largest private sector banks offering a wide range of financial services.', 'is_in': True},
            'RELIANCE': {'ticker': 'RELIANCE.NS', 'name': 'Reliance Industries', 'description': 'Reliance Industries is a conglomerate with businesses in petrochemicals, retail, and telecommunications.', 'is_in': True},
            'WIPRO': {'ticker': 'WIPRO.NS', 'name': 'Wipro', 'description': 'Wipro is an Indian multinational corporation providing IT services and consulting.', 'is_in': True},
            'HINDUNILVR': {'ticker': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever', 'description': 'Hindustan Unilever is a leading Indian consumer goods company offering products in health, beauty, and home care.', 'is_in': True},
        }


        # Perform the search
        results = [
            {'company_name': key}
            for key, value in companies.items()
            if search_term.lower() in key.lower() or search_term.lower() in value['name'].lower()
        ]
        if not results:
            return JsonResponse({'message': 'No results found'}, status=404)

        return JsonResponse(list(results), safe=False, status=200)




@csrf_exempt
def reddit_post_fetcher_by_company(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    ticker = request.GET.get('ticker', '').upper()
    if not ticker:
        return JsonResponse({'error': 'Ticker is required'}, status=400)

    hardcoded_posts = {
        'META': [
                "https://reddit.com/r/stocks/comments/1k1eyjd/united_healthcare_currently_down_23_today_after/",
    "https://reddit.com/r/stocks/comments/1k0o2jh/why_is_meta_trading_down_so_hard/",
    "https://reddit.com/r/stocks/comments/1k0nih7/after_surge_to_record_highs_gold_overtakes/",
    "https://reddit.com/r/stocks/comments/1k0k38d/0416_interesting_stocks_today_he_who_controls_the/",
    "https://reddit.com/r/stocks/comments/1jzvy6f/netflix_stock_pops_after_report_streaming_giant/"
        ],
        'TSLA': [
                "https://reddit.com/r/stocks/comments/1k0nih7/after_surge_to_record_highs_gold_overtakes/",
    "https://reddit.com/r/stocks/comments/1jzhipc/how_bad_is_this_for_tsla/",
    "https://reddit.com/r/stocks/comments/1jxvwyf/teslas_stock_is_set_for_a_death_cross_on_monday/",
    "https://reddit.com/r/stocks/comments/1jxj6cv/easy_10x_lcid/",
    "https://reddit.com/r/stocks/comments/1jwwtut/tsla_bulls_what_makes_you_hopeful_about_the/"
        ],
        'MSFT': [
                "https://reddit.com/r/stocks/comments/1k1eclh/how_to_bet_on_goog_genai/",
    "https://reddit.com/r/stocks/comments/1k17f0f/my_30_year_global_macro_portfolio_designed_to/",
    "https://reddit.com/r/stocks/comments/1k0nih7/after_surge_to_record_highs_gold_overtakes/",
    "https://reddit.com/r/stocks/comments/1jzvy6f/netflix_stock_pops_after_report_streaming_giant/",
    "https://reddit.com/r/stocks/comments/1ju5fzj/are_you_trying_to_snipe_the_dips/"
        ],
        'TCS.NS': [
    "https://reddit.com/r/stocks/comments/18cv69n/could_hasbro_face_another_activist_shareholder/",
    "https://reddit.com/r/stocks/comments/13pllwu/523_tuesdays_premarket_stock_movers_news/",
    "https://reddit.com/r/stocks/comments/tyby4k/47_thursdays_premarket_stock_movers_news/",
    "https://reddit.com/r/stocks/comments/soe1vt/29_wednesdays_premarket_stock_movers_news/",
    "https://reddit.com/r/stocks/comments/oh0dst/i_went_over_the_entire_tsx_and_tsxv_heres_what_i/"
        ],
        'INFY.NS': [
                "https://reddit.com/r/stocks/comments/19dg6lh/how_ai_might_improve_it_service_margins_here_a/",
    "https://reddit.com/r/stocks/comments/12zegz8/426_wednesdays_premarket_stock_movers_news/",
    "https://reddit.com/r/stocks/comments/y3tcye/1014_fridays_premarket_stock_movers_news/",
    "https://reddit.com/r/stocks/comments/t86jco/investing_in_flawed_democracies/",
    "https://reddit.com/r/stocks/comments/nl7km5/help_with_dd_it_service_firms/"
        ],
        'HDFCBANK.NS': [
                "https://reddit.com/r/stocks/comments/8r29no/tomorrows_nse_support_resistance/",
    "https://reddit.com/r/stocks/comments/2baj5b/hdfc_bank_limited_hdfcbank_hdfc_limited_hdfc_post/",
    "https://reddit.com/r/StockMarket/comments/1jtj0i6/what_website_is_this/"
        ],
        'RELIANCE.NS': [
    "https://reddit.com/r/stocks/comments/1e9kp03/india_sends_100_antitrust_queries_for_reliance/",
    "https://reddit.com/r/stocks/comments/1b2avuq/disney_press_release_disney_india_reliance_merge/",
    "https://reddit.com/r/stocks/comments/1b26ym6/disney_and_reliance_to_merge_media_businesses_in/",
    "https://reddit.com/r/stocks/comments/1azuxn4/disney_reliance_in_india_sign_a_binding_agreement/",
    "https://reddit.com/r/stocks/comments/19cpcdr/sony_sends_termination_letter_to_zee_on_10_bln/"
        ],
        'WIPRO.NS': [
    "https://reddit.com/r/stocks/comments/t86jco/investing_in_flawed_democracies/",
    "https://reddit.com/r/stocks/comments/t27ova/changes_need_to_be_made_invest_in_democracies/",
    "https://reddit.com/r/stocks/comments/p470l8/ive_compiled_a_shortlist_of_companies_that_show/",
    "https://reddit.com/r/stocks/comments/nl7km5/help_with_dd_it_service_firms/",
    "https://reddit.com/r/stocks/comments/izuti9/what_is_wrong_with_this_company_wit/"
        ],
        'HINDUNILVR.NS': [
        "https://reddit.com/r/stocks/comments/q8nehu/just_a_trade_idea_with_fundamentals_rada_rada/",
    "https://reddit.com/r/stocks/comments/htfkds/what_do_you_think_about_indian_stocks/",
    "https://reddit.com/r/stocks/comments/9qf0pg/how_does_dividend_from_a_subsidiary_work/",
    "https://reddit.com/r/investing/comments/9cvx7w/understanding_fund_movements/",
    "https://reddit.com/r/investing/comments/190yy5/what_will_cancun_tell_us_about_zincs_year_ahead/"
        ],

    }

    if ticker not in hardcoded_posts:
        return JsonResponse({'error': 'Ticker not supported'}, status=400)

    return JsonResponse({'ticker': ticker, 'post_links': hardcoded_posts[ticker]})



COMPANY_LIST = [
    'META', 'TSLA', 'MSFT', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'WIPRO.NS',
    'HINDUNILVR.NS', 'AMZN', 'GOOGL', 'NVDA', 'ITC.NS', 'LT', 'BAJFINANCE'
]

@csrf_exempt
@require_http_methods(["GET", "POST"])
def company_poll_api(request, company_name):
    today = timezone.now().date()
    company_name_upper = company_name.upper()

    if company_name_upper not in COMPANY_LIST:
        return JsonResponse({'error': 'Company not found'}, status=404)

    poll = DailyPoll.objects.filter(company_name=company_name_upper, created_at=today).first()
    if not poll:
        return JsonResponse({'error': 'Poll not available for today'}, status=404)

    if request.method == 'GET':
        options = poll.options.all() # type: ignore
        total_votes = sum(opt.votes for opt in options)

        option_data = []
        leading_option = None
        max_votes = -1

        for opt in options:
            percent = round((opt.votes / total_votes) * 100, 2) if total_votes > 0 else 0.0
            option_data.append({
                'id': opt.id,
                'text': opt.option_text,
                'votes': opt.votes,
                'percentage': percent
            })

            if opt.votes > max_votes:
                leading_option = {
                    'text': opt.option_text,
                    'percentage': percent
                }
                max_votes = opt.votes

        return JsonResponse({
            'company': poll.company_name,
            'question': poll.question,
            'poll_id': poll.id, # type: ignore
            'total_votes': total_votes,
            'options': option_data,
            'leading_sentiment': leading_option or {}
        }, status=200)


    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            option_id = data.get('option_id')

            if not session_id or not option_id:
                return JsonResponse({'error': 'Missing session_id or option_id'}, status=400)

            if Vote.objects.filter(poll=poll, session_id=session_id).exists():
                return JsonResponse({'message': 'You have already voted today'}, status=403)

            option = PollOption.objects.get(id=option_id, poll=poll)
            option.votes += 1
            option.save()

            Vote.objects.create(poll=poll, option=option, session_id=session_id)
            return JsonResponse({'message': 'Vote submitted successfully'}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)



# COMPANY_LIST = [
#     'META', 'TSLA', 'MSFT', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'WIPRO.NS',
#     'HINDUNILVR.NS', 'AMZN', 'GOOGL', 'NVDA', 'ITC.NS', 'LT', 'BAJFINANCE'
# ]

# @csrf_exempt
# def all_company_news(request):
#     all_articles = []

#     for company in COMPANY_LIST:
#         rss_url = f"https://news.google.com/rss/search?q={company}"
#         feed = feedparser.parse(rss_url)
#         articles = [
#             {
#                 'company': company,
#                 'title': entry.title,
#                 'url': entry.link
#             }
#             for entry in feed.entries[:10]
#         ]
#         all_articles.extend(articles)

#     return JsonResponse({'news': all_articles}, safe=False)


# Configure logging
logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 5
MAX_RETRIES = 2

@csrf_exempt
@require_GET
def all_company_news(request):  # sourcery skip: low-code-quality
    """
    Fetch news articles for all companies from Google News RSS.
    Includes error handling, timeouts, and detailed logging.
    """
    cache_key = "all_company_news"
    cached_news = cache.get(cache_key)
    
    if cached_news:
        logger.info("Returning cached news results")
        return JsonResponse({'news': cached_news}, safe=False)
    
    logger.info("Cache miss for all_company_news - fetching fresh data")
    all_articles = []
    errors = []
    
    for company in COMPANY_LIST:
        try:
            logger.debug(f"Fetching news for company: {company}")
            start_time = time.time()
            
            # Construct URL with company name properly encoded
            from urllib.parse import quote_plus
            company_encoded = quote_plus(company)
            rss_url = f"https://news.google.com/rss/search?q={company_encoded}"
            
            # Set up feedparser with timeout using requests
            try:
                # Try with requests first for better timeout control
                response = requests.get(rss_url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                
                # Use the response content with feedparser
                feed = feedparser.parse(response.content)
            except (requests.RequestException, requests.Timeout) as req_error:
                logger.warning(f"Request error for {company}: {req_error}")
                # Fall back to direct feedparser (less timeout control)
                feed = feedparser.parse(rss_url)
            
            # Check if the feed has entries and wasn't an error
            if hasattr(feed, 'entries') and feed.entries and not feed.get('bozo_exception'):
                articles = []
                for entry in feed.entries[:10]:
                    try:
                        article = {
                            'company': company,
                            'title': entry.title,
                            'url': entry.link
                        }
                        articles.append(article)
                    except AttributeError as e:
                        logger.error(f"Error processing entry for {company}: {e}")
                        continue
                
                all_articles.extend(articles)
                elapsed = time.time() - start_time
                logger.debug(f"Processed {len(articles)} articles for {company} in {elapsed:.2f}s")
            else:
                error_msg = f"No valid entries found for {company}"
                if hasattr(feed, 'bozo_exception'):
                    error_msg += f": {feed.bozo_exception}"
                logger.warning(error_msg)
                errors.append(error_msg)
                
        except Exception as e:
            logger.exception(f"Unexpected error processing {company}: {str(e)}")
            errors.append(f"Error for {company}: {str(e)}")
            continue
    
    # If we got at least some articles, cache them
    if all_articles:
        logger.info(f"Caching {len(all_articles)} news articles for {CACHE_TIMEOUT_MEDIUM}s")
        cache.set(cache_key, all_articles, timeout=CACHE_TIMEOUT_MEDIUM)
        
        # If there were some errors but we have some results, log the errors
        if errors:
            logger.warning(f"Completed with {len(errors)} errors: {', '.join(errors)}")
        
        return JsonResponse({'news': all_articles}, safe=False)
    else:
        # If no articles were found at all, return an error
        error_message = "Failed to fetch news for any companies"
        logger.error(error_message)
        if errors:
            error_message += f": {', '.join(errors)}"
        return JsonResponse({'error': error_message}, status=500)




# --- SIGNUP VIEW ---
@csrf_exempt
def signup_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')

            if not (username and email and password):
                return JsonResponse({'error': 'Missing fields'}, status=400)

            if Viewer.objects.filter(email=email).exists():
                return JsonResponse({'error': 'Email already in use'}, status=400)

            # Create User for name only (as you intended)
            user = User.objects.create(username=username)

            # Hash password and create Trader
            hashed_password = make_password(password)
            viewer = Viewer.objects.create(
                username=user,
                email=email,
                password=hashed_password
            )

            return JsonResponse({
                'message': 'Signup successful',
                'viewer_id': str(viewer.viewer_id)
            }, status=201)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


# --- LOGIN VIEW ---
@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password')

            if not (email and password):
                return JsonResponse({'error': 'Missing credentials'}, status=400)

            try:
                viewer = Viewer.objects.get(email=email)
            except Viewer.DoesNotExist:
                return JsonResponse({'error': 'Invalid email or password'}, status=401)

            if not check_password(password, viewer.password):
                return JsonResponse({'error': 'Invalid email or password'}, status=401)

            # Optional: login(request, trader.username) if using session-based login
            return JsonResponse({
                'message': 'Login successful',
                'trader_id': str(viewer.viewer_id),
                'username': viewer.username.username
            }, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def toggle_favourite(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            viewer_id = data.get('viewer_id')
            company_name = data.get('company_name')

            if not (viewer_id and company_name):
                return JsonResponse({'error': 'Missing fields'}, status=400)

            viewer = Viewer.objects.get(viewer_id=viewer_id)

            # Check if this company is already favourited by the user
            existing = favourite.objects.filter(user=viewer, company_name=company_name).first()

            if existing:
                existing.delete()
                return JsonResponse({'message': 'Company removed from favourites', 'is_favourite': False}, status=200)
            else:
                favourite.objects.create(user=viewer, company_name=company_name)
                return JsonResponse({'message': 'Company added to favourites', 'is_favourite': True}, status=200)

        except Viewer.DoesNotExist:
            return JsonResponse({'error': 'Viewer not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def get_favourites(request):
    if request.method == 'GET':
        try:
            viewer_id = request.GET.get('viewer_id')
            if not viewer_id:
                return JsonResponse({'error': 'Viewer ID is required'}, status=400)

            viewer = Viewer.objects.get(viewer_id=viewer_id)
            favourites = favourite.objects.filter(user=viewer).values_list('company_name', flat=True)

            return JsonResponse({'favourites': list(favourites)}, status=200)

        except Viewer.DoesNotExist:
            return JsonResponse({'error': 'Viewer not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)