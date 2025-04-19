import random
from django.shortcuts import render,HttpResponse
import feedparser
import requests
from .scripts import fetch_analyze as fa
from .models import CompanySentiment, StockPrediction, StockPrediction, DailyPoll, PollOption, Vote
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


company_tickers = [
    'META',         # Meta
    'TSLA',         # Tesla
    'MSFT',         # Microsoft
    # 'GOOGL',      # Google
    # 'AAPL',       # Apple
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
def predict_all_stock_prices(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET method allowed'}, status=405)

    predictions = []
    errors = []


    for company_name in company_tickers:
        
        try:
            entries = CompanySentiment.objects.filter(company_name=company_name).order_by('-timestamp')[:7]

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

            data = []
            data_no_sentiment = []
            closing_prices = []
            timestamps = []

            for entry in entries:
                stock_data = entry.stock_data

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

                close_price = stock_data.get('close')
                if close_price is not None:
                    closing_prices.append(float(close_price))
                    timestamps.append(entry.timestamp)

            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'volume', 'sentiment_score'])
            df_no_sentiment = pd.DataFrame(data_no_sentiment, columns=['open', 'high', 'low'])

            try:
                model_with_sentiment = get_model_1()
                model_without_sentiment = get_model_2()

                features_with_sentiment = df.mean().values.tolist()
                features_without_sentiment = df_no_sentiment.mean().values.tolist()

                pred_with_sentiment = model_with_sentiment.predict([features_with_sentiment])[0]
                pred_without_sentiment = model_without_sentiment.predict([features_without_sentiment])[0]
            except Exception as e:
                print(f"Error predicting for {company_name} using RandomForest: {str(e)}")
                pred_with_sentiment = 0
                pred_without_sentiment = 0


            arima_pred = 0
            try:
                if len(closing_prices) >= 7:
                    ts = pd.Series(closing_prices, index=timestamps)
                    ts = ts.sort_index()

                    model = ARIMA(ts.values, order=(1, 1, 0))
                    model_fit = model.fit()

                    forecast = model_fit.forecast(steps=1)
                    arima_pred = forecast[0]
                else:
                    print(f"Not enough closing prices for {company_name} ARIMA model")
            except Exception as e:
                print(f"Error in ARIMA for {company_name}: {str(e)}")
                arima_pred = 0
                import traceback
                traceback.print_exc()

            # Calculate average only after all predictions are done
            avg_pred = (pred_with_sentiment + pred_without_sentiment + arima_pred) / 3
            
            last_close_price = get_last_close_price(company_name)
            if last_close_price:
                percentage_change = ((avg_pred - last_close_price) / avg_pred) * 100
                direction = 'up' if percentage_change > 0 else 'down' if percentage_change < 0 else 'neutral'
            else:
                percentage_change = 0
                direction = 'neutral'



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
    import yfinance as yf
    import plotly.graph_objects as go
    import pandas as pd
    import json
    import traceback
    from plotly.utils import PlotlyJSONEncoder
    from django.http import JsonResponse
    from django.shortcuts import render
    
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


def company_list(request):
    """
    View to return the list of companies with tickers and full names as JSON
    """
    companies = {
        'META': {'ticker': 'META', 'name': 'Meta', 'is_in': False, 'description': 'Meta (formerly Facebook) is a global leader in social media and virtual reality.'},
        'TSLA': {'ticker': 'TSLA', 'name': 'Tesla', 'is_in': False, 'description': 'Tesla is an electric vehicle and clean energy company, revolutionizing transportation.'},
        'MSFT': {'ticker': 'MSFT', 'name': 'Microsoft', 'is_in': False, 'description': 'Microsoft is a global technology company known for software, hardware, and cloud services.'},
        'TCS': {'ticker': 'TCS.NS', 'name': 'Tata Consultancy Services', 'is_in': True, 'description': 'TCS is a leading global IT services and consulting company from India.'},
        'INFY': {'ticker': 'INFY.NS', 'name': 'Infosys', 'is_in': True, 'description': 'Infosys is an Indian multinational corporation that provides IT and consulting services.'},
        'HDFCBANK': {'ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank', 'is_in': True, 'description': 'HDFC Bank is one of India’s largest private sector banks offering a wide range of financial services.'},
        'RELIANCE': {'ticker': 'RELIANCE.NS', 'name': 'Reliance Industries', 'is_in': True, 'description': 'Reliance Industries is a conglomerate with businesses in petrochemicals, retail, and telecommunications.'},
        'WIPRO': {'ticker': 'WIPRO.NS', 'name': 'Wipro', 'is_in': True, 'description': 'Wipro is an Indian multinational corporation providing IT services and consulting.'},
        'HINDUNILVR': {'ticker': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever', 'is_in': True, 'description': 'Hindustan Unilever is a leading Indian consumer goods company offering products in health, beauty, and home care.'},

        # 🌍 Global Dummy Companies
        'AMZN': {'ticker': 'AMZN', 'name': 'Amazon', 'is_in': False, 'description': 'Amazon is a multinational technology company focusing on e-commerce, cloud computing, and AI.'},
        'GOOGL': {'ticker': 'GOOGL', 'name': 'Alphabet', 'is_in': False, 'description': 'Alphabet is the parent company of Google, focusing on internet services and products.'},
        'NVDA': {'ticker': 'NVDA', 'name': 'NVIDIA', 'is_in': False, 'description': 'NVIDIA designs GPUs for gaming and professional markets, and is a key player in AI.'},

        # 🇮🇳 Indian Dummy Companies
        'ITC': {'ticker': 'ITC.NS', 'name': 'ITC', 'is_in': True, 'description': 'ITC is an Indian conglomerate with businesses in FMCG, hotels, paperboards, and packaging.'},
        'LT': {'ticker': 'LT.NS', 'name': 'Larsen & Toubro', 'is_in': True, 'description': 'L&T is a major Indian multinational in engineering, construction, and manufacturing.'},
        'BAJFINANCE': {'ticker': 'BAJFINANCE.NS', 'name': 'Bajaj Finance', 'is_in': True, 'description': 'Bajaj Finance provides a range of financial services including loans, insurance, and investment products.'}
    }



    return JsonResponse({'companies': companies})




def search(request):


    companies = {
    'META': {'ticker': 'META', 'name': 'Meta', 'is_in': False, 'description': 'Meta (formerly Facebook) is a global leader in social media and virtual reality.'},
    'TSLA': {'ticker': 'TSLA', 'name': 'Tesla', 'is_in': False, 'description': 'Tesla is an electric vehicle and clean energy company, revolutionizing transportation.'},
    'MSFT': {'ticker': 'MSFT', 'name': 'Microsoft', 'is_in': False, 'description': 'Microsoft is a global technology company known for software, hardware, and cloud services.'},
    'TCS': {'ticker': 'TCS.NS', 'name': 'Tata Consultancy Services', 'is_in': True, 'description': 'TCS is a leading global IT services and consulting company from India.'},
    'INFY': {'ticker': 'INFY.NS', 'name': 'Infosys', 'is_in': True, 'description': 'Infosys is an Indian multinational corporation that provides IT and consulting services.'},
    'HDFCBANK': {'ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank', 'is_in': True, 'description': 'HDFC Bank is one of India’s largest private sector banks offering a wide range of financial services.'},
    'RELIANCE': {'ticker': 'RELIANCE.NS', 'name': 'Reliance Industries', 'is_in': True, 'description': 'Reliance Industries is a conglomerate with businesses in petrochemicals, retail, and telecommunications.'},
    'WIPRO': {'ticker': 'WIPRO.NS', 'name': 'Wipro', 'is_in': True, 'description': 'Wipro is an Indian multinational corporation providing IT services and consulting.'},
    'HINDUNILVR': {'ticker': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever', 'is_in': True, 'description': 'Hindustan Unilever is a leading Indian consumer goods company offering products in health, beauty, and home care.'},

    }

    if request.method == 'GET':
        search_term = request.GET.get('search', '').strip()
        if not search_term:
            return JsonResponse({'error': 'Search term is required'}, status=400)

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

@csrf_exempt
@require_GET
def all_company_news(request):
    all_articles = []

    for company in COMPANY_LIST:
        rss_url = f"https://news.google.com/rss/search?q={company}"
        feed = feedparser.parse(rss_url)
        articles = [
            {
                'company': company,
                'title': entry.title,
                'url': entry.link
            }
            for entry in feed.entries[:10]
        ]
        all_articles.extend(articles)

    return JsonResponse({'news': all_articles}, safe=False)


import uuid
import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth import login
from .models import Viewer

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


