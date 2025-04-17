from django.shortcuts import render,HttpResponse
import requests
from .scripts import fetch_analyze as fa
from .models import CompanySentiment, StockPrediction
from django.http import JsonResponse
from django.utils.timezone import now
from datetime import datetime, timedelta
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

# Usage example
# model = get_model_1()
# model_2 = get_model_2()

def get_last_close_price(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # In case of weekends/holidays

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # if not data.empty:
    #     # Get the last available close price
    #     last_close = data['Close'].iloc[-1]
    #     return float(last_close)
    # else:
    #     return None

    if data is None or data.empty:  # Check both for None and empty DataFrame
        return None

    # Get the last available close price
    last_close = data['Close'].iloc[-1]
    return float(last_close)

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
                        'predicted_price_with_sentiment': 0,
                        'predicted_price_without_sentiment': 0,
                        'avg_predicted_price': 0,
                        'prediction_time': datetime.now(),
                        'predicted_percentage_change': 0,
                        'direction': 'neutral',
                    }
                )
                errors.append({'company': company_name, 'error': 'Insufficient data'})
                continue

            model_with_sentiment = get_model_1()
            model_without_sentiment = get_model_2()

            # Prepare input data
            data = []
            data_no_sentiment = []
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

            df = pd.DataFrame(data)
            df_no_sentiment = pd.DataFrame(data_no_sentiment)

            features_with_sentiment = df.mean().tolist()
            features_without_sentiment = df_no_sentiment.mean().tolist()

            # Make predictions
            pred_with_sentiment = model_with_sentiment.predict([features_with_sentiment])[0]
            pred_without_sentiment = model_without_sentiment.predict([features_without_sentiment])[0]
            avg_pred = (pred_with_sentiment + pred_without_sentiment) / 2

            # Get last close price
            last_close_price = get_last_close_price(company_name)
            if last_close_price:
                percentage_change = ((avg_pred - last_close_price) / last_close_price) * 100
                direction = 'up' if percentage_change > 0 else 'down' if percentage_change < 0 else 'neutral'
            else:
                percentage_change = 0
                direction = 'neutral'

            # Save to DB
            StockPrediction.objects.update_or_create(
                company_name=company_name,
                defaults={
                    'predicted_price_with_sentiment': round(float(pred_with_sentiment), 2),
                    'predicted_price_without_sentiment': round(float(pred_without_sentiment), 2),
                    'avg_predicted_price': round(float(avg_pred), 2),
                    'prediction_time': datetime.now(),
                    'predicted_percentage_change': round(float(percentage_change), 2),
                    'direction': direction,
                }
            )

            predictions.append({
                'company': company_name,
                'with_sentiment': round(float(pred_with_sentiment), 2),
                'without_sentiment': round(float(pred_without_sentiment), 2),
                'average': round(float(avg_pred), 2),
                'predicted_percentage_change': round(float(percentage_change), 2),
                'direction': direction,
            })

        except Exception as e:
            errors.append({'company': company_name, 'error': str(e)})

    return JsonResponse({'predictions': predictions, 'errors': errors}, status=200)




@csrf_exempt
def get_predicted_stock_price(request, company_name):
    if request.method != 'GET':
        return

    try:
        # Get the latest prediction for the company
        prediction = StockPrediction.objects.filter(company_name=company_name).order_by('-prediction_time').first()

        if not prediction:
            return JsonResponse({'error': 'Prediction not found for this company'}, status=404)

        # Build the base URL
        base_url = request.build_absolute_uri('/')[:-1]  # removes trailing slash

        # API links for different periods
        candle_urls = {
            'realtime': f"{base_url}/api/stock-chart/?company={company_name}&interval=realtime",
            '1d': f"{base_url}/api/stock-chart/?company={company_name}&interval=1d",
            '7d': f"{base_url}/api/stock-chart/?company={company_name}&interval=7d",
        }

        # Return the predicted information along with the API links
        return JsonResponse({
            'company': company_name,
            'predicted_with_sentiment': round(float(prediction.predicted_price_with_sentiment), 2),
            'predicted_without_sentiment': round(float(prediction.predicted_price_without_sentiment), 2),
            'avg_predicted_price': round(float(prediction.avg_predicted_price), 2),
            'prediction_time': prediction.prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_percentage_change': round(float(prediction.predicted_percentage_change), 2),
            'direction': prediction.direction,
            'api_links': candle_urls,
        }
        , status=200)

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
        'META': {'ticker': 'META', 'name': 'Meta',  'is_in': False, 'description': 'Meta (formerly Facebook) is a global leader in social media and virtual reality.'},
        'TSLA': {'ticker': 'TSLA', 'name': 'Tesla',  'is_in': False, 'description': 'Tesla is an electric vehicle and clean energy company, revolutionizing transportation.'},
        'MSFT': {'ticker': 'MSFT', 'name': 'Microsoft',  'is_in': False, 'description': 'Microsoft is a global technology company known for software, hardware, and cloud services.'},
        'TCS': {'ticker': 'TCS.NS', 'name': 'Tata Consultancy Services',  'is_in': True, 'description': 'TCS is a leading global IT services and consulting company from India.'},
        'INFY': {'ticker': 'INFY.NS', 'name': 'Infosys',  'is_in': True, 'description': 'Infosys is an Indian multinational corporation that provides IT and consulting services.'},
        'HDFCBANK': {'ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank',  'is_in': True, 'description': 'HDFC Bank is one of India’s largest private sector banks offering a wide range of financial services.'},
        'RELIANCE': {'ticker': 'RELIANCE.NS', 'name': 'Reliance Industries',  'is_in': True, 'description': 'Reliance Industries is a conglomerate with businesses in petrochemicals, retail, and telecommunications.'},
        'WIPRO': {'ticker': 'WIPRO.NS', 'name': 'Wipro',  'is_in': True, 'description': 'Wipro is an Indian multinational corporation providing IT services and consulting.'},
        # 'ITC': {'ticker': 'ITCLTD.NS', 'name': 'ITC',  'is_in': True, 'description': 'ITC is an Indian conglomerate with businesses in FMCG, hotels, paperboards, and packaging.'},
        'HINDUNILVR': {'ticker': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever',  'is_in': True, 'description': 'Hindustan Unilever is a leading Indian consumer goods company offering products in health, beauty, and home care.'},
    }


    return JsonResponse({'companies': companies})

@csrf_exempt
def reddit_post_fetcher_by_company(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    ticker = request.GET.get('ticker', '').upper()
    if not ticker:
        return JsonResponse({'error': 'Ticker is required'}, status=400)

    subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket', 'RobinHood', 'options']
    post_links = []

    for subreddit in subreddits:
        try:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            headers = {'User-agent': 'Mozilla/5.0'}
            params = {
                'q': ticker,
                'sort': 'new',
                'restrict_sr': 'on',
                'limit': 10
            }

            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                for post in data.get('data', {}).get('children', []):
                    link = f"https://reddit.com{post['data']['permalink']}"
                    if link not in post_links:
                        post_links.append(link)
                    if len(post_links) >= 5:
                        break

            if len(post_links) >= 5:
                break
        except Exception as e:
            continue

    return JsonResponse({'ticker': ticker, 'post_links': post_links[:5]})