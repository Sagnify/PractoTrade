from django.shortcuts import render,HttpResponse
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


# Set up the model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stock_model.pkl')

# Create a function to lazily load the model
def get_model():
    if not hasattr(get_model, "_model"):
        # Load the model only once, when it is needed
        print("Loading model...")
        get_model._model = joblib.load(MODEL_PATH)
    return get_model._model

# Usage example
model = get_model()

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
            'predicted_Close': round(float(prediction.predicted_price), 2),
            'prediction_time': prediction.prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_percentage_change': prediction.predicted_percentage_change,
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
        'META': 'Meta',
        'TSLA': 'Tesla',
        'MSFT': 'Microsoft',
        # 'GOOGL': 'Google',
        # 'AAPL': 'Apple',
        'TCS.NS': 'Tata Consultancy Services',
        'INFY.NS': 'Infosys',
        'HDFCBANK.NS': 'HDFC Bank',
        'RELIANCE.NS': 'Reliance Industries',
        'WIPRO.NS': 'Wipro',
        'ITCLTD.NS': 'ITC',
        'HINDUNILVR.NS': 'Hindustan Unilever',
    }
    return JsonResponse({'companies': companies})