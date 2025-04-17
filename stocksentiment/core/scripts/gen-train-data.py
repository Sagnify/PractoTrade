import datetime
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import time

# # Optional sentiment packages - currently unused
# import requests
# import feedparser
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from urllib.parse import quote_plus
# import random

# analyzer = SentimentIntensityAnalyzer()

# # Sentiment-related functions (commented out)
# def fetch_reddit_posts(...): ...
# def fetch_clean_google_news(...): ...
# def get_average_sentiment_for_date(...): ...

def get_last_month_stock_data(ticker, company_name):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
    df.reset_index(inplace=True)
    
    # Add company and ticker info
    df['Company'] = company_name
    df['Ticker'] = ticker

    # Optional: only keep required columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Company', 'Ticker']]
    
    return df

# Dictionary of ticker: company name
tickers = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "ITC.NS": "ITC"
}

# List to hold all data
all_dataframes = []
for ticker, company in tickers.items():
    print(f"Fetching data for {company} ({ticker})...")
    df = get_last_month_stock_data(ticker, company)
    all_dataframes.append(df)
    time.sleep(1)  # Avoid hitting API rate limits

# Combine and save
combined_df = pd.concat(all_dataframes)
combined_df.to_csv("stock_dataset.csv", index=False)
print("Saved dataset to stock_dataset.csv")
