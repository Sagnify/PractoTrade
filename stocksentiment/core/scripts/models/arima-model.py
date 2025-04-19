import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# 1. Function to get stock data for multiple companies (e.g., TCS, RELIANCE, etc.)
def get_multiple_companies_data(symbols):
    all_data = []
    for symbol in symbols:
        # Fetch 6 months of historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if not data.empty:
                data['Symbol'] = symbol  # Add a column for the company symbol
                all_data.append(data)
            else:
                print(f"Warning: No data downloaded for {symbol}")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
    
    if not all_data:
        raise ValueError("No data was downloaded for any symbols")
    
    # Concatenate all the DataFrames
    combined_data = pd.concat(all_data)
    
    # Reset index to make Date a column instead of the index
    combined_data = combined_data.reset_index()
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Columns: {combined_data.columns.tolist()}")
    
    return combined_data

# 2. Function to prepare the data for training (calculating average price)
def prepare_data_for_training(data):
    # Make a copy of the data to avoid any reference issues
    df = data.copy()
    
    # Check column types
    print("Column types before conversion:")
    print(df.dtypes)
    
    # Check for MultiIndex columns - if present, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        print(f"Flattened MultiIndex columns: {df.columns.tolist()}")
    
    # Convert date column to datetime if it's not already
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Create new columns for Open, High, Low, Close by averaging across all stocks
    price_types = ['Open', 'High', 'Low', 'Close']
    
    for price_type in price_types:
        # Find all columns that contain this price type
        matching_cols = [col for col in df.columns if col.startswith(f"{price_type} ")]
        
        if matching_cols:
            print(f"Found {len(matching_cols)} columns for {price_type}: {matching_cols}")
            # Create a new column with the average of all stocks for this price type
            df[price_type] = df[matching_cols].mean(axis=1)
        else:
            print(f"No columns found for {price_type}")
            # If no columns found, set to NaN
            df[price_type] = np.nan
    
    # Check if we have the required columns now
    for col in price_types:
        if col not in df.columns:
            raise ValueError(f"Failed to create {col} column")
    
    # Calculate average price (mean of 'Open', 'High', 'Low', 'Close')
    df['AveragePrice'] = df[price_types].mean(axis=1)
    
    # Handle missing values by filling with the previous day's value
    df = df.fillna(method='ffill')
    
    print("Data preparation complete!")
    return df

# 3. Function to train ARIMA model on combined data
def train_arima_model(data):
    try:
        # Debug info
        print(f"Train data shape: {data.shape}")
        
        # Prepare the data for ARIMA
        data = prepare_data_for_training(data)
        
        # Ensure 'Date' is in the proper datetime format
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        else:
            raise ValueError("'Date' column not found in the data.")
        
        # Check for AveragePrice column
        if 'AveragePrice' not in data.columns:
            raise ValueError("AveragePrice column not found after data preparation")
        
        # Group by date and calculate mean of AveragePrice
        print("Grouping data by date...")
        data_grouped = data.groupby('Date')['AveragePrice'].mean().reset_index()
        
        print(f"Grouped data shape: {data_grouped.shape}")
        print(f"First few rows of grouped data:\n{data_grouped.head()}")
        
        # Convert to time series for ARIMA
        time_series = data_grouped.set_index('Date')['AveragePrice']
        
        # Check if we have enough data points
        if len(time_series) < 10:
            raise ValueError(f"Not enough data points for ARIMA modeling. Got {len(time_series)} points.")
        
        print(f"Training ARIMA model with {len(time_series)} data points...")
        
        # Fit ARIMA model on the 'AveragePrice' column
        model = ARIMA(time_series, order=(5,1,0))
        model_fit = model.fit()
        
        # Save the trained model for later use using joblib with .pkl extension
        joblib.dump(model_fit, 'arima_model_combined.pkl')
        
        print("Model trained and saved successfully!")
        return model_fit
    
    except Exception as e:
        print(f"Error in train_arima_model: {e}")
        import traceback
        traceback.print_exc()
        raise

# 4. Function to load the trained ARIMA model
def load_model(filename="arima_model_combined.pkl"):
    model_fit = joblib.load(filename)
    return model_fit

# Main function to train and save the model
def main():
    try:
        # Define list of companies (Indian companies in this case)
        companies = ['TCS.NS', 'RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        
        # Step 1: Get historical data for multiple companies
        data = get_multiple_companies_data(companies)
        
        # Step 2: Train ARIMA model on the combined data and save the model
        model_fit = train_arima_model(data)
        
        print("The model is saved as 'arima_model_combined.pkl'. You can use it in other codes.")
    
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()