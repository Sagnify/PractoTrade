import yfinance as yf
import plotly.graph_objects as go

def fetch_and_plot_stock_data(company_name, period='1d', interval='1m'):
    # Fetch stock data using yfinance
    data = yf.download(company_name, period=period, interval=interval)

    if data.empty: # type: ignore
        print(f"No data found for {company_name}")
        return

    # Print the first few rows of the data for inspection
    print(f"Fetched data for {company_name}:")
    print(data.head()) # type: ignore
    
    # Reset index to make the date a column
    data.reset_index(inplace=True) # type: ignore
    print("Column names after reset_index:")
    print(data.columns) # type: ignore
    
    # Create the candlestick chart using Plotly with MultiIndex columns
    fig = go.Figure(data=[go.Candlestick(
        x=data[('Datetime', '')], # type: ignore
        open=data[('Open', company_name)], # type: ignore
        high=data[('High', company_name)], # type: ignore
        low=data[('Low', company_name)], # type: ignore
        close=data[('Close', company_name)], # type: ignore
        name=company_name
    )])

    fig.update_layout(
        title=f'{company_name} Candlestick Chart',
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"  # Use dark theme for better visualization
    )

    fig.show()

# Test the function for TCS (Tata Consultancy Services)
fetch_and_plot_stock_data('TCS.NS', period='1d', interval='1m')