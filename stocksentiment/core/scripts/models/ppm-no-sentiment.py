import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# List of companies, including Meta (META), Tesla (TSLA), and Microsoft (MSFT)
companies = ['TCS.NS', 'RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
             'META', 'TSLA', 'MSFT']

# Download the last 6 months of data for the companies
data = yf.download(companies, period="6mo", group_by='ticker')

# Create an empty list to store the dataframes for each company
df_list = []

# Extract relevant data for each company
for company in companies:
    company_data = data[company][['Open', 'High', 'Low', 'Close']].reset_index()
    company_data['Company'] = company  # Add a 'Company' column for identification
    df_list.append(company_data)

# Combine all the company data into one dataframe
df = pd.concat(df_list, ignore_index=True)

# Drop rows with missing values in relevant columns
df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

# Define features and target
features = ['Open', 'High', 'Low']
target = 'Close'

X = df[features]
y = df[target]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model
joblib.dump(model, "stock_model_v2.pkl")
print("Model saved as stock_model_v2.pkl")
