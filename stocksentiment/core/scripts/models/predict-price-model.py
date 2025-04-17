# sourcery skip: pandas-avoid-inplace
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("stock_sentiment_dataset.csv")

# Drop rows with missing values
df.dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close', 'SentimentScore'], inplace=True)

# Define features and target
features = ['Open', 'High', 'Low', 'Volume', 'SentimentScore']
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
joblib.dump(model, "stock_model.pkl")
print("Model saved as stock_model.pkl")
