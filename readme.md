# 📉 PractoTrade — Sentiment-Driven Stock Price Predictor

A hackathon project that forecasts the **next-day closing prices** of Indian blue-chip stocks by combining **real-time sentiment analysis** with **historical market data** using dual regression models.

---

## 🚀 Overview

**PractoTrade** is a full-stack AI application that analyzes financial news and social media sentiment every few hours and blends it with recent stock performance to predict the next day's closing price. It leverages **two regression models**—one with sentiment scores, and one without—to return an **aggregated and smarter prediction**.

---

## 🛠️ Tech Stack

- **Backend**: Django (App name: `core`)
- **Frontend**: NEXT JS + ShadCN (⚡ polished and production-ready)
- **Scheduler**: Celery + Redis (runs automatic sentiment polling every 6 hours)
- **Machine Learning**: Scikit-learn (2 custom regression models)
- **NLP**: VADER via NLTK for sentiment scoring
- **Data Source**: Yahoo Finance API, Reddit, Google News scraping
- **Database**: PostgreSQL



---

## 🔁 Workflow

### 1. Automated Data Collection

- Runs every 6 hours using scheduled management commands.
- For each predefined company:
  - Scrapes top posts from Reddit and headlines from Google News.
  - Assigns a **sentiment score** using VADER.

### 2. Data Aggregation

- Stores:
  - Past 7 days' OHLC stock data
  - Real-time sentiment scores
- Saves everything in the `CompanySentiment` model.

### 3. Prediction

- Two regression models are trained:
  - **Model 1**: Predicts next-day close using **historical data only**
  - **Model 2**: Uses **historical data + sentiment score**
- Final result is the **average of both model outputs** for stability.

### 4. Community Polling System

- Uses a management command to automatically post daily polls for each company.
- Rotates through a set of predefined question templates.
- To run the daily poll post logic manually:

```bash
python manage.py create_daily_poll
```

This helps collect user sentiment data for each company.

---

## 📊 Machine Learning Models

### Model 1: Stock Prediction without Sentiment Analysis

#### **Inputs**:
- **Past 7 days' stock data**:
  - **Open**, **High**, **Low**, **Close**, and **Volume** for the previous 7 trading days. These features capture the key market dynamics such as trends, volatility, and trading activity.

#### **Model Type**:
- **Linear Regression**: A basic linear regression model is employed to model the relationship between historical stock data and the next day's closing price. The model assumes a linear correlation between the input features and the target variable.

#### **Target**:
- **Next-day closing price**: The model predicts the closing price for the next trading day based on the patterns observed from the past week's stock data.

---

### Model 2: Stock Prediction with Sentiment Analysis

#### **Inputs**:
- **Past 7 days' stock data**:
  - Same as Model 1, using the **Open**, **High**, **Low**, **Close**, and **Volume** values for the last 7 trading days.
  
- **Latest sentiment score**:
  - A sentiment score derived from various sources such as news articles, social media, and financial forums. This score quantifies the market's sentiment (positive, neutral, or negative) towards the stock, providing an additional layer of insight into market psychology and external influences on stock price movement.

#### **Model Type**:
- **Linear Regression** (with potential extensions):
  - The model combines both historical stock data and sentiment scores using linear regression. However, the architecture is designed to be extended to more advanced models, such as:
    - **LSTM (Long Short-Term Memory)**: A deep learning technique for time series forecasting that captures temporal dependencies and long-term trends in stock prices.
    - **XGBoost**: A gradient-boosted tree model known for its efficiency in regression tasks, capable of handling complex, non-linear relationships between input features and the target.

#### **Target**:
- **Next-day closing price**: The model predicts the next-day closing price, incorporating both historical stock data and the sentiment score to improve prediction accuracy.

---

### Post-Processing with Sentiment Analysis Using VADER

The sentiment analysis process is an essential part of the prediction pipeline. Here's how it works:

#### **Data Collection**:
- **Reddit Posts**: The latest 20 Reddit posts related to each company are fetched.
- **News Articles**: The latest 10 news articles related to each company are fetched from various news sources.

#### **Sentiment Analysis**:
- Each Reddit post and news article is analyzed for sentiment using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, a sentiment analysis tool designed for social media and textual data.
  - **Positive Sentiment**: The article/post expresses optimism or positive outlook towards the company.
  - **Negative Sentiment**: The article/post expresses pessimism or a negative outlook.
  - **Neutral Sentiment**: The sentiment is balanced or neutral.

#### **Sentiment Aggregation**:
- After analyzing each post and article, a **total aggregated sentiment score** is calculated. This score represents the overall market sentiment towards the company at the time of the analysis.
  - The aggregation involves averaging the sentiment scores of the 20 Reddit posts and 10 news articles for each company.
  - The resulting score reflects the collective sentiment (positive, neutral, or negative) toward the company at that specific point in time.

#### **Scheduled Analysis**:
- The sentiment analysis process is repeated at regular intervals:
  - Every **6 hours**, the system fetches the latest posts and articles, performs sentiment analysis, and updates the sentiment score for the company.

---

### Prediction Process

After the sentiment score is updated, the prediction models are used to forecast the next day's closing price based on both the sentiment data and historical stock data.

#### **Data Input**:
- For each day, the model uses:
  - The **historical stock data** from the last 7 days (Open, High, Low, Close, Volume).
  - The **aggregated sentiment score** from the latest analysis of Reddit posts and news articles.

#### **Model Execution**:
1. **Model 1** (without sentiment) processes the historical stock data to predict the next day's closing price.
2. **Model 2** (with sentiment) incorporates both the historical stock data and the sentiment score for a more nuanced prediction.

#### **Final Prediction**:
- The final prediction is an **aggregate of both models' outputs** (with and without sentiment). This combined prediction aims to provide a more accurate next-day closing price by leveraging both quantitative market data and qualitative sentiment insights.

---

### Final Prediction Strategy

The final predicted next-day closing price is typically an aggregate of the predictions from both models:
- **Model 1**: Prediction based on historical stock data.
- **Model 2**: Prediction enhanced with sentiment analysis.

By combining these predictions, the system aims to leverage both quantitative (price and volume) and qualitative (market sentiment) data to generate a more robust and accurate stock price forecast.


---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sagnify/PractoTrade.git
cd PractoTrade
```

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Migrations & Server

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

---

## 📈 Sample Output

```
Company: Tata Consultancy Services (TCS)
Predicted Close (with Sentiment): ₹3578.40
Predicted Close (without Sentiment): ₹3590.10
📌 Final Aggregated Prediction: ₹3584.25
```

---

## 🚣 Future Scope

- 🔍 Upgrade to **FinBERT** for more accurate financial sentiment detection
- 📊 Add real-time data visualization with interactive charts and analytics
- 🔢 Introduce prediction **confidence intervals** and model evaluation metrics
- 🐳 Improve containerization with Docker Compose & environment variable configs
- 📩 Send **email/Telegram alerts** for abnormal market sentiment or prediction shifts
- 📱 Add mobile-friendly frontend and PWA capabilities
- ⏱️ Support for **multi-day predictions** and intraday forecasts


---

## ✨ Features

- 🔁 **Automated Data Collection** (every 6 hours)
- 📊 **Dual-Model Prediction** logic for reliability
- 🧠 **NLP-Powered Sentiment Analysis** from real news/posts
- 📉 **Company-wise Trend Tracking**
- 💻 **Beautiful Frontend** with no-code poll UI
- 🗳️ **Community Polling System** where users can vote daily on market sentiment per company

---

## 📍 Focused Companies

- Reliance, TCS, Infosys, HDFC Bank, ICICI, Wipro, Hindustan Unilever, META, TESLA

---

## 📄 License

This project is open-source under the **MIT License**.

---

## 🔗 Contributing

PRs, Issues, and Feature Requests are welcome!\
Just fork the repo, branch out, and submit a pull request 🙌

---

> 💡 *Made with ❤️ for innovation at Hackathons — PractoTrade is not financial advice*

