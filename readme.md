# 📈 Sentiment-Driven Stock Price Predictor

A hackathon project that uses sentiment analysis of social media and financial news to predict next-day stock prices of Indian blue-chip companies.

---

## 🚀 Overview

This project collects and analyzes real-time sentiment data from financial news and social media platforms every 6 hours for 10 selected Indian blue-chip stocks. It then uses the sentiment scores along with recent stock price data to predict the stock's closing price for the next day using a machine learning model.

---

## 🛠️ Tech Stack

- **Backend Framework**: Python (Django)
- **Task Scheduling & Queue**: Celery + Redis
- **Natural Language Processing**: NLTK + VADER Sentiment Analyzer
- **Machine Learning**: Scikit-learn
- **Stock Data Source**: Yahoo Finance API / NSE API
- **News Sources**: Financial APIs (e.g., Reuters, Bloomberg)
- **Social Media Sources**: Twitter API / Reddit API (or scrapers)

---

## 🔁 How It Works

### 1. Data Collection

- Every 6 hours, a Celery worker:
  - Collects ~80 social media posts and ~20 financial news articles per company.
  - This is done for **10 Indian blue-chip companies**.
  - After processing all 10 companies, it waits 6 hours and repeats the cycle.

### 2. Sentiment Analysis

- Uses `nltk` and `vaderSentiment` to score each post/article:
  - Scores range from negative (e.g., -3.2) to positive (e.g., +2.5).
- Sentiment values are averaged and stored per company per cycle.

### 3. Stock Price Tracking

- Retrieves the **opening and closing prices** for the last 7 days for each stock.

### 4. Prediction

- Sentiment scores + 7 days of historical stock prices are fed into a **Scikit-learn regression model**.
- The model outputs the **predicted closing price** for the next trading day.

---

## 🧠 Machine Learning Model

- **Features**:
  - 7-day stock price data (open & close)
  - Recent sentiment scores
- **Model**: Linear Regression (extendable to advanced models like XGBoost or LSTM)
- **Target**: Next-day closing price of the stock

---

## 📦 Installation & Setup

### Clone the repository

```bash
git clone https://github.com/yourusername/sentiment-stock-predictor.git
cd sentiment-stock-predictor

Install dependencies

bash
pip install -r requirements.txt

Set up Redis
Make sure Redis server is running:
redis-server


Start Celery and the FastAPI server
bash
Copy
Edit

celery -A app.celery_app worker --loglevel=info
uvicorn app.main:app --reload


.
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── celery_app.py        # Celery config
│   ├── tasks.py             # Scheduled data collection tasks
│   ├── sentiment/           # Sentiment analysis logic
│   ├── ml_model/            # Machine learning training/prediction code
│   └── data/                # Raw and processed data
├── requirements.txt
├── README.md
└── .env                     # API keys and configs


🔮 Example Output
Tata Steel:

Last sentiment score: +1.8

Predicted closing price (tomorrow): ₹128.35

Reliance:

Last sentiment score: -0.7

Predicted closing price (tomorrow): ₹2673.10

🕒 Automation Cycle
Task workers (Celery) are configured to:

Run jobs every 6 hours

Collect, analyze, and store sentiment & stock data

Predict prices

Sleep and repeat

📊 Future Scope
Upgrade sentiment analysis using transformer-based models (e.g., FinBERT, BERT)

Add dashboard with Next.js or React for visualization

Use Docker & Kubernetes for scalable deployment

Add alert system for extreme sentiment shifts

Explore DeFi sentiment and crypto markets



📝 License
This project is open-source under the MIT License.
Let me know if you’d like me to help add badges, screenshots, or set this up for deployment!
