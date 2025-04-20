<img src="https://raw.githubusercontent.com/RaunakDiesFromCode/PractoTrade-app/main/public/practo.png" alt="drawing" width="200"/>

# 📉 PractoTrade — Sentiment-Driven Stock Price Predictor

A hackathon project that forecasts the **next-day closing prices** of Indian blue-chip stocks by combining **real-time sentiment analysis** with **historical market data** using dual regression models.

---

<img src="https://i.ibb.co/Tq271j2G/image.png" alt="stock-Details" />

---

## 🚀 Overview

**PractoTrade** is a full-stack AI application that analyzes financial news and social media sentiment every few hours and blends it with recent stock performance to predict the next day's closing price. It leverages **two regression models**—one with sentiment scores, and one without—to return an **aggregated and smarter prediction**.

---

## 🛠️ Tech Stack

- **Backend**: Django (App name: `core`)
- **Frontend**: Next.js + ShadCN (⚡ polished and production-ready)
- **Scheduler**: Celery + Redis (runs automatic sentiment polling every 6 hours)
- **Machine Learning**: Scikit-learn (2 custom regression models)
- **NLP**: VADER via NLTK for sentiment scoring
- **Data Source**: Yahoo Finance API, Reddit, Google News scraping
- **Database**: PostgreSQL
- **Caching**: django-redis

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

### 📊 Backend Workflow

<img src="https://i.ibb.co/2YK1SCBd/Sentiment-Algo.jpg" alt="Sentiment-Algo" />

<img src="https://i.ibb.co/JFG1Th5F/prediction-Algo.jpg" alt="prediction-Algo" />

---

## 🧠 Machine Learning Models

### Model 1: Stock Prediction without Sentiment Analysis

#### Inputs:
- Past 7 days of stock data (Open, High, Low, Close, Volume)

#### Model:
- **Linear Regression**

#### Output:
- Next-day closing price

---

### Model 2: Stock Prediction with Sentiment Analysis

#### Inputs:
- Same past 7 days' stock data
- Latest sentiment score (aggregated from news and Reddit)

#### Model:
- **Linear Regression** with future scope for **LSTM** or **XGBoost**

#### Output:
- Next-day closing price

---

### Model 3: ARIMA

- Statistical approach for time series forecasting using historical stock data.

---

## 📝 Sentiment Analysis Using VADER

### Data Collection:
- 20 Reddit posts
- 10 News articles

### Sentiment Analysis:
- Scored as Positive / Neutral / Negative using VADER

### Aggregation:
- Sentiment scores are averaged across all sources for each company

### Scheduler:
- Runs every 6 hours to update sentiment data

---

## 🔮 Prediction Process

### Inputs:
- Past 7 days of stock data
- Latest sentiment score

### Execution:
1. Model 1: With sentiment
2. Model 2: Without sentiment
3. Model 3: ARIMA

### Final Output:
- **Average of all three models** for final prediction

---

## 🎯 Final Prediction Strategy

- Combines:
  - **Model 1**: With sentiment
  - **Model 2**: Without sentiment
  - **Model 3**: ARIMA

> 📌 Future scope: Weighted average based on model confidence

---

## 🖼️ UI Screens

<img src="https://i.ibb.co/Xfgx163Y/homePage.jpg" alt="homePage" />

<img src="https://i.ibb.co/7xGMNtvC/stock-Details.jpg" alt="stock-Details" />

---

## ⚙️ Setup Instructions

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

### 4. Start Celery & Redis

**Redis:**
```bash
wsl start
redis-server
redis-cli
```

**Celery:**
```bash
celery -A stocksentiment worker --loglevel=info --pool=solo
celery -A stocksentiment beat --loglevel=info
```

---

## 🧾 Sample Output

```
Company: Tata Consultancy Services (TCS)
Predicted Close (with Sentiment): ₹3578.40
Predicted Close (without Sentiment): ₹3590.10
📌 Final Aggregated Prediction: ₹3584.25
```

---

## 🌱 Future Scope

- Upgrade to **FinBERT** for more accurate sentiment detection
- Add interactive charts and analytics
- Show **confidence intervals** and model evaluation
- Improve Dockerization
- Add **Telegram/email alerts** for sentiment shifts
- Support **mobile frontend** & PWA
- Add support for **multi-day and intraday predictions**
- Weekly model retraining
- Use weighted model outputs based on performance

---

## ✨ Features

- ⏱️ Automated data fetching every 6 hours
- 🔍 Sentiment-powered predictions
- 📊 Dual-model prediction logic
- 📈 Company trend tracking
- 💻 Clean frontend with poll UI
- 🗳️ Community-driven sentiment polls

---

## 🧭 Focused Companies

- Reliance, TCS, Infosys, HDFC Bank, ICICI, Wipro, Hindustan Unilever, META, TESLA

---

## 📄 License

Open-source under the **MIT License**

---

## 🤝 Contributing

Pull requests, feature ideas, and issues are welcome. Fork the repo, branch out, and raise a PR 🙌

---

> 💡 *Made with ❤️ at Hackathons — PractoTrade is not financial advice*







