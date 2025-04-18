# 💹 PractoTrade — Sentiment-Driven Stock Price Predictor

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
- **Database**: SQLite / PostgreSQL
- **DevOps**: Dockerized for easy setup and deployment (optional)

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

---

## 📈 Machine Learning Models

- **Inputs**:
  - Past 7 days' open, high, low, close, volume
  - Latest sentiment score (for model 2)
- **Model Type**: Linear Regression (extendable to LSTM or XGBoost)
- **Target**: Next-day closing price

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

### 4. (Optional) Run Scheduled Jobs
```bash
python manage.py fetch_daily_sentiment
```

---

## 📊 Sample Output

```
Company: Tata Consultancy Services (TCS)
Predicted Close (with Sentiment): ₹3578.40
Predicted Close (without Sentiment): ₹3590.10
📌 Final Aggregated Prediction: ₹3584.25
```

---

## 🛣️ Future Scope

- [ ] Upgrade to **FinBERT** for more accurate financial sentiment detection
- [ ] Integrate real-time data visualization dashboard with charts
- [ ] Add prediction confidence intervals
- [ ] Containerize with Docker for scalable deployments
- [ ] Send alerts via Telegram/Email for abnormal market sentiment shifts

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
- Reliance, TCS, Infosys, HDFC Bank, ICICI, Tata Steel, Axis Bank, SBI, Wipro, HCL Tech

---

## 📄 License

This project is open-source under the **MIT License**.

---

## 🔗 Contributing

PRs, Issues, and Feature Requests are welcome!  
Just fork the repo, branch out, and submit a pull request 🙌

---

> 💡 _Made with ❤️ for innovation at Hackathons — PractoTrade is not financial advice_
