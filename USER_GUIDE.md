
---

### 📘 `docs/USER_GUIDE.md`

```markdown
# 🧠 User Guide — Options Outbreak Dashboard

This document helps traders and analysts understand how to use and interpret the dashboard.

---

## 1. Interface Overview

- **Ticker Input**: Stock ticker (e.g., `AAPL`, `TSLA`)
- **Expiry Dropdown**: Auto-fetched from Yahoo Finance
- **Earnings Banner**: Displays upcoming earnings warnings
- **Volume Chart**: Options plotted by strike vs volume
- **ML Anomaly**: Red markers flagged by Isolation Forest
- **Headlines**: Top 10 recent news stories w/ sources
- **Summary**: Number of contracts flagged as suspicious

---

## 2. Behind the Scenes

### 📈 Option Data

Pulled from `yfinance.Ticker(option_chain=expiry)` and includes:

- Strike
- Volume
- Open Interest
- Implied Volatility

### 🧠 Anomaly Detection

Uses `IsolationForest` with 10% contamination threshold to detect volume/volatility outliers.

### 📰 News Integration

Uses NewsAPI to pull real-time news sorted by `publishedAt`. The top 10 articles are shown with date, title, and source.

### 💬 Sentiment Analysis

Uses `ProsusAI/finbert` from HuggingFace for financial sentiment scoring:
- `POSITIVE` → +1
- `NEUTRAL` → 0
- `NEGATIVE` → -1

Sentiment scores are averaged for headline list context.

---

## 3. Known Issues

- NewsAPI has a request limit (100/day free tier)
- YFinance data can be stale/missing for small tickers
- ML model is retrained on every ticker switch (performance can be improved)

---

## 4. Future Improvements

- Add authentication for deploying securely
- Integrate options volume alerts via email or SMS
- Cache ticker data for faster reloads

---
