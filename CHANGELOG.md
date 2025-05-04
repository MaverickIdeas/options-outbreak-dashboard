# 📘 Options Outbreak Dashboard — Changelog

## 🔄 Version: May 3, 2025 Update

---

### 🧠 FinBERT Sentiment Enhancements
- ✅ Headline-level sentiment display with emoji (🟢 Bullish, 🟡 Neutral, 🔴 Bearish)
- ✅ Displayed next to each NewsAPI headline on the dashboard
- ✅ Average sentiment score calculated and displayed in the summary
- ✅ Debug output prints headline list and FinBERT raw outputs (if `DEBUG_MODE=true`)

---

### 📰 Headline Parsing Improvements
- ⬆️ Increased NewsAPI headline count from 10 ➜ 30
- ✅ Extracts plain title from structured Dash elements for sentiment analysis
- ✅ Filters out extra fragments (e.g. after em dashes) for cleaner NLP input

---

### 📅 Earnings Date Warning System
- ✅ Fallback logic to extract earnings date from:
  - `nextEarningsDate` (string or datetime)
  - `earningsDate` (list from yfinance)
  - `stock.calendar` (both DataFrame and dict format supported)
- ✅ Displays banner:
  - 🚨 If earnings are within 7 days
  - 📅 If earnings are in the future
  - 📅 If past earnings already occurred
- ✅ Robust debug logging to show detected fields and fallback errors

---

### 🛠️ General Enhancements
- ✅ Centralized `DEBUG_MODE` flag from `.env`
- ✅ Verbose debugging across:
  - Metadata fetch
  - NewsAPI errors
  - Sentiment analysis issues
  - Earnings parsing logic
- ✅ Pipeline for FinBERT is loaded once at the top