# File: app.py
# Main application file for Options Outbreak Dashboard

import yfinance as yf
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import requests
from sklearn.ensemble import IsolationForest
from transformers import pipeline
import os
import re
from dotenv import load_dotenv
from datetime import datetime, date

# --- Load .env and API Keys ---
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# --- Initialize Dash App ---
app = dash.Dash(__name__)
app.title = "Options Outbreak Dashboard with News + ML"

# --- Define App Layout ---
app.layout = html.Div([
    html.H1("Options Outbreak Dashboard", style={"textAlign": "center"}),
    dcc.Input(id="ticker-input", type="text", placeholder="Enter Stock Ticker", value="TSLA"),
    html.Div(id="company-name", style={"textAlign": "center", "fontWeight": "bold", "marginBottom": "10px"}),
    dcc.Dropdown(id="expiry-dropdown", placeholder="Select expiry date"),
    html.Div(id="earnings-banner", style={"color": "red", "textAlign": "center", "marginBottom": "10px"}),
    dcc.Graph(id="volume-chart"),
    html.Div(id="summary-text", style={"padding": "10px"}),
    html.H3("Related Headlines:", style={"marginTop": "20px"}),
    html.Ul(id="headline-list")
])

# --- Parse earnings date from yfinance info dict ---
def parse_earnings_date(stock):
    info = stock.info
    date_obj = None
    raw1 = info.get("nextEarningsDate")
    raw2 = info.get("earningsDate")

    if DEBUG_MODE:
        print(f"📊 Earnings fields: nextEarningsDate={raw1}, earningsDate={raw2}")

    # Try known fields first
    if isinstance(raw1, datetime):
        date_obj = raw1.date()
    elif isinstance(raw1, str):
        try:
            date_obj = datetime.strptime(raw1, "%Y-%m-%d").date()
        except:
            pass
    elif isinstance(raw2, list) and raw2 and isinstance(raw2[0], datetime):
        date_obj = raw2[0].date()

    # Fallback to calendar field
    if not date_obj:
        try:
            calendar_val = stock.calendar
            if isinstance(calendar_val, pd.DataFrame):
                if not calendar_val.empty:
                    earnings_val = calendar_val.loc["Earnings Date"].values[0]
                    if isinstance(earnings_val, pd.Timestamp):
                        date_obj = earnings_val.date()
            elif isinstance(calendar_val, dict) and "Earnings Date" in calendar_val:
                maybe = calendar_val["Earnings Date"]
                if isinstance(maybe, (pd.Timestamp, datetime)):
                    date_obj = maybe.date()
        except Exception as e:
            if DEBUG_MODE:
                print(f"🛑 Calendar fallback failed: {e}")

    return date_obj

# --- Callback: Populate dropdown and company info ---
@app.callback(
    Output("expiry-dropdown", "options"),
    Output("expiry-dropdown", "value"),
    Output("company-name", "children"),
    Output("earnings-banner", "children"),
    Input("ticker-input", "value")
)
def update_metadata(ticker):
    if not ticker or len(ticker.strip()) == 0:
        return [], None, "", ""
    try:
        stock = yf.Ticker(ticker)
        dates = stock.options
        info = stock.info
        company_name = info.get("longName", "")
        earnings_date = parse_earnings_date(stock)
        earnings_banner = ""

        if earnings_date:
            today = date.today()
            delta_days = (earnings_date - today).days

            if delta_days < 0:
                earnings_banner = f"📅 Last earnings was on {earnings_date}"
            elif delta_days <= 7:
                earnings_banner = f"🚨 Earnings in {delta_days} day(s)! 📅 {earnings_date}"
            else:
                earnings_banner = f"📅 Next earnings on {earnings_date}"

        if DEBUG_MODE:
            print(f"📈 Metadata for {ticker}: {company_name}, Earnings: {earnings_date}, Options: {dates}")
        return [{"label": d, "value": d} for d in dates], dates[0] if dates else None, company_name, earnings_banner
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Error in update_metadata: {e}")
        return [], None, "Unable to fetch company info", ""

# --- Fetch NewsAPI headlines ---
def fetch_news_headlines(ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=30&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        data = response.json()
        items = []
        if "articles" in data:
            for a in data["articles"]:
                date_str = a.get("publishedAt", "")[:10]
                title = a.get("title", "No Title")
                source = a.get("source", {}).get("name", "Unknown")
                url = a.get("url", "#")
                items.append(html.Li([
                    html.Strong(f"[{date_str}] "),
                    f"{title} — ",
                    html.Em(source),
                    html.A(" 🔗", href=url, target="_blank", style={"marginLeft": "6px"})
                ]))
        if DEBUG_MODE:
            print(f"📰 {len(items)} headlines fetched for {ticker}")
        return items or [html.Li("No headlines found.")]
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ News fetch error: {e}")
        return [html.Li("No headlines found or API limit reached.")]

# --- Detect unusual contracts ---
def detect_anomalies(df):
    df = df.copy()
    df["volume_oi_ratio"] = df["volume"] / (df["openInterest"] + 1)
    df["volume_oi_ratio"] = df["volume_oi_ratio"].replace([float("inf"), -float("inf")], 0).fillna(0)
    features = df[["volume", "openInterest", "impliedVolatility", "volume_oi_ratio"]].fillna(0)
    if len(features) < 10:
        df["anomaly"] = False
        return df
    clf = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly"] = clf.fit_predict(features) == -1
    if DEBUG_MODE:
        print(f"📊 Anomalies detected: {df['anomaly'].sum()}")
    return df

# --- Load FinBERT sentiment pipeline ---
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --- Analyze sentiment and attach emoji tags ---
def analyze_sentiment(headlines):
    if not headlines:
        if DEBUG_MODE:
            print("⚠️ No headlines passed to sentiment analyzer.")
        return 0, []
    try:
        if DEBUG_MODE:
            print(f"🧐 Analyzing headlines:\n{headlines}")
        results = sentiment_pipeline(headlines)
        if DEBUG_MODE:
            print(f"📈 FinBERT results:\n{results}")
        scores = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
        emoji = {'POSITIVE': "🟢", 'NEUTRAL': "🟡", 'NEGATIVE': "🔴"}

        labeled = []
        total = 0
        for hl, result in zip(headlines, results):
            label = result['label'].upper()
            score = scores.get(label, 0)
            total += score
            sentiment_tag = f"{emoji.get(label, '')} {label.title()}"
            labeled.append((hl, sentiment_tag))

        avg_score = total / len(results)
        return avg_score, labeled
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Sentiment analysis failed: {e}")
        return 0, [(hl, "🟡 Unknown") for hl in headlines]

# --- Dashboard main callback ---
@app.callback(
    Output("volume-chart", "figure"),
    Output("summary-text", "children"),
    Output("headline-list", "children"),
    Input("ticker-input", "value"),
    Input("expiry-dropdown", "value")
)
def update_dashboard(ticker, expiry):
    if not ticker or not expiry or len(ticker.strip()) == 0:
        return {}, "Please enter a valid stock ticker.", []

    stock = yf.Ticker(ticker)
    try:
        opt = stock.option_chain(expiry)
        df = pd.concat([opt.calls.assign(type="call"), opt.puts.assign(type="put")])
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Option chain fetch failed: {e}")
        return {}, "Failed to load option chain.", []

    df = detect_anomalies(df)
    fig = px.scatter(
        df,
        x="strike",
        y="volume",
        color="type",
        symbol="anomaly",
        size="volume_oi_ratio",
        hover_data=["openInterest", "impliedVolatility"],
        title=f"{ticker.upper()} Options Activity - {expiry}"
    )

    anomaly_count = df["anomaly"].sum()
    headline_items = fetch_news_headlines(ticker)
    headlines_only = []
    for item in headline_items:
        if isinstance(item, html.Li) and isinstance(item.children, list):
            headline = next((c for c in item.children if isinstance(c, str)), None)
            if headline:
                cleaned = re.sub(r'—.*$', '', headline).strip()
                headlines_only.append(cleaned)

    avg_sentiment, labeled_headlines = analyze_sentiment(headlines_only)
    sentiment_label = (
        "🟢 Bullish" if avg_sentiment >= 0.1 else
        "🟡 Neutral" if -0.1 < avg_sentiment < 0.1 else
        "🔴 Bearish"
    )
    sentiment_msg = f"FinBERT Sentiment: {sentiment_label} (avg score: {avg_sentiment:+.2f})"

    final_headline_items = []
    for i, (headline, sentiment_tag) in enumerate(labeled_headlines):
        if i >= len(headline_items): continue
        old = headline_items[i]
        new_li = html.Li([
            html.Span(f"{sentiment_tag}  ", style={"marginRight": "6px"}),
            *old.children
        ])
        final_headline_items.append(new_li)

    summary = f"{anomaly_count} unusual contracts detected by ML model. {sentiment_msg}"
    return fig, summary, final_headline_items

# --- Launch app ---
if __name__ == "__main__":
    app.run(debug=True)
