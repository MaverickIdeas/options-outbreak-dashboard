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
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

app = dash.Dash(__name__)
app.title = "Options Outbreak Dashboard with News + ML"

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
        earnings_date = info.get("nextEarningsDate", None)
        earnings_banner = f"\u26a0\ufe0f Upcoming earnings on {earnings_date}" if earnings_date else ""
        return [{"label": d, "value": d} for d in dates], dates[0] if dates else None, company_name, earnings_banner
    except:
        return [], None, "Unable to fetch company info", ""

def fetch_news_headlines(ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=10&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        data = response.json()
        if "articles" in data:
            items = []
            for a in data["articles"]:
                date = a.get("publishedAt", "")[:10]
                title = a.get("title", "No Title")
                source = a.get("source", {}).get("name", "Unknown")
                url = a.get("url", "#")
                items.append(html.Li([
                    html.Strong(f"[{date}] "),
                    f"{title} — ",
                    html.Em(source),
                    html.A(" \ud83d\udd17", href=url, target="_blank", style={"marginLeft": "6px"})
                ]))
            return items
    except Exception as e:
        print(f"News error: {e}")
    return [html.Li("No headlines found or API limit reached.")]

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
    return df

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
    except:
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
    summary = f"{anomaly_count} unusual contracts detected by ML model."

    headline_items = fetch_news_headlines(ticker)

    return fig, summary, headline_items

sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(headlines):
    if not headlines:
        return 0
    results = sentiment_pipeline(headlines)
    scores = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    sentiment_score = sum(scores.get(r['label'], 0) for r in results) / len(results)
    return sentiment_score

if __name__ == "__main__":
    app.run(debug=True)
