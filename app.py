# File: app.py
# Main application file for Options Outbreak Dashboard

import yfinance as yf
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import os
import re
from dotenv import load_dotenv
from datetime import datetime, date, timedelta, time
import pandas_ta as ta
import pytz
import numpy as np

# --- Load .env and API Keys ---
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY not found in .env file")

# --- Initialize Dash App ---
app = dash.Dash(__name__)
app.title = "Options Outbreak Dashboard with News + ML"

# --- Define App Layout ---
app.layout = html.Div([
    html.H1("Options Outbreak Dashboard", style={"textAlign": "center"}),
    dcc.Input(id="ticker-input", type="text", placeholder="Enter Stock Ticker", value="TSLA"),
    html.Div(id="company-name", style={"textAlign": "center", "fontWeight": "bold", "marginBottom": "10px"}),
    # New Prediction Flag
    html.Div(id="prediction-flag", style={"textAlign": "center", "fontWeight": "bold", "marginBottom": "10px"}),
    # Technical Indicators Cluster
    html.Div(id="technical-indicators", style={"textAlign": "center", "marginBottom": "20px"}),
    dcc.Graph(id="candlestick-chart"),
    dcc.Dropdown(id="expiry-dropdown", placeholder="Select expiry date"),
    html.Div(id="earnings-banner", style={"color": "red", "textAlign": "center", "marginBottom": "10px"}),
    dcc.Graph(id="volume-chart"),
    html.Div(id="summary-text", style={"padding": "10px"}),
    dcc.Graph(id="sentiment-chart"),
    html.H3("Related Headlines:", style={"marginTop": "20px"}),
    html.Ul(id="headline-list")
])

# --- Check if market is open ---
def is_market_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_weekday = now.weekday() < 5
    is_trading_hours = market_open <= now <= market_close
    return is_weekday and is_trading_hours

# --- Get candlestick date ---
def get_candlestick_date():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    today = now.date()
    if now.weekday() < 5:  # Weekday
        if now.time() < time(9, 30):
            # Before market open, use previous trading day
            if now.weekday() == 0:  # Monday
                return today - timedelta(days=3)  # Friday
            else:
                return today - timedelta(days=1)
        else:
            # During or after trading hours, use today
            return today
    else:  # Weekend
        # Use previous Friday
        days_back = (now.weekday() - 4) % 7
        return today - timedelta(days=days_back)

# --- Fetch historical data for ML training ---
def fetch_historical_data_for_ml(ticker, days=60):
    eastern = pytz.timezone("US/Eastern")
    end_date = get_candlestick_date()
    start_date = end_date - timedelta(days=days)
    start_ts = int(datetime.combine(start_date, time(0, 0), tzinfo=eastern).astimezone(pytz.utc).timestamp() * 1000)
    end_ts = int(datetime.combine(end_date, time(23, 59), tzinfo=eastern).astimezone(pytz.utc).timestamp() * 1000)

    # Fetch daily OHLC data
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_ts}/{end_ts}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url).json()
    if "results" not in response or not response["results"]:
        return None

    df = pd.DataFrame(response["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "Date"}, inplace=True)
    df.set_index("Date", inplace=True)

    # Fetch indicators
    base_params = {
        "timespan": "day",
        "adjusted": "true",
        "series_type": "close",
        "order": "desc",
        "limit": days,
        "apiKey": POLYGON_API_KEY
    }

    # SMA
    sma_20_params = base_params.copy()
    sma_20_params["window"] = 20
    sma_20_response = requests.get(f"https://api.polygon.io/v1/indicators/sma/{ticker}", params=sma_20_params).json()
    sma_50_params = base_params.copy()
    sma_50_params["window"] = 50
    sma_50_response = requests.get(f"https://api.polygon.io/v1/indicators/sma/{ticker}", params=sma_50_params).json()

    # EMA
    ema_20_params = base_params.copy()
    ema_20_params["window"] = 20
    ema_20_response = requests.get(f"https://api.polygon.io/v1/indicators/ema/{ticker}", params=ema_20_params).json()
    ema_50_params = base_params.copy()
    ema_50_params["window"] = 50
    ema_50_response = requests.get(f"https://api.polygon.io/v1/indicators/ema/{ticker}", params=ema_50_params).json()

    # MACD
    macd_params = base_params.copy()
    macd_params.update({"short_window": 12, "long_window": 26, "signal_window": 9})
    macd_response = requests.get(f"https://api.polygon.io/v1/indicators/macd/{ticker}", params=macd_params).json()

    # RSI
    rsi_params = base_params.copy()
    rsi_params["window"] = 14
    rsi_response = requests.get(f"https://api.polygon.io/v1/indicators/rsi/{ticker}", params=rsi_params).json()

    # Process indicators into DataFrames
    indicators = {}
    for ind, response in [("sma_20", sma_20_response), ("sma_50", sma_50_response),
                          ("ema_20", ema_20_response), ("ema_50", ema_50_response),
                          ("macd", macd_response), ("rsi", rsi_response)]:
        if "results" in response and "values" in response["results"]:
            ind_df = pd.DataFrame(response["results"]["values"])
            ind_df["timestamp"] = pd.to_datetime(ind_df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
            ind_df.set_index("timestamp", inplace=True)
            indicators[ind] = ind_df

    # Combine data
    df["SMA_20"] = indicators["sma_20"]["value"] if "sma_20" in indicators else np.nan
    df["SMA_50"] = indicators["sma_50"]["value"] if "sma_50" in indicators else np.nan
    df["EMA_20"] = indicators["ema_20"]["value"] if "ema_20" in indicators else np.nan
    df["EMA_50"] = indicators["ema_50"]["value"] if "ema_50" in indicators else np.nan
    df["MACD"] = indicators["macd"]["value"] if "macd" in indicators else np.nan
    df["MACD_Signal"] = indicators["macd"]["signal"] if "macd" in indicators else np.nan
    df["RSI"] = indicators["rsi"]["value"] if "rsi" in indicators else np.nan

    # Drop NaN rows
    df = df.dropna()
    return df

# --- Predict Buy/Sell Flag with Confidence and Holding Period ---
def predict_buy_sell(ticker, rsi, macd, macd_signal, sma_20, sma_50, ema_20, ema_50, finbert_score, call_put_ratio):
    # Fetch historical data for training
    historical_data = fetch_historical_data_for_ml(ticker, days=60)
    if historical_data is None or len(historical_data) < 20:
        return "Prediction unavailable: Insufficient historical data", "gray", "N/A"

    # Prepare features
    historical_data["SMA_Diff"] = historical_data["SMA_20"] - historical_data["SMA_50"]
    historical_data["EMA_Diff"] = historical_data["EMA_20"] - historical_data["EMA_50"]
    historical_data["MACD_Diff"] = historical_data["MACD"] - historical_data["MACD_Signal"]
    
    # Simulate FinBERT sentiment (historical sentiment not available, use random for training)
    np.random.seed(42)
    historical_data["FinBERT"] = np.random.uniform(-1, 1, size=len(historical_data))
    
    # Simulate call/put ratio (historical options data not directly available, use random for training)
    historical_data["Call_Put_Ratio"] = np.random.uniform(0.5, 2, size=len(historical_data))

    # Target: 1 if next day's close is higher, 0 if lower
    historical_data["Target"] = (historical_data["Close"].shift(-1) > historical_data["Close"]).astype(int)
    historical_data = historical_data[:-1]  # Remove last row since no next day data

    # Features for training
    features = ["RSI", "MACD_Diff", "SMA_Diff", "EMA_Diff", "FinBERT", "Call_Put_Ratio"]
    X = historical_data[features]
    y = historical_data["Target"]

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Prepare latest data for prediction
    latest_data = pd.DataFrame({
        "RSI": [rsi],
        "MACD_Diff": [macd - macd_signal if macd and macd_signal else 0],
        "SMA_Diff": [sma_20 - sma_50 if sma_20 and sma_50 else 0],
        "EMA_Diff": [ema_20 - ema_50 if ema_20 and ema_50 else 0],
        "FinBERT": [finbert_score],
        "Call_Put_Ratio": [call_put_ratio]
    })
    latest_data_scaled = scaler.transform(latest_data)

    # Predict
    prediction = model.predict(latest_data_scaled)[0]
    confidence = model.predict_proba(latest_data_scaled)[0][prediction] * 100

    # Estimate holding period based on historical consecutive signals
    historical_predictions = model.predict(scaler.transform(X))
    runs = []
    current_run = 1
    current_label = historical_predictions[0]
    for pred in historical_predictions[1:]:
        if pred == current_label:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
            current_label = pred
    runs.append(current_run)
    avg_holding_period = int(np.mean(runs)) if runs else 1

    # Format prediction
    flag = "Buy" if prediction == 1 else "Sell"
    color = "green" if flag == "Buy" else "red"
    prediction_text = f"{flag} (Confidence: {confidence:.1f}%, Estimated Hold: {avg_holding_period} days)"

    return prediction_text, color, avg_holding_period

# --- Fetch Technical Indicators from Polygon.io ---
def fetch_technical_indicators(ticker):
    eastern = pytz.timezone("US/Eastern")
    candlestick_date = get_candlestick_date()
    timestamp = candlestick_date.strftime("%Y-%m-%d")
    
    # Base parameters for API calls
    base_params = {
        "timestamp": timestamp,
        "timespan": "day",
        "adjusted": "true",
        "series_type": "close",
        "order": "desc",
        "limit": 1,
        "apiKey": POLYGON_API_KEY
    }

    indicators = {}

    # Fetch SMA (20-day and 50-day)
    try:
        sma_20_params = base_params.copy()
        sma_20_params["window"] = 20
        sma_20_response = requests.get(f"https://api.polygon.io/v1/indicators/sma/{ticker}", params=sma_20_params).json()
        sma_50_params = base_params.copy()
        sma_50_params["window"] = 50
        sma_50_response = requests.get(f"https://api.polygon.io/v1/indicators/sma/{ticker}", params=sma_50_params).json()

        sma_20 = sma_20_response["results"]["values"][0]["value"] if "results" in sma_20_response and sma_20_response["results"]["values"] else None
        sma_50 = sma_50_response["results"]["values"][0]["value"] if "results" in sma_50_response and sma_50_response["results"]["values"] else None
        sma_trend = "Bullish (bottoming out)" if sma_20 and sma_50 and sma_20 > sma_50 else "Bearish (outbreak)" if sma_20 and sma_50 else "Unknown"
        sma_color = "green" if sma_trend.startswith("Bullish") else "red" if sma_trend.startswith("Bearish") else "gray"
        indicators["SMA"] = {"value": f"SMA 20: {sma_20:.2f}, 50: {sma_50:.2f} ({sma_trend})", "color": sma_color, "raw": {"sma_20": sma_20, "sma_50": sma_50}}
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Error fetching SMA: {e}")
        indicators["SMA"] = {"value": "SMA: Error fetching data", "color": "gray", "raw": {"sma_20": None, "sma_50": None}}

    # Fetch EMA (20-day and 50-day)
    try:
        ema_20_params = base_params.copy()
        ema_20_params["window"] = 20
        ema_20_response = requests.get(f"https://api.polygon.io/v1/indicators/ema/{ticker}", params=ema_20_params).json()
        ema_50_params = base_params.copy()
        ema_50_params["window"] = 50
        ema_50_response = requests.get(f"https://api.polygon.io/v1/indicators/ema/{ticker}", params=ema_50_params).json()

        ema_20 = ema_20_response["results"]["values"][0]["value"] if "results" in ema_20_response and ema_20_response["results"]["values"] else None
        ema_50 = ema_50_response["results"]["values"][0]["value"] if "results" in ema_50_response and ema_50_response["results"]["values"] else None
        ema_trend = "Bullish (bottoming out)" if ema_20 and ema_50 and ema_20 > ema_50 else "Bearish (outbreak)" if ema_20 and ema_50 else "Unknown"
        ema_color = "green" if ema_trend.startswith("Bullish") else "red" if ema_trend.startswith("Bearish") else "gray"
        indicators["EMA"] = {"value": f"EMA 20: {ema_20:.2f}, 50: {ema_50:.2f} ({ema_trend})", "color": ema_color, "raw": {"ema_20": ema_20, "ema_50": ema_50}}
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Error fetching EMA: {e}")
        indicators["EMA"] = {"value": "EMA: Error fetching data", "color": "gray", "raw": {"ema_20": None, "ema_50": None}}

    # Fetch MACD
    try:
        macd_params = base_params.copy()
        macd_params.update({"short_window": 12, "long_window": 26, "signal_window": 9})
        macd_response = requests.get(f"https://api.polygon.io/v1/indicators/macd/{ticker}", params=macd_params).json()
        macd_data = macd_response["results"]["values"][0] if "results" in macd_response and macd_response["results"]["values"] else {}
        macd_value = macd_data.get("value")
        signal_value = macd_data.get("signal")
        macd_trend = "Bullish (bottoming out)" if macd_value and signal_value and macd_value > signal_value else "Bearish (outbreak)" if macd_value and signal_value else "Unknown"
        macd_color = "green" if macd_trend.startswith("Bullish") else "red" if macd_trend.startswith("Bearish") else "gray"
        indicators["MACD"] = {"value": f"MACD: {macd_value:.2f}, Signal: {signal_value:.2f} ({macd_trend})", "color": macd_color, "raw": {"macd": macd_value, "signal": signal_value}}
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Error fetching MACD: {e}")
        indicators["MACD"] = {"value": "MACD: Error fetching data", "color": "gray", "raw": {"macd": None, "signal": None}}

    # Fetch RSI
    try:
        rsi_params = base_params.copy()
        rsi_params["window"] = 14
        rsi_response = requests.get(f"https://api.polygon.io/v1/indicators/rsi/{ticker}", params=rsi_params).json()
        rsi_value = rsi_response["results"]["values"][0]["value"] if "results" in rsi_response and rsi_response["results"]["values"] else None
        rsi_status = "Oversold (buy)" if rsi_value and rsi_value < 30 else "Overbought (sell)" if rsi_value and rsi_value > 70 else "Neutral"
        rsi_color = "green" if rsi_status.startswith("Oversold") else "red" if rsi_status.startswith("Overbought") else "gray"
        indicators["RSI"] = {"value": f"RSI: {rsi_value:.2f} ({rsi_status})", "color": rsi_color, "raw": rsi_value}
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Error fetching RSI: {e}")
        indicators["RSI"] = {"value": "RSI: Error fetching data", "color": "gray", "raw": None}

    return indicators

# --- Create candlestick chart with pattern detection using Polygon.io ---
def create_candlestick_chart(ticker):
    try:
        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)
        if is_market_open():
            start_time = eastern.localize(datetime.combine(now.date(), time(9, 30)))
            end_time = now
            title = f"{ticker.upper()} Intraday Candlestick Chart (Today)"
        else:
            candlestick_date = get_candlestick_date()
            start_time = eastern.localize(datetime.combine(candlestick_date, time(9, 30)))
            end_time = eastern.localize(datetime.combine(candlestick_date, time(16, 0)))
            title = f"{ticker.upper()} Candlestick Chart (Last Trading Day: {candlestick_date})"

        # Convert to UTC timestamps in milliseconds
        start_ts = int(start_time.astimezone(pytz.utc).timestamp() * 1000)
        end_ts = int(end_time.astimezone(pytz.utc).timestamp() * 1000)

        # Fetch 1-minute candlestick data from Polygon.io
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_ts}/{end_ts}?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "results" not in data or not data["results"]:
            return {
                "data": [],
                "layout": {
                    "title": title,
                    "annotations": [{
                        "text": "No price data available from Polygon.io",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 20}
                    }]
                }
            }

        # Process Polygon.io data into a DataFrame
        candles = data["results"]
        df = pd.DataFrame(candles)
        df["t"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "t": "Date"
        }, inplace=True)
        df.set_index("Date", inplace=True)

        # Prepare DataFrame for pandas_ta
        df_ta = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df_ta.columns = ["open", "high", "low", "close", "volume"]
        if DEBUG_MODE:
            print(f"📊 Candlestick DataFrame columns: {df_ta.columns.tolist()}")
            print(f"📊 DataFrame shape: {df_ta.shape}")

        # Validate DataFrame for pattern detection
        pattern_names = ["doji", "hammer", "engulfing", "morningstar", "eveningstar"]
        if len(df_ta) < 5:
            if DEBUG_MODE:
                print("⚠️ Insufficient data for pattern detection (less than 5 rows)")
            patterns = pd.DataFrame(index=df_ta.index, columns=[f"CDL_{name.upper()}" for name in pattern_names]).fillna(0)
        elif df_ta[["open", "high", "low", "close"]].isnull().any().any():
            if DEBUG_MODE:
                print("⚠️ DataFrame contains NaN values, skipping pattern detection")
            patterns = pd.DataFrame(index=df_ta.index, columns=[f"CDL_{name.upper()}" for name in pattern_names]).fillna(0)
        else:
            # Detect candlestick patterns using ta.cdl_pattern
            try:
                patterns = df_ta.ta.cdl_pattern(name=pattern_names)
                if DEBUG_MODE:
                    print(f"📊 Detected patterns:\n{patterns.head()}")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"❌ Error detecting candlestick patterns: {e}")
                patterns = pd.DataFrame(index=df_ta.index, columns=[f"CDL_{name.upper()}" for name in pattern_names]).fillna(0)

        # Filter detected patterns
        detected_patterns = patterns[patterns != 0].dropna(how="all")

        # Create candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC"
            )
        ])

        # Add pattern annotations
        annotations = []
        for date, row in detected_patterns.iterrows():
            for pattern, value in row.items():
                if pd.notnull(value) and value != 0:
                    pattern_name = pattern.replace("CDL_", "").title()
                    direction = "Bullish" if value > 0 else "Bearish"
                    annotations.append({
                        "x": date,
                        "y": df.loc[date]["High"] if value > 0 else df.loc[date]["Low"],
                        "text": f"{pattern_name} ({direction})",
                        "showarrow": True,
                        "arrowhead": 2,
                        "ax": 0,
                        "ay": -30 if value > 0 else 30
                    })

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            xaxis_range=[start_time, end_time],
            annotations=annotations
        )

        return fig
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Candlestick chart error: {e}")
        return {
            "data": [],
            "layout": {
                "title": f"{ticker.upper()} Candlestick Chart",
                "annotations": [{
                    "text": f"Failed to load price data: {str(e)}",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 20}
                }]
            }
        }

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
        raw_articles = []
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
                raw_articles.append({
                    "title": title,
                    "date": date_str,
                    "publishedAt": a.get("publishedAt")
                })
        if DEBUG_MODE:
            print(f"📰 {len(items)} headlines fetched for {ticker}")
        return items or [html.Li("No headlines found.")], raw_articles
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ News fetch error: {e}")
        return [html.Li("No headlines found or API limit reached.")], []

# --- Detect unusual contracts ---
def detect_anomalies(df):
    df = df.copy()
    # Convert relevant columns to numeric
    columns = ["volume", "openInterest", "impliedVolatility"]
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Calculate volume_oi_ratio
    df["volume_oi_ratio"] = df["volume"] / (df["openInterest"] + 1)
    # Replace inf and fill NaN
    df["volume_oi_ratio"] = df["volume_oi_ratio"].replace([float("inf"), -float("inf")], 0).fillna(0)
    # Select features and fill NaN with 0
    features = df[columns + ["volume_oi_ratio"]].fillna(0)
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
def analyze_sentiment(headlines, raw_articles):
    if not headlines:
        if DEBUG_MODE:
            print("⚠️ No headlines passed to sentiment analyzer.")
        return 0, [], pd.DataFrame()
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
        sentiment_data = []
        for hl, result, article in zip(headlines, results, raw_articles):
            label = result['label'].upper()
            score = scores.get(label, 0)
            total += score
            sentiment_tag = f"{emoji.get(label, '')} {label.title()}"
            labeled.append((hl, sentiment_tag))
            sentiment_data.append({
                "date": article["date"],
                "score": score,
                "publishedAt": article["publishedAt"]
            })

        avg_score = total / len(results)
        # Create DataFrame for daily sentiment
        sentiment_df = pd.DataFrame(sentiment_data)
        if not sentiment_df.empty:
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
            daily_sentiment = sentiment_df.groupby("date")["score"].mean().reset_index()
        else:
            daily_sentiment = pd.DataFrame(columns=["date", "score"])
        return avg_score, labeled, daily_sentiment
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Sentiment analysis failed: {e}")
        return 0, [(hl, "🟡 Unknown") for hl in headlines], pd.DataFrame(columns=["date", "score"])

# --- Create daily sentiment plot ---
def create_sentiment_plot(sentiment_df, ticker):
    if sentiment_df.empty:
        return {
            "data": [],
            "layout": {
                "title": f"{ticker.upper()} Daily News Sentiment",
                "annotations": [{
                    "text": "No sentiment data available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 20}
                }]
            }
        }
    fig = px.line(
        sentiment_df,
        x="date",
        y="score",
        title=f"{ticker.upper()} Daily News Sentiment",
        labels={"date": "Date", "score": "Average Sentiment Score"}
    )
    fig.update_traces(line_color="#1f77b4")
    fig.update_layout(
        yaxis_range=[-1, 1],
        yaxis_tickvals=[-1, 0, 1],
        yaxis_ticktext=["Bearish", "Neutral", "Bullish"]
    )
    return fig

# --- Dashboard main callback ---
@app.callback(
    Output("prediction-flag", "children"),  # New output for prediction flag
    Output("technical-indicators", "children"),
    Output("candlestick-chart", "figure"),
    Output("volume-chart", "figure"),
    Output("summary-text", "children"),
    Output("headline-list", "children"),
    Output("sentiment-chart", "figure"),
    Input("ticker-input", "value"),
    Input("expiry-dropdown", "value")
)
def update_dashboard(ticker, expiry):
    if not ticker or not expiry or len(ticker.strip()) == 0:
        return "", "", {}, {}, "Please enter a valid stock ticker.", [], {}

    stock = yf.Ticker(ticker)

    # Fetch technical indicators
    indicators = fetch_technical_indicators(ticker)
    indicator_elements = [
        html.Span(indicators[ind]["value"], style={"color": indicators[ind]["color"], "marginRight": "20px"})
        for ind in ["SMA", "EMA", "MACD", "RSI"]
    ]

    # Fetch candlestick chart
    candlestick_fig = create_candlestick_chart(ticker)

    # Fetch options data and calculate call/put volume ratio
    try:
        opt = stock.option_chain(expiry)
        df = pd.concat([opt.calls.assign(type="call"), opt.puts.assign(type="put")])
        call_volume = df[df["type"] == "call"]["volume"].sum()
        put_volume = df[df["type"] == "put"]["volume"].sum()
        call_put_ratio = call_volume / (put_volume + 1)  # Avoid division by zero
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Option chain fetch failed: {e}")
        call_put_ratio = 1.0  # Neutral default
        df = pd.DataFrame()

    # Detect anomalies if options data is available
    anomaly_count = 0
    volume_fig = {}
    if not df.empty:
        df = detect_anomalies(df)
        volume_fig = px.scatter(
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
    else:
        volume_fig = {}
        anomaly_count = 0

    # Fetch news headlines and sentiment
    headline_items, raw_articles = fetch_news_headlines(ticker)
    headlines_only = []
    for item in headline_items:
        if isinstance(item, html.Li) and isinstance(item.children, list):
            headline = next((c for c in item.children if isinstance(c, str)), None)
            if headline:
                cleaned = re.sub(r'—.*$', '', headline).strip()
                headlines_only.append(cleaned)

    avg_sentiment, labeled_headlines, daily_sentiment = analyze_sentiment(headlines_only, raw_articles)
    sentiment_fig = create_sentiment_plot(daily_sentiment, ticker)
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

    # Predict buy/sell flag
    rsi = indicators["RSI"]["raw"]
    macd = indicators["MACD"]["raw"]["macd"]
    macd_signal = indicators["MACD"]["raw"]["signal"]
    sma_20 = indicators["SMA"]["raw"]["sma_20"]
    sma_50 = indicators["SMA"]["raw"]["sma_50"]
    ema_20 = indicators["EMA"]["raw"]["ema_20"]
    ema_50 = indicators["EMA"]["raw"]["ema_50"]
    prediction_text, prediction_color, holding_period = predict_buy_sell(
        ticker, rsi, macd, macd_signal, sma_20, sma_50, ema_20, ema_50, avg_sentiment, call_put_ratio
    )
    prediction_element = html.Span(prediction_text, style={"color": prediction_color})

    summary = f"{anomaly_count} unusual contracts detected by ML model. {sentiment_msg}"
    return prediction_element, indicator_elements, candlestick_fig, volume_fig, summary, final_headline_items, sentiment_fig

# --- Launch app ---
if __name__ == "__main__":
    app.run(debug=True)