📈 Options Outbreak Dashboard with News + Machine Learning
A powerful, interactive dashboard built with Dash (Plotly) for options traders. It identifies anomalous volume/open interest activity using Isolation Forest ML, displays relevant news headlines with sentiment analysis using FinBERT, and provides a buy/sell prediction using a Random Forest Classifier.

🚀 Features

🟢 Buy/Sell Prediction with Confidence:

Uses a Random Forest Classifier to predict buy/sell signals based on RSI, MACD, SMA, EMA, FinBERT sentiment, and call/put volume ratio.
Displays a prediction flag with confidence score and estimated holding period (e.g., "Buy (Confidence: 85%, Estimated Hold: 3 days)").
Color-coded for quick interpretation (green for buy, red for sell).


📊 Technical Indicators Cluster:

Displays SMA (20-day, 50-day), EMA (20-day, 50-day), MACD, and RSI with bullish/bearish interpretations.
Sourced from Polygon.io for accurate, real-time data.
Highlights oversold conditions (potential buy) and overbought conditions (potential sell).


📈 Candlestick Chart:

Visualizes 1-minute candlestick data for the current or last trading day (9:30 AM to 4:00 PM ET).
Includes pattern detection (e.g., Doji, Hammer) with TA-Lib integration.
Sourced from Polygon.io for high-resolution data.


📊 Options Chain Visualization:

Displays calls and puts with volume, open interest, and implied volatility.
Identifies anomalies in volume/open interest using Isolation Forest ML.
Calculates call/put volume ratio to gauge consumer interest (bullish/bearish sentiment).


🤖 ML-Based Anomaly Detection:

Uses Isolation Forest to detect unusual contracts based on volume, open interest, and implied volatility.
Highlights anomalies directly on the options chart.


📰 News Headline Scraper:

Fetches recent news headlines related to the ticker via NewsAPI.
Displays headlines with sentiment tags (Positive, Neutral, Negative) using FinBERT.


💬 FinBERT Sentiment Analysis:

Analyzes news headlines to compute an average sentiment score.
Visualizes daily sentiment trends in a line chart.
Integrates sentiment into the buy/sell prediction model.


📅 Earnings Date Warning System:

Alerts users of upcoming earnings dates within 7 days.
Displays past earnings dates if already occurred.




📦 Setup Instructions

Clone the Repo
git clone https://github.com/yourusername/options-outbreak-dashboard.git
cd options-outbreak-dashboard


Set Up a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies
pip install -r requirements.txt


Install TA-Lib for Candlestick Pattern Detection

On Linux:wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..
pip install TA-Lib


On Windows or other systems, refer to TA-Lib installation instructions.


Set Up API Keys

Create a .env file in the project root:touch .env


Add your API keys (obtain from NewsAPI and Polygon.io):NEWSAPI_KEY=your_newsapi_key
POLYGON_API_KEY=your_polygon_api_key
DEBUG_MODE=true


Alternatively, run the setup script to configure the .env file:bash setup.sh




Run the Dashboard
python app.py


Open your browser and navigate to http://127.0.0.1:8050.




🛠️ Requirements

Python 3.8+
Dependencies listed in requirements.txt:
dash==2.14.2
yfinance
pandas
plotly
scikit-learn
transformers
requests
python-dotenv
torch
numpy==1.26.4
pandas-ta==0.3.14b0
pytz
TA-Lib (for candlestick pattern detection)




📚 Usage

Enter a Stock Ticker:

Input a ticker symbol (e.g., TSLA, AAPL) in the text box at the top.
The dashboard will update with data for the specified ticker.


Interpret the Prediction:

View the buy/sell prediction at the top (e.g., "Buy (Confidence: 85%, Estimated Hold: 3 days)").
Use the confidence score and holding period to inform trading decisions.


Analyze Technical Indicators:

Check the SMA, EMA, MACD, and RSI values for trend signals.
Look for oversold (RSI < 30) or overbought (RSI > 70) conditions.


View Candlestick Chart:

Examine 1-minute candlestick patterns with annotations for Doji, Hammer, etc.
Zoom in/out to analyze price movements throughout the trading day.


Explore Options Data:

See call/put options with anomaly markers for unusual volume/open interest.
The call/put volume ratio influences the buy/sell prediction.


Read News and Sentiment:

Browse recent news headlines with sentiment tags.
Check the sentiment chart for daily trends, which feed into the prediction model.




🐞 Debugging

Enable Debug Mode:

Set DEBUG_MODE=true in .env to see detailed logs in the console.
Logs include API fetch errors, ML model training details, and data processing steps.


Common Issues:

API Key Errors: Ensure NEWSAPI_KEY and POLYGON_API_KEY are valid and have access to required endpoints.
TA-Lib Installation: If candlestick patterns fail, verify TA-Lib is installed correctly.
Prediction Unavailable: If historical data is insufficient, the buy/sell prediction will display an error message.




🌟 Future Enhancements

Integrate historical sentiment and options data for more accurate ML predictions.
Add more technical indicators (e.g., Bollinger Bands, Stochastic Oscillator).
Implement user-defined parameters for the prediction model (e.g., adjust holding period estimation).
Enhance visualization with interactive tooltips for indicators and predictions.


📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request with your enhancements or bug fixes.
