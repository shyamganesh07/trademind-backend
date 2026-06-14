import yfinance as yf
import requests

print("--- Testing yfinance library ---")
try:
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="3mo")
    if not df.empty:
        print("Success fetching AAPL!")
        print("Latest date:", df.index[-1])
        print("Latest Close price:", df['Close'].iloc[-1])
        print("Data shape:", df.shape)
    else:
        print("Empty DataFrame returned for AAPL.")
except Exception as e:
    print("yfinance failed with exception:", e)

print("\n--- Testing basic requests to Yahoo Finance API ---")
try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get("https://query2.finance.yahoo.com/v8/finance/chart/AAPL?range=3mo&interval=1d", headers=headers, timeout=10)
    print("HTTP Status Code:", r.status_code)
    data = r.json()
    meta = data.get('chart', {}).get('result', [{}])[0].get('meta', {})
    print("Meta Symbol:", meta.get('symbol'))
    print("Meta Regular Market Price:", meta.get('regularMarketPrice'))
    indicators = data.get('chart', {}).get('result', [{}])[0].get('indicators', {}).get('quote', [{}])[0]
    close_prices = indicators.get('close', [])
    if close_prices:
        non_null = [p for p in close_prices if p is not None]
        if non_null:
            print("Latest Close in raw JSON:", non_null[-1])
except Exception as e:
    print("Requests test failed with exception:", e)
