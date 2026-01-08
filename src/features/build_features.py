import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

def build_features(ticker: str, ticker_id_map: dict):

    df = yf.download(ticker, period="3mo", interval="1d")
    if df.empty:
        return None, f"No data found for {ticker}"
    
    df.columns = ['_'.join(col).strip() for col in df.columns]
    df = df.rename(columns= {
            f"Close_{ticker}": "close",
            f"Open_{ticker}": "open",
            f"High_{ticker}": "high",
            f"Low_{ticker}": "low",
            f"Volume_{ticker}": "volume"
        })
    close = df["close"]

    df["returns"] = close.pct_change()

    # Technical Indicators
    df["sma_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["sma_20"] = SMAIndicator(close=close, window=20).sma_indicator()

    df["ema_10"] = EMAIndicator(close=close, window=10).ema_indicator()

    rsi = RSIIndicator(close=close, window=14)
    df["rsi_14"] = rsi.rsi()

    macd = MACD(close=close)
    df["macd"]= macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["ticker_id"] = ticker_id_map.get(ticker.upper(), -1)

    df = df.dropna()

    if df.empty:
        return None, "Not enough data for indicators."
    
    # Select last row for prediction
    feature_cols = ["close", "open", "high", "low", "volume",
                    "returns", "sma_10", "sma_20", "ema_10", 
                    "rsi_14", "macd", "macd_signal", "macd_diff", "ticker_id"]
    
    return df.iloc[-1:][feature_cols], None