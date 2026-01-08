import yfinance as yf 
import pandas as pd 
import numpy as np 
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import mlflow
import os

mlflow.set_tracking_uri("file:mlflow")
mlflow.set_experiment("data_preparation")

def prepare_data():
    all_data = []
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA"]

    for ticker in tickers:
        print("Fetching ->", ticker)

        df = yf.download(ticker, start="2015-01-01", end="2025-11-01")
        
        if df.empty:
            continue
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

        df["ticker"] = ticker

        df["target"] = close.shift(-1)

        df = df.dropna()

        all_data.append(df)

    data = pd.concat(all_data)
    data = data.reset_index()

    data["ticker_id"] = data["ticker"].astype("category").cat.codes

    data.to_csv("data/processed/multi_processed_stock.csv", index=False)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("tickers", tickers)
        mlflow.log_param("num_rows", data.shape[0])
        mlflow.log_param("num_features", data.shape[1])
        mlflow.log_param("data_start", str(data["Date"].min()))
        mlflow.log_param("data_end", str(data["Date"].max()))

        # Attaching the dataset file
        mlflow.log_artifact("data/processed/multi_processed_stock.csv")

    print("Dataset logged to MLflow!")

if __name__ == "__main__":
    prepare_data()
