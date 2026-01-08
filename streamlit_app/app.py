import streamlit as st
import requests, os
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("Multi-Stock Price Predictor (MLOps Project)")
st.markdown("Enter any stock ticker (AAPL, MSFT, GOOG, TSLA, AMZN, NVDA) and get next-day prediction. ")

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000") + "/predict"
st.write(f"Using API: {API_URL}")
# Tricker imput
ticker = st.text_input("Enter Stock Ticker", value='AAPL').upper()

# Fetch Recent Chart 
def show_chart(ticket):
    df = yf.download(ticker, period="3mo", interval="1d")
    if not df.empty:
        st.line_chart(df["Close"], height=250)
    else:
        st.error("Could not the stock data")

# Prediction button

if st.button("Predict"):
    if ticker not in ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA"]:
        st.error("Ticker not supported. Please enter a valid ticker")
    else:
        with st.spinner("Predicting..."):
            try:
                response = requests.post(API_URL, json={"ticker": ticker})
                data = response.json()

                if "error" in data:
                    st.error(f"{data['error']}")
                else:
                    predicted_price = data["predicted_next_day_close"]

                    st.success(f"Predicted next-day price for **{ticker}**:")
                    st.metric(label="Predicted Price", value=f"${predicted_price:.2f}")

                    st.subheader("Recent Price Chart")
                    show_chart(ticker)

            except Exception as e:
                st.error(f"Error connecting to API: {e}")