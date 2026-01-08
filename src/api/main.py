from fastapi import FastAPI
from pydantic import BaseModel

from src.models.load_model import load_model
from src.features.build_features import build_features

app = FastAPI(title="Stock Price Predictor API", version="1.0")

model = load_model()

ticker_id_map = {
    "AAPL": 0,
    "MSFT": 1,
    "GOOG": 2,
    "TSLA": 3,
    "AMZN": 4,
    "NVDA": 5
}

class PredictionRequest(BaseModel):
    ticker: str

@app.post("/predict")
def predict(req: PredictionRequest):
    ticker = req.ticker.upper()

    features, err = build_features(ticker, ticker_id_map)
    if err:
        return {"error": err}
    
    prediction = model.predict(features)[0]

    return {
        "ticker": ticker,
        "predicted_next_day_close": float(prediction)
    }

