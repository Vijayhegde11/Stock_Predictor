from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert "predicted_next_day_close" in response.json()