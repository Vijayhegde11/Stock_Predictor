from src.features.build_features import build_features

ticker_id_map = {
    "AAPL": 0, "MSFT":1, "GOOG": 2, "TSLA": 3, "AMZN": 4, "NVDA": 5
}

def test_feature_builder():
    X, err = build_features("AAPL", ticker_id_map)
    assert err is None
    assert X is not None
    assert X.shape[0] == 1