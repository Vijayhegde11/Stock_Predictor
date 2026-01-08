from src.models.load_model import load_model
import numpy as np

def test_model_load_and_predict():
    model = load_model()
    assert model is not None