import joblib 
import os

def load_model():
    model_dir = "trained_models"

    versions = [
        f for f in os.listdir(model_dir)
        if f.startswith('stock_predictor_v') and f.endswith('.pkl')
    ]

    if not versions:
        raise FileNotFoundError("No trained model found")
    
    versions_sorted = sorted(
        versions,
        key=lambda x: int(x.split("_v")[1].split(".")[0])
    )

    latest_model = versions_sorted[-1]
    model_path = os.path.join(model_dir, latest_model)

    print(f"Loding Model: {model_path}")

    return joblib.load(model_path)

if __name__ == "__main__":
    load_model()

