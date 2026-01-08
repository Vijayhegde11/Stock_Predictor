import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os, joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

mlflow.set_tracking_uri("file:mlflow")
mlflow.set_experiment("stock_model_training")

def train_model():

    df = pd.read_csv("data/processed/multi_processed_stock.csv")

    feature_cols = [
        "close", "open", "high", "low", "volume",
        "returns", "sma_10", "sma_20", "ema_10", "rsi_14",
        "macd", "macd_signal", "macd_diff", "ticker_id"
    ]

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model_candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200),
        "XGBRegressor": XGBRegressor(n_estimators=300, learning_rate=0.05)
    }

    results = []

    for model_name, model in model_candidates.items():

        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)

            # Log model inside MLflow
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Store TRAINED model object
            results.append((model_name, mae, model))

    # Select best model based on MAE
    best_model_name, best_mae, best_model = sorted(results, key=lambda x: x[1])[0]

    print(f"Best model: {best_model_name} with MAE: {best_mae}")

    # Get best run ID for registry
    latest_run = mlflow.search_runs(order_by=["metrics.MAE ASC"]).iloc[0]
    run_id = latest_run.run_id

    mlflow.register_model(
        f"runs:/{run_id}/model",
        name="stock_predictor",
    )

    # Save model manually
    os.makedirs("trained_models", exist_ok=True)

    versions = [
        int(f.split("_v")[1].split(".")[0])
        for f in os.listdir("trained_models")
        if f.startswith("stock_predictor_v")
    ]

    next_version = max(versions) + 1 if versions else 1

    model_path = f"trained_models/stock_predictor_v{next_version}.pkl"
    joblib.dump(best_model, model_path)

    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    train_model()
