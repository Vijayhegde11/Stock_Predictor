Stock Predictor MLOps System

An end-to-end MLOps project that demonstrates how to build, track, containerize, and deploy a machine learning system for multi-stock price prediction using MLflow, FastAPI, Streamlit, Docker, and CI pipelines.

This project covers the full ML lifecycle:
data ingestion → feature engineering → model training → experiment tracking → model export → inference service → UI → CI automation.

--------------------------------------------------------------------------------------------------------------------------------------------------

Problem Statement

Build a scalable machine learning system that:

Fetches real stock market data

Performs feature engineering

Trains and compares multiple models

Tracks experiments with MLflow

Serves predictions through an API

Provides a user-friendly web interface

Is fully containerized and CI-enabled

--------------------------------------------------------------------------------------------------------------------------------------------------

System Architecture

              ┌────────────────────┐
              │   Yahoo Finance     │
              │   (Data Source)     │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Feature Engineering │
              │ (build_features.py)│
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Model Training      │
              │ (train_model.py)   │
              │ + MLflow Tracking  │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Saved Models (.pkl) │
              └─────────┬──────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│   FastAPI API    │◄────────►│  Streamlit UI    │
│   (Inference)    │          │  (Visualization) │
└──────────────────┘          └──────────────────┘
        ▲
        │
   Docker + CI Pipeline

-----------------------------------------------------------------------------------------------------------------------------------------------

MLOps Workflow

1. Data & Feature Pipeline

-> Fetches stock data using yfinance

-> Builds technical indicators (SMA, EMA, RSI, MACD, returns)

-> Encodes multi-stock information

2. Model Training

-> Trains multiple models (Linear, Random Forest, XGBoost)

-> Logs parameters & metrics to MLflow

-> Selects best model based on evaluation metric

-> Saves final model to trained_models/

3. Experiment Tracking (MLflow)

-> Tracks:

-> model parameters

-> evaluation metrics

-> artifacts

-> Enables reproducibility and comparison

4. Inference Service (FastAPI)

-> Loads trained model

-> Builds features in real time

-> Exposes /predict endpoint

-> Returns next-day stock price prediction

5. User Interface (Streamlit)

-> User selects a stock ticker

-> Calls FastAPI service

-> Displays predicted price

-> Shows recent stock price chart

6. Containerization & Orchestration

-> FastAPI and Streamlit run as separate containers

-> Managed via Docker Compose

7. CI Pipeline (GitHub Actions)

-> Runs on every push

-> Installs dependencies

-> Builds both Docker images

-> Ensures system is always deployable