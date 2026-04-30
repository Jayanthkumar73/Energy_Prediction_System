# ⚡ Energy Consumption Forecasting System

A full-stack AI-powered smart meter analytics platform that predicts energy usage using **SARIMA** and **LSTM** models, with anomaly detection, optimization suggestions, and a real-time React dashboard.

---

## 📋 Project Overview

**Domain:** Time Series Forecasting  
**Dataset:** Smart Meter Data (hourly energy consumption)  
**Objective:** Predict energy usage to help users reduce costs and optimize consumption patterns

---

## 🚀 Key Features

- **SARIMA + LSTM Models** — Statistical and deep learning models trained on real smart meter data
- **Auto Model Selection** — Automatically picks the best model based on RMSE, MAE, and wMAPE
- **Prediction API** — FastAPI server with endpoints for forecasting, anomaly detection, and optimization
- **React Dashboard** — Interactive visualization with forecast charts, anomaly detection, and suggestions
- **Anomaly Detection** — Z-score and Isolation Forest methods to flag unusual consumption
- **Optimization Engine** — Rule-based suggestions to reduce energy costs

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | SARIMA (statsmodels), LSTM (TensorFlow/Keras) |
| Anomaly Detection | Z-score, Isolation Forest (scikit-learn) |
| API | FastAPI, Uvicorn |
| Frontend | React, Recharts, Vite |
| Data Processing | Pandas, NumPy, Scikit-learn |

---

## ⚙️ How to Run

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train.py
```
This runs the full 10-step pipeline and saves models to `models/`

If your model files are in Google Drive, the API can download them automatically at startup from this folder:
`https://drive.google.com/drive/u/0/folders/1fkneN8SuULjFOYjXFxS5ShEEsQwUqhi9`

You can override the Drive source or cache location with environment variables:
```bash
$env:MODEL_ARTIFACTS_DRIVE_URL = "https://drive.google.com/drive/u/0/folders/1fkneN8SuULjFOYjXFxS5ShEEsQwUqhi9"
$env:MODEL_ARTIFACTS_DIR = "C:\path\to\your\drive\models"
$env:MODEL_ARTIFACTS_CACHE_DIR = "C:\path\to\cache\models"
```
`MODEL_ARTIFACTS_DRIVE_URL` points the API to the Google Drive folder, `MODEL_ARTIFACTS_DIR` can still be used to force a local artifact folder if you already synced one, and `MODEL_ARTIFACTS_CACHE_DIR` controls where the Drive files are cached.
If you use cmd.exe instead of PowerShell, the equivalent is `set MODEL_ARTIFACTS_DRIVE_URL=...` and `set MODEL_ARTIFACTS_CACHE_DIR=...`.

### 3. Start the API
```bash
python api.py
```
API runs at → http://localhost:8000  
Swagger docs → http://localhost:8000/docs

The backend now downloads model artifacts from Google Drive when the cache is empty, then loads them from the local cache directory.

### 4. Start the Dashboard
```bash
cd frontend
npm install
npm run dev
```
Dashboard runs at → http://localhost:5173

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API and model status |
| POST | `/predict` | Forecast energy consumption |
| POST | `/anomalies` | Detect anomalies in readings |
| POST | `/optimize` | Get energy-saving suggestions |
| GET | `/model/info` | View model metrics and features |

---

## 📉 Model Performance

| Model | RMSE | MAE | sMAPE |
|---|---|---|---|
| SARIMA | ~0.05 | ~0.03 | ~8% |
| LSTM | ~0.04 | ~0.02 | ~6% |



