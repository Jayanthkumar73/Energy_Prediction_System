"""
=============================================================================
  Smart Meter Energy Forecasting — Prediction API  (US-07)
  FastAPI server exposing: /predict, /anomalies, /optimize, /health
=============================================================================
"""

import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Load saved model artifacts ────────────────────────────────────────────────
MODELS_DIR = Path("models")

def load_metadata():
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise RuntimeError("models/model_metadata.json not found. Run train.py first.")

    with open(meta_path) as f:
        return json.load(f)


def load_sarima_model():
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise RuntimeError("SARIMA artifact models/best_model.pkl not found.")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_lstm_model():
    model_path = MODELS_DIR / "best_model.keras"
    scaler_path = MODELS_DIR / "lstm_scaler.pkl"

    if not model_path.exists():
        raise RuntimeError("LSTM artifact models/best_model.keras not found.")
    if not scaler_path.exists():
        raise RuntimeError("LSTM scaler models/lstm_scaler.pkl not found.")

    import tensorflow as tf

    model = tf.keras.models.load_model(str(model_path))
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    META = load_metadata()
    BEST_NAME = META["best_model"]
    LOOKBACK = META.get("lookback", 24)
    FEATURE_COLS = META.get("feature_cols", [])
    TARGET_COL = META.get("target_col", "Electricity_Consumed")

    MODEL_BUNDLES = {}
    AVAILABLE_MODELS = []

    try:
        MODEL_BUNDLES["SARIMA"] = {"model": load_sarima_model(), "scaler": None}
        AVAILABLE_MODELS.append("SARIMA")
    except Exception as exc:
        log.warning("SARIMA model unavailable: %s", exc)

    try:
        lstm_model, lstm_scaler = load_lstm_model()
        MODEL_BUNDLES["LSTM"] = {"model": lstm_model, "scaler": lstm_scaler}
        AVAILABLE_MODELS.append("LSTM")
    except Exception as exc:
        log.warning("LSTM model unavailable: %s", exc)

    MODEL_READY = len(AVAILABLE_MODELS) > 0
except Exception as exc:
    log.warning("Could not load model metadata: %s — /predict will return 503", exc)
    META = {}
    BEST_NAME = "UNKNOWN"
    LOOKBACK, FEATURE_COLS, TARGET_COL = 24, [], "Electricity_Consumed"
    MODEL_BUNDLES = {}
    AVAILABLE_MODELS = []
    MODEL_READY = False


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="SARIMA + LSTM based energy forecasting with anomaly detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    horizon: int = Field(default=24, ge=1, le=168, description="Hours to forecast (1–168)")
    model_name: Optional[str] = Field(default=None, description="Model to use: SARIMA or LSTM")
    temperature: Optional[float] = Field(default=0.5, ge=0, le=1)
    humidity:    Optional[float] = Field(default=0.5, ge=0, le=1)
    wind_speed:  Optional[float] = Field(default=0.3, ge=0, le=1)
    start_time:  Optional[str]   = Field(default=None, description="ISO-8601 start timestamp")

class ForecastPoint(BaseModel):
    timestamp: str
    predicted_kwh: float

class ForecastResponse(BaseModel):
    model_used: str
    horizon_hours: int
    predictions: List[ForecastPoint]

class AnomalyRequest(BaseModel):
    readings: List[float] = Field(..., description="List of hourly energy readings (kWh)")
    timestamps: Optional[List[str]] = None

class OptimizationRequest(BaseModel):
    hourly_avg: List[float] = Field(..., description="24-element list of avg hourly usage")
    has_weekend_spike: Optional[bool] = False
    anomaly_count: Optional[int] = 0


# ── Helper: build a single feature row for LSTM ───────────────────────────────
def _make_feature_row(ts: datetime, temp: float, hum: float, wind: float,
                      prev_val: float, lag_24: float, roll6: float) -> dict:
    season_map = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
    return {
        "Temperature":          temp,
        "Humidity":             hum,
        "Wind_Speed":           wind,
        "Avg_Past_Consumption": prev_val,
        "hour_sin":  np.sin(2*np.pi*ts.hour/24),
        "hour_cos":  np.cos(2*np.pi*ts.hour/24),
        "month_sin": np.sin(2*np.pi*ts.month/12),
        "month_cos": np.cos(2*np.pi*ts.month/12),
        "is_weekend": int(ts.weekday() >= 5),
        "season":     season_map.get(ts.month, 2),
        "lag_1h":     prev_val,
        "lag_24h":    lag_24,
        "rolling_mean_6h": roll6,
    }


def _seed_consumption_baseline(ts: datetime) -> float:
    """Create a deterministic baseline that varies by hour and weekend."""
    hour_wave = 0.08 * np.sin(2 * np.pi * ts.hour / 24)
    weekend_bump = 0.04 if ts.weekday() >= 5 else 0.0
    return float(np.clip(0.5 + hour_wave + weekend_bump, 0.05, 0.95))


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_READY else "model_not_loaded",
        "best_model": BEST_NAME,
        "lookback": LOOKBACK,
        "available_models": AVAILABLE_MODELS,
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    if not MODEL_READY:
        raise HTTPException(503, "Model not loaded. Run train.py first.")

    start = datetime.fromisoformat(req.start_time.replace("Z", "")) if req.start_time else datetime.utcnow()
    horizon = req.horizon
    requested_model = (req.model_name or BEST_NAME or "SARIMA").upper()

    if requested_model not in MODEL_BUNDLES:
        raise HTTPException(400, f"Requested model '{requested_model}' is not available. Available models: {', '.join(AVAILABLE_MODELS) or 'none'}")

    bundle = MODEL_BUNDLES[requested_model]
    model = bundle["model"]
    scaler = bundle["scaler"]

    predictions = []

    if requested_model == "SARIMA":
        try:
            forecast = model.forecast(steps=horizon)
            vals = np.clip(np.array(forecast), 0, None)
        except Exception as e:
            raise HTTPException(500, f"SARIMA forecast error: {e}")

        for i, v in enumerate(vals):
            ts = start + timedelta(hours=i)
            predictions.append(ForecastPoint(
                timestamp=ts.isoformat(),
                predicted_kwh=round(float(v), 4)
            ))

    else:  # LSTM
        # Build a timestamp-aware synthetic seed window
        window_rows = []
        target_history = []

        for i in range(LOOKBACK):
            ts = start - timedelta(hours=LOOKBACK - i)
            seed_val = _seed_consumption_baseline(ts)
            lag24 = target_history[-24] if len(target_history) >= 24 else seed_val
            roll6 = float(np.mean(target_history[-6:])) if len(target_history) >= 6 else seed_val
            row = _make_feature_row(ts, req.temperature, req.humidity,
                                    req.wind_speed, seed_val, lag24, roll6)
            window_rows.append([row.get(c, 0.0) for c in FEATURE_COLS] + [seed_val])
            target_history.append(seed_val)

        window = np.array(window_rows, dtype=np.float32)
        window_scaled = scaler.transform(window)

        for i in range(horizon):
            ts = start + timedelta(hours=i)
            x = window_scaled[np.newaxis, :, :-1]
            p = float(model.predict(x, verbose=0)[0, 0])

            # inverse transform
            dummy = np.zeros((1, len(FEATURE_COLS) + 1))
            dummy[0, -1] = p
            val = float(np.clip(scaler.inverse_transform(dummy)[0, -1], 0, None))

            predictions.append(ForecastPoint(
                timestamp=ts.isoformat(),
                predicted_kwh=round(val, 4)
            ))

            # Build the next row with updated time and lag features so date/time input affects the trajectory.
            target_history.append(val)
            lag1_next = target_history[-1]
            lag24_next = target_history[-24] if len(target_history) >= 24 else target_history[0]
            roll6_next = float(np.mean(target_history[-6:]))
            next_ts = start + timedelta(hours=i + 1)

            next_features = _make_feature_row(
                next_ts,
                req.temperature,
                req.humidity,
                req.wind_speed,
                lag1_next,
                lag24_next,
                roll6_next,
            )
            next_unscaled_row = np.array([next_features.get(c, 0.0) for c in FEATURE_COLS] + [val], dtype=np.float32)
            next_scaled_row = scaler.transform(next_unscaled_row.reshape(1, -1))[0]

            # slide window
            window_scaled = np.vstack([window_scaled[1:], next_scaled_row])

    return ForecastResponse(
        model_used=requested_model,
        horizon_hours=horizon,
        predictions=predictions,
    )


@app.post("/anomalies")
def detect_anomalies(req: AnomalyRequest):
    """Z-score anomaly detection on provided readings."""
    readings = np.array(req.readings, dtype=float)
    if len(readings) < 3:
        raise HTTPException(400, "Need at least 3 readings for anomaly detection.")

    mean, std = readings.mean(), readings.std()
    z_scores = np.abs((readings - mean) / (std + 1e-9))
    flags = z_scores > 3.0

    n = len(readings)
    timestamps = req.timestamps or [
        (datetime.utcnow() - timedelta(hours=n-i)).isoformat()
        for i in range(n)
    ]

    anomalies = [
        {"timestamp": timestamps[i], "value": float(readings[i]),
         "z_score": round(float(z_scores[i]), 3)}
        for i in range(n) if flags[i]
    ]

    return {
        "total_readings": n,
        "anomalies_detected": int(flags.sum()),
        "anomalies": anomalies,
    }


@app.post("/optimize")
def optimize(req: OptimizationRequest):
    """Rule-based optimization suggestions."""
    avg = np.array(req.hourly_avg)
    if len(avg) != 24:
        raise HTTPException(400, "hourly_avg must have exactly 24 values.")

    suggestions = []
    peak_hour = int(np.argmax(avg))
    off_peak  = int(np.argmin(avg))
    evening   = avg[18:23].mean()
    night     = avg[0:6].mean()

    if peak_hour in range(18, 23):
        suggestions.append(
            f"🔴 Peak demand at {peak_hour:02d}:00. Shift heavy loads to {off_peak:02d}:00."
        )
    if evening > 1.5 * night + 1e-6:
        suggestions.append("⚡ Evening consumption is significantly higher. Distribute load away from 18–22h.")
    if req.has_weekend_spike:
        suggestions.append("📅 Weekend usage 20%+ above weekdays. Review HVAC/lighting schedules.")
    if req.anomaly_count and req.anomaly_count > 0:
        suggestions.append(f"🔍 {req.anomaly_count} anomalous spikes detected. Inspect equipment.")
    suggestions.append(
        f"✅ Best off-peak window: {off_peak:02d}:00–{(off_peak+2)%24:02d}:00. "
        "Schedule laundry, EV charging, dishwashers here."
    )

    return {"suggestions": suggestions, "peak_hour": peak_hour, "off_peak_hour": off_peak}


@app.get("/model/info")
def model_info():
    if not MODEL_READY:
        raise HTTPException(503, "Model not loaded.")
    return {
        "best_model": BEST_NAME,
        "feature_cols": FEATURE_COLS,
        "lookback": LOOKBACK,
        "available_models": AVAILABLE_MODELS,
        "metrics": META.get("all_metrics", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=10000, reload=True)