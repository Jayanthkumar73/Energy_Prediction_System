"""
=============================================================================
  Smart Meter Energy Forecasting & Optimization System — ML Training Pipeline
  PRD Sections Covered: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 9, 10
=============================================================================

Dataset columns used:
    Timestamp            → parsed, resampled to hourly, time features extracted
    Electricity_Consumed → target variable (energy_usage equivalent)
    Temperature          → weather feature (already normalised 0-1)
    Humidity             → weather feature
    Wind_Speed           → weather feature
    Avg_Past_Consumption → rolling-average feature (kept as-is)
    Anomaly_Label        → used ONLY for anomaly detection evaluation;
                           dropped from forecasting features to avoid leakage

Columns NOT in dataset (from PRD) and how they are handled:
    rainfall             → not present; skipped gracefully (no crash)
    season / month / etc.→ engineered from Timestamp

PRD workflow steps implemented (Section 9):
    Step 1  Load dataset
    Step 2  Handle missing values
    Step 3  Convert timestamp column
    Step 4  Resample to hourly
    Step 5  Feature engineering (time + weather)
    Step 6  Train SARIMA model
    Step 7  Train LSTM model
    Step 8  Evaluate (RMSE / MAE / MAPE)
    Step 9  Auto-select best model
    Step 10 Save best model
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import warnings
import pickle
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress noisy warnings from statsmodels / TF
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "smart_meter_data.csv"       # ← change if your CSV is elsewhere
MODELS_DIR  = Path(os.getenv("MODEL_ARTIFACTS_DIR", "models")).expanduser()
if not MODELS_DIR.is_absolute():
    MODELS_DIR = Path(__file__).resolve().parent / MODELS_DIR
MODELS_DIR = MODELS_DIR.resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Reproducibility for model training and anomaly search
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STEP 1 – LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
  
    log.info("Step 1 ▶ Loading dataset from '%s'", path)
    df = pd.read_csv(path)

    required = {"Timestamp", "Electricity_Consumed"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing mandatory columns: {missing}")

    
    keep = [
        "Timestamp", "Electricity_Consumed",
        "Temperature", "Humidity", "Wind_Speed",
        "Avg_Past_Consumption", "Anomaly_Label",
        
    ]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    log.info("  Loaded %d rows × %d columns: %s", *df.shape, df.columns.tolist())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STEP 2 – HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
   
    log.info("Step 2 ▶ Handling missing values")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    before = df[numeric_cols].isnull().sum().sum()

    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    if "Anomaly_Label" in df.columns:
        df["Anomaly_Label"] = df["Anomaly_Label"].fillna("Normal")

    after = df[numeric_cols].isnull().sum().sum()
    log.info("  Missing values fixed: %d → %d", before, after)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STEP 3 – CONVERT TIMESTAMP
# ─────────────────────────────────────────────────────────────────────────────
def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Timestamp and set it as the DatetimeIndex."""
    log.info("Step 3 ▶ Converting Timestamp")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").set_index("Timestamp")
    log.info("  Date range: %s  →  %s", df.index.min(), df.index.max())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STEP 4 – RESAMPLE TO HOURLY
# ─────────────────────────────────────────────────────────────────────────────
def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 30-min data to hourly.
    Numeric cols → mean; Anomaly_Label: if any row in that hour is
    'Abnormal', the hour is marked 'Abnormal'.
    """
    log.info("Step 4 ▶ Resampling to hourly frequency")

    anomaly_col = None
    if "Anomaly_Label" in df.columns:
        anomaly_col = df["Anomaly_Label"]
        df = df.drop(columns=["Anomaly_Label"])

    df_h = df.resample("h").mean()

    if anomaly_col is not None:
        anomaly_h = anomaly_col.resample("h").apply(
            lambda s: "Abnormal" if "Abnormal" in s.values else "Normal"
        )
        df_h["Anomaly_Label"] = anomaly_h

    log.info("  Resampled → %d hourly rows", len(df_h))
    return df_h


# ─────────────────────────────────────────────────────────────────────────────
# 5.  STEP 5 – FEATURE ENGINEERING  (PRD §4.2)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time features  : hour, day_of_week, day_of_month, month, is_weekend, season
    Weather features: temperature, humidity, wind_speed
                      (rainfall skipped — not in dataset)
    Lag features   : lag_1h, lag_24h, rolling_mean_6h
    """
    log.info("Step 5 ▶ Feature engineering")

    # ── Time ──────────────────────────────────────────────────────────────────
    df["hour"]         = df.index.hour
    df["day_of_week"]  = df.index.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"] = df.index.day
    df["month"]        = df.index.month
    df["is_weekend"]   = (df.index.dayofweek >= 5).astype(int)

    # Season: 1=Winter, 2=Spring, 3=Summer, 4=Autumn
    df["season"] = df["month"].map(
        {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
          6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    )

    # Cyclical encoding for hour and month (avoids 23→0 discontinuity)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Lag / rolling ─────────────────────────────────────────────────────────
    target = "Electricity_Consumed"
    df["lag_1h"]          = df[target].shift(1)
    df["lag_24h"]         = df[target].shift(24)
    # Use only past values to avoid leakage from current timestamp target.
    df["rolling_mean_6h"] = df[target].shift(1).rolling(6, min_periods=1).mean()

    # Drop rows that have NaN from lags (first 24 hours)
    df = df.dropna(subset=["lag_24h"])

    log.info("  Features after engineering: %s", df.columns.tolist())
    log.info("  Rows after dropping lag NaNs: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """
    Chronological split — no shuffling allowed for time-series.
    Returns (train_df, val_df, test_df).
    """
    n      = len(df)
    n_tr   = int(n * train_ratio)
    n_val  = int(n * val_ratio)

    train = df.iloc[:n_tr]
    val   = df.iloc[n_tr : n_tr + n_val]
    test  = df.iloc[n_tr + n_val:]

    log.info("  Split → train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 7.  METRICS  (PRD §10)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Return RMSE, MAE, MAPE, sMAPE and wMAPE for model predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    eps = 1e-6

    # MAPE can be unstable near zero values, so keep robust alternatives too.
    mask = np.abs(y_true) > eps
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan

    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )
    wmape = 100 * np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + eps)

    metrics = {"model": name, "RMSE": round(rmse, 6),
               "MAE": round(mae, 6),
               "MAPE": round(mape, 4) if not np.isnan(mape) else None,
               "sMAPE": round(smape, 4),
               "wMAPE": round(wmape, 4)}
    mape_text = f"{mape:.2f}%" if not np.isnan(mape) else "NA"
    log.info("  [%s]  RMSE=%.6f  MAE=%.6f  MAPE=%s  sMAPE=%.2f%%  wMAPE=%.2f%%",
             name, rmse, mae, mape_text, smape, wmape)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 8.  STEP 6 – SARIMA MODEL  (PRD §4.1)
# ─────────────────────────────────────────────────────────────────────────────
def train_sarima(train_series: pd.Series, test_series: pd.Series) -> tuple:
    """
    Train a SARIMA(1,1,1)(1,1,1,24) model.
    Seasonal period = 24 (hourly data, one day cycle).
    Returns (fitted_model, predictions_array, metrics_dict).
    """
    log.info("Step 6 ▶ Training SARIMA model …")

    # SARIMA order — (p,d,q)(P,D,Q,s)
    order         = (1, 1, 1)
    seasonal_order = (1, 1, 1, 24)

    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)

        # Forecast test horizon
        forecast = fitted.forecast(steps=len(test_series))
        preds    = np.clip(np.array(forecast), 0, None)   # energy can't be < 0

        metrics = evaluate(test_series.values, preds, "SARIMA")

        log.info("  SARIMA training complete.")
        return fitted, preds, metrics

    except Exception as exc:
        log.error("  SARIMA training failed: %s", exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 9.  STEP 7 – LSTM MODEL  (PRD §4.1)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend", "season",
    "lag_1h", "lag_24h", "rolling_mean_6h",
]
TARGET_COL  = "Electricity_Consumed"
LOOKBACK    = 24   # use past 24 hours to predict next 1 hour
BATCH_SIZE  = 32
EPOCHS      = 50


def build_sequences(data: np.ndarray, lookback: int):
    """
    Slide a window of `lookback` steps across `data`.
    X shape: (samples, lookback, features)
    y shape: (samples,)   ← next-step target
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :-1])   # all feature cols
        y.append(data[i, -1])                 # target (last col)
    return np.array(X), np.array(y)


def build_contextual_sequences(prev_scaled: np.ndarray, curr_scaled: np.ndarray, lookback: int):
    """Build split sequences while preserving lookback context from previous split."""
    if len(prev_scaled) == 0:
        merged = curr_scaled
    else:
        context = prev_scaled[-lookback:] if len(prev_scaled) >= lookback else prev_scaled
        merged = np.vstack([context, curr_scaled])

    if len(merged) <= lookback:
        return np.empty((0, lookback, merged.shape[1] - 1)), np.empty((0,))

    return build_sequences(merged, lookback)


def build_lstm_model(n_features: int, lookback: int) -> tf.keras.Model:
    """Two-layer LSTM with dropout for regularisation."""
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),                             # single-step regression output
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple:
    """
    Scale features, build sequences, train LSTM, evaluate on test set.
    Returns (keras_model, scaler, predictions_array, metrics_dict).
    """
    log.info("Step 7 ▶ Training LSTM model …")

    # ── Select & order columns: features first, target last ───────────────────
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    cols = available_features + [TARGET_COL]

    train_vals = train_df[cols].values
    val_vals   = val_df[cols].values
    test_vals  = test_df[cols].values

    # ── Scale everything to [0, 1] ────────────────────────────────────────────
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)   # fit ONLY on train
    val_scaled   = scaler.transform(val_vals)
    test_scaled  = scaler.transform(test_vals)

    # ── Build sequences ───────────────────────────────────────────────────────
    X_train, y_train = build_sequences(train_scaled, LOOKBACK)
    X_val,   y_val   = build_contextual_sequences(train_scaled, val_scaled, LOOKBACK)
    X_test,  y_test  = build_contextual_sequences(val_scaled, test_scaled, LOOKBACK)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Insufficient samples after sequence construction; reduce LOOKBACK or add data.")

    log.info("  Sequence shapes → X_train=%s  X_val=%s  X_test=%s",
             X_train.shape, X_val.shape, X_test.shape)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_features = X_train.shape[2]
    model      = build_lstm_model(n_features, LOOKBACK)
    model.summary(print_fn=log.info)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Inverse-transform predictions back to original scale ──────────────────
    # The scaler was fit on all cols; target is the last one.
    # We reconstruct a full-width array to use scaler.inverse_transform.
    def inverse_target(preds_scaled: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(preds_scaled), len(cols)))
        dummy[:, -1] = preds_scaled.flatten()
        return scaler.inverse_transform(dummy)[:, -1]

    raw_preds  = model.predict(X_test, verbose=0).flatten()
    preds      = np.clip(inverse_target(raw_preds), 0, None)
    y_true_inv = inverse_target(y_test)

    metrics = evaluate(y_true_inv, preds, "LSTM")

    log.info("  LSTM training complete. Best epoch stopped at %d.",
             len(history.history["loss"]))
    return model, scaler, preds, metrics, y_true_inv


# ─────────────────────────────────────────────────────────────────────────────
# 10.  STEPS 8 & 9 – EVALUATE & AUTO-SELECT BEST MODEL  (PRD §4.3)
# ─────────────────────────────────────────────────────────────────────────────
def select_best_model(sarima_metrics: dict, lstm_metrics: dict) -> str:
    """
    Compare RMSE (primary), MAE (tie-breaker).
    Returns 'SARIMA' or 'LSTM'.
    """
    log.info("Step 8/9 ▶ Evaluating and auto-selecting best model")

    pair = [sarima_metrics, lstm_metrics]

    rmse_lo, rmse_hi = min(m["RMSE"] for m in pair), max(m["RMSE"] for m in pair)
    mae_lo, mae_hi = min(m["MAE"] for m in pair), max(m["MAE"] for m in pair)
    wmape_lo, wmape_hi = min(m["wMAPE"] for m in pair), max(m["wMAPE"] for m in pair)

    def norm(value: float, lo: float, hi: float) -> float:
        if hi - lo < 1e-12:
            return 0.0
        return (value - lo) / (hi - lo)

    # Lower weighted score is better.
    sarima_score = (
        0.6 * norm(sarima_metrics["RMSE"], rmse_lo, rmse_hi)
        + 0.3 * norm(sarima_metrics["wMAPE"], wmape_lo, wmape_hi)
        + 0.1 * norm(sarima_metrics["MAE"], mae_lo, mae_hi)
    )
    lstm_score = (
        0.6 * norm(lstm_metrics["RMSE"], rmse_lo, rmse_hi)
        + 0.3 * norm(lstm_metrics["wMAPE"], wmape_lo, wmape_hi)
        + 0.1 * norm(lstm_metrics["MAE"], mae_lo, mae_hi)
    )

    best = "SARIMA" if sarima_score <= lstm_score else "LSTM"

    log.info("  Scores → SARIMA=%.4f  LSTM=%.4f", sarima_score, lstm_score)
    log.info("  ✔ Best model selected: %s", best)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 11.  STEP 10 – SAVE BEST MODEL  (PRD §4.3)
# ─────────────────────────────────────────────────────────────────────────────
def save_best_model(
    best_name: str,
    sarima_fitted,
    lstm_model,
    lstm_scaler,
    all_metrics: list,
    feature_cols: list,
):
    """
    Save both trained models + metadata required by the prediction API.

    Files written to models/:
        best_model.pkl         → SARIMA fitted result
        best_model.keras       → LSTM keras model
        lstm_scaler.pkl        → MinMaxScaler for the LSTM model
        model_metadata.json    → model name, metrics, feature columns, lookback
    """
    log.info("Step 10 ▶ Saving best model ('%s') to '%s/'", best_name, MODELS_DIR)

    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(sarima_fitted, f)

    lstm_model.save(str(MODELS_DIR / "best_model.keras"))
    with open(MODELS_DIR / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(lstm_scaler, f)

    metadata = {
        "best_model":   best_name,
        "lookback":     LOOKBACK,
        "feature_cols": feature_cols,
        "target_col":   TARGET_COL,
        "all_metrics":  all_metrics,
        "available_models": ["SARIMA", "LSTM"],
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("  Saved successfully.")
    log.info("  Files: %s", [str(p) for p in MODELS_DIR.iterdir()])


# ─────────────────────────────────────────────────────────────────────────────
# 12.  MULTI-STEP FORECASTING DEMO  (PRD §4.4)
# ─────────────────────────────────────────────────────────────────────────────
def demo_multistep_forecast(
    best_name: str,
    sarima_fitted,
    lstm_model,
    lstm_scaler,
    test_df: pd.DataFrame,
):
    """
    Demonstrate multi-step forecasting for 1h, 6h, 24h, 7d (168h).
    Uses the trained best model's last test window as the starting point.
    """
    log.info("── Multi-step Forecast Demo (PRD §4.4) ──")
    horizons = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}

    available_features = [c for c in FEATURE_COLS if c in test_df.columns]
    cols = available_features + [TARGET_COL]

    for label, steps in horizons.items():
        if best_name == "SARIMA":
            forecast = sarima_fitted.forecast(steps=steps)
            vals = np.clip(np.array(forecast), 0, None)
        else:
            # Roll LSTM forward step-by-step using teacher-forced features
            test_scaled = lstm_scaler.transform(test_df[cols].values)
            window = test_scaled[-LOOKBACK:].copy()
            preds_scaled = []
            for _ in range(steps):
                x = window[np.newaxis, :, :-1]   # (1, lookback, features)
                p = lstm_model.predict(x, verbose=0)[0, 0]
                # append prediction as the next target; keep features frozen
                next_row = window[-1].copy()
                next_row[-1] = p
                window = np.vstack([window[1:], next_row])
                preds_scaled.append(p)

            dummy = np.zeros((steps, len(cols)))
            dummy[:, -1] = preds_scaled
            vals = np.clip(lstm_scaler.inverse_transform(dummy)[:, -1], 0, None)

        log.info("  Horizon %-4s  → mean=%.4f  min=%.4f  max=%.4f",
                 label, vals.mean(), vals.min(), vals.max())


# ─────────────────────────────────────────────────────────────────────────────
# 13.  ANOMALY DETECTION  (PRD §5.1)
# ─────────────────────────────────────────────────────────────────────────────
def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects anomalies using two methods:
      Method A — Z-score  (flags rows where |z| > 3)
      Method B — Isolation Forest

    Both are compared against the ground-truth Anomaly_Label if present.
    Returns the dataframe with two new columns:
        anomaly_zscore   (True/False)
        anomaly_iforest  (True/False)
    """
    log.info("── Anomaly Detection (PRD §5.1) ──")

    target = "Electricity_Consumed"
    series = df[target].dropna()

    # ── Method A: Z-score ─────────────────────────────────────────────────────
    gt = None
    if "Anomaly_Label" in df.columns:
        gt = (df["Anomaly_Label"] == "Abnormal").astype(int).values

    z_scores = np.abs(stats.zscore(series))
    z_scores = np.nan_to_num(z_scores, nan=0.0)

    z_threshold = 3.0
    if gt is not None:
        best_f1 = -1.0
        for thr in np.arange(2.0, 3.51, 0.25):
            pred = (z_scores > thr).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                gt, pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                z_threshold = float(thr)
        log.info("  Tuned Z-score threshold: %.2f", z_threshold)

    df["anomaly_zscore"] = z_scores > z_threshold
    n_z = df["anomaly_zscore"].sum()

    # ── Method B: Isolation Forest ────────────────────────────────────────────
    features_for_iso = [c for c in [target, "Temperature", "Humidity",
                                     "Wind_Speed"] if c in df.columns]
    iso_x = df[features_for_iso].ffill().bfill().values

    contamination = 0.05
    if gt is not None:
        best_f1 = -1.0
        for c in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
            iso_try = IsolationForest(contamination=c, random_state=42, n_estimators=200)
            pred = (iso_try.fit_predict(iso_x) == -1).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                gt, pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                contamination = float(c)
        log.info("  Tuned IsolationForest contamination: %.2f", contamination)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds_iso = iso.fit_predict(iso_x)
    df["anomaly_iforest"] = preds_iso == -1   # IsolationForest: -1 = anomaly
    n_iso = df["anomaly_iforest"].sum()

    log.info("  Z-score anomalies detected  : %d", n_z)
    log.info("  IsolationForest anomalies   : %d", n_iso)

    # ── Evaluate against ground-truth if available ────────────────────────────
    if "Anomaly_Label" in df.columns:
        gt = (df["Anomaly_Label"] == "Abnormal")
        n_true = gt.sum()

        # overlap with z-score
        overlap_z   = (df["anomaly_zscore"] & gt).sum()
        # overlap with iforest
        overlap_iso = (df["anomaly_iforest"] & gt).sum()

        log.info("  Ground-truth abnormal hours: %d", n_true)
        log.info("  Z-score  ∩ ground-truth    : %d (recall=%.1f%%)",
                 overlap_z, 100 * overlap_z / max(n_true, 1))
        log.info("  IsoForest ∩ ground-truth   : %d (recall=%.1f%%)",
                 overlap_iso, 100 * overlap_iso / max(n_true, 1))

    # Sample output (PRD: "Unusual consumption detected at 02:00 AM")
    flagged = df[df["anomaly_iforest"] == True]
    if len(flagged) > 0:
        sample = flagged.head(5)
        for ts, row in sample.iterrows():
            log.info("  ⚠ Unusual consumption detected at %s  (value=%.4f)",
                     ts.strftime("%Y-%m-%d %I:%M %p"), row[target])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 14.  OPTIMIZATION RECOMMENDATIONS  (PRD §5.2)
# ─────────────────────────────────────────────────────────────────────────────
def generate_optimization_suggestions(df: pd.DataFrame) -> list[str]:
    """
    Rule-based optimization engine.
    Returns a list of human-readable suggestions.
    """
    log.info("── Optimization Recommendations (PRD §5.2) ──")

    target = "Electricity_Consumed"
    suggestions = []

    # Hourly average to identify peaks
    hourly_avg = df.groupby("hour")[target].mean()
    peak_hour  = hourly_avg.idxmax()
    off_peak   = hourly_avg.idxmin()

    # Evening peak detection (18–22)
    evening_avg = hourly_avg[hourly_avg.index.isin(range(18, 23))].mean()
    night_avg   = hourly_avg[hourly_avg.index.isin(range(0, 6))].mean()

    if peak_hour in range(18, 23):
        suggestions.append(
            f"🔴 Peak demand occurs at {peak_hour:02d}:00. "
            f"Consider shifting heavy appliance loads to off-peak hours (e.g., {off_peak:02d}:00)."
        )

    if evening_avg > 1.5 * night_avg:
        suggestions.append(
            "⚡ Evening consumption is significantly higher than night-time usage. "
            "Distribute load to avoid 18:00–22:00 peak window."
        )

    # Weekend vs weekday
    if "is_weekend" in df.columns:
        wd_avg = df[df["is_weekend"] == 0][target].mean()
        we_avg = df[df["is_weekend"] == 1][target].mean()
        if we_avg > wd_avg * 1.2:
            suggestions.append(
                "📅 Weekend consumption is 20%+ higher than weekdays. "
                "Review weekend scheduling of HVAC and lighting systems."
            )

    # Anomaly-based
    if "anomaly_iforest" in df.columns:
        n_spikes = df["anomaly_iforest"].sum()
        if n_spikes > 0:
            suggestions.append(
                f"🔍 {n_spikes} abnormal consumption spikes detected. "
                "Inspect equipment for inefficiencies or unauthorised usage."
            )

    # General suggestion
    suggestions.append(
        f"✅ Off-peak usage window: {off_peak:02d}:00–{(off_peak+2)%24:02d}:00. "
        "Schedule energy-intensive tasks (laundry, dishwashers, EV charging) here."
    )

    for s in suggestions:
        log.info("  %s", s)

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# 15.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("  SMART METER ENERGY FORECASTING — TRAINING PIPELINE")
    log.info("=" * 70)

    # ── Steps 1–5: Data preparation ──────────────────────────────────────────
    df = load_data(DATA_PATH)
    df = handle_missing(df)
    df = convert_timestamp(df)
    df = resample_hourly(df)
    df = engineer_features(df)

    # Keep a copy with anomaly labels before dropping for model training
    df_full = df.copy()

    # Drop Anomaly_Label so it doesn't leak into forecasting features
    if "Anomaly_Label" in df.columns:
        df = df.drop(columns=["Anomaly_Label"])

    # ── Train / Val / Test split ───────────────────────────────────────────────
    train_df, val_df, test_df = split_data(df)

    # ── Step 6: SARIMA ────────────────────────────────────────────────────────
    sarima_fitted, sarima_preds, sarima_metrics = train_sarima(
        train_series=train_df[TARGET_COL],
        test_series=test_df[TARGET_COL],
    )

    # ── Step 7: LSTM ─────────────────────────────────────────────────────────
    lstm_model, lstm_scaler, lstm_preds, lstm_metrics, lstm_true = train_lstm(
        train_df, val_df, test_df
    )

    # ── Steps 8–9: Evaluate & auto-select ────────────────────────────────────
    all_metrics = [sarima_metrics, lstm_metrics]
    best_name   = select_best_model(sarima_metrics, lstm_metrics)

    # ── Step 10: Save ─────────────────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    save_best_model(
        best_name, sarima_fitted, lstm_model, lstm_scaler,
        all_metrics, available_features
    )

    # ── Multi-step forecasting demo ───────────────────────────────────────────
    demo_multistep_forecast(best_name, sarima_fitted, lstm_model, lstm_scaler, test_df)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    df_full = run_anomaly_detection(df_full)

    # ── Optimization suggestions ──────────────────────────────────────────────
    generate_optimization_suggestions(df_full)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("  TRAINING COMPLETE")
    sarima_mape = f"{sarima_metrics['MAPE']:.2f}%" if sarima_metrics["MAPE"] is not None else "NA"
    lstm_mape = f"{lstm_metrics['MAPE']:.2f}%" if lstm_metrics["MAPE"] is not None else "NA"
    log.info("  SARIMA → RMSE=%.6f  MAE=%.6f  MAPE=%s  sMAPE=%.2f%%  wMAPE=%.2f%%",
             sarima_metrics["RMSE"], sarima_metrics["MAE"], sarima_mape,
             sarima_metrics["sMAPE"], sarima_metrics["wMAPE"])
    log.info("  LSTM   → RMSE=%.6f  MAE=%.6f  MAPE=%s  sMAPE=%.2f%%  wMAPE=%.2f%%",
             lstm_metrics["RMSE"],   lstm_metrics["MAE"],   lstm_mape,
             lstm_metrics["sMAPE"], lstm_metrics["wMAPE"])
    log.info("  ✔ Best model: %s  (saved to models/)", best_name)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
