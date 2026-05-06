"""
ml_model.py
-----------
Machine Learning integration for the trading strategy.

Trains a Random Forest Classifier to predict whether the equity market
will go UP or DOWN over the next 10 days, using our engineered features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.joblib"

FEATURES = [
    "Equity_RollingVol20",
    "Equity_Momentum10",
    "Equity_Momentum30",
    "Inflation_RoC",
    "InterestRate_RoC",
    "Sentiment_norm",
    "Corr_Equity_Gold30",
    "Corr_Equity_Oil30"
]

def prepare_ml_data(df: pd.DataFrame, forward_window: int = 10):
    """Creates the target variable for ML training."""
    future_returns = df["Equity_Price"].shift(-forward_window) / df["Equity_Price"] - 1
    df["Target"] = (future_returns > 0.0).astype(int)
    
    ml_df = df.dropna(subset=FEATURES + ["Target"]).copy()
    ml_df = ml_df.iloc[:-forward_window]
    
    return ml_df

def train_model(df: pd.DataFrame):
    """Tunes and trains the best model using TimeSeriesSplit."""
    print("[ml_model] Preparing data for hyperparameter tuning...")
    ml_df = prepare_ml_data(df)
    
    X = ml_df[FEATURES]
    y = ml_df["Target"]
    
    # Chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[ml_model] Starting Grid Search on {len(X_train)} samples...")
    
    # TimeSeriesSplit prevents look-ahead bias during cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [20, 50, 100]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    
    # GridSearchCV finds the best combination
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\n[ml_model] Best Parameters Found: {grid_search.best_params_}")
    
    # Evaluate best model
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[ml_model] Tuned Test Accuracy: {acc:.2%}")
    print("[ml_model] Classification Report:")
    print(classification_report(y_test, preds))
    
    # Feature Importance
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n[ml_model] Feature Importances:")
    print(importances.to_string())
    
    # Save the best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"[ml_model] Best model saved to {MODEL_PATH}")
    return best_model

def predict_signal(row: pd.Series, model: RandomForestClassifier) -> int:
    """Predicts 1 (UP) or 0 (DOWN) for a single row of data."""
    # Extract features in the correct order
    x = [row.get(f, 0.0) for f in FEATURES]
    pred = model.predict([x])[0]
    return int(pred)

def get_prediction_probability(row: pd.Series, model: RandomForestClassifier) -> float:
    """Returns the probability of the UP (1) class."""
    x = [row.get(f, 0.0) for f in FEATURES]
    prob = model.predict_proba([x])[0][1]
    return float(prob)
