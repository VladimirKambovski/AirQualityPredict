"""
Train a Random Forest model to predict next-day PM2.5 for Skopje.

Pipeline:
    1. Load processed features
    2. Chronological train/test split (NOT random — time series!)
    3. Train Random Forest Regressor
    4. Evaluate on both sets (detect overfitting)
    5. Print feature importances
    6. Save model to disk
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from config import (
    PROCESSED_DATA_PATH, MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE, N_ESTIMATORS,
)


def load_data():
    """Load processed features and split into X (features) and y (target)."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    print(f"Loaded {len(df)} samples, {len(FEATURE_COLUMNS)} features.")
    return X, y


def chronological_split(X, y):
    """Split data by time — train on the past, test on the future.

    Why not random split? In time series, random splitting leaks future
    information into training. The model sees December data while training,
    then gets 'tested' on October — that's cheating.

    Chronological split is the only honest evaluation for time series.
    """
    split_idx = int(len(X) * (1 - TEST_SIZE))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"Train: {len(X_train)} samples (older data)")
    print(f"Test:  {len(X_test)} samples (newer data)")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train a Random Forest Regressor.

    Why Random Forest?
    - Handles non-linear patterns (weather → pollution is complex)
    - Built-in feature importance (explainability)
    - No feature scaling needed (splits on thresholds, not distances)
    - Ensemble of 100 trees reduces overfitting vs a single tree
    """
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print(f"\nTrained Random Forest with {N_ESTIMATORS} trees.")
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    """Print RMSE, MAE, R² for train and test sets.

    Comparing train vs test metrics tells us about overfitting:
    - Similar values → model generalizes well
    - Train much better than test → overfitting
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
    }

    print("\n--- Evaluation ---")
    print(f"{'Metric':<8} {'Train':>10} {'Test':>10}")
    print(f"{'RMSE':<8} {metrics['train_rmse']:>10.2f} {metrics['test_rmse']:>10.2f}")
    print(f"{'MAE':<8} {metrics['train_mae']:>10.2f} {metrics['test_mae']:>10.2f}")
    print(f"{'R²':<8} {metrics['train_r2']:>10.3f} {metrics['test_r2']:>10.3f}")

    # Overfitting check: compare train vs test R²
    r2_gap = metrics["train_r2"] - metrics["test_r2"]
    if r2_gap > 0.1:
        print(f"\nWarning: possible overfitting (R² gap: {r2_gap:.3f}).")
        print("    Consider: max_depth limit, fewer trees, or more data.")
    else:
        print(f"\nTrain/test R² gap: {r2_gap:.3f} — model generalizes well.")

    return metrics


def show_feature_importance(model):
    """Show which features the model relies on most.

    This is one of Random Forest's best features — built-in importance
    scores. Great for explaining the model to stakeholders or interviewers.
    """
    importances = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)

    print("\n--- Feature Importance ---")
    for name, score in pairs:
        bar = "#" * int(score * 50)
        print(f"  {name:<18} {score:.3f} {bar}")


def save_model(model):
    """Save trained model to disk using joblib."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\nModel saved to {MODEL_PATH} ({model_size:.1f} MB)")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    model = train_model(X_train, y_train)
    evaluate(model, X_train, X_test, y_train, y_test)
    show_feature_importance(model)
    save_model(model)

    print("\nDone! Next step: python app.py")
