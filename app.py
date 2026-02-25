"""
FastAPI inference endpoint for AirQualityPredict.

Endpoints:
    GET  /health   → is the model loaded and ready?
    POST /predict  → predict next-day PM2.5 for Skopje
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
from config import MODEL_PATH, FEATURE_COLUMNS, CITY


# --- Load model once at startup (not per-request) ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"Warning: no model at {MODEL_PATH} — run train.py first.")


# --- App ---
app = FastAPI(
    title="AirQualityPredict",
    description=f"Predict next-day PM2.5 air quality for {CITY}",
    version="1.0.0",
)


# --- Schemas ---
class PredictRequest(BaseModel):
    """Input features for prediction. All values are from the current day."""
    pm25: float = Field(..., description="Today's PM2.5 (µg/m³)", ge=0)
    pm25_lag_1: float = Field(..., description="Yesterday's PM2.5", ge=0)
    pm25_lag_2: float = Field(..., description="2 days ago PM2.5", ge=0)
    pm25_lag_7: float = Field(..., description="7 days ago PM2.5", ge=0)
    pm25_rolling_3: float = Field(..., description="3-day rolling average", ge=0)
    pm25_rolling_7: float = Field(..., description="7-day rolling average", ge=0)
    day_of_week: int = Field(..., description="Tomorrow's day (0=Mon, 6=Sun)", ge=0, le=6)
    month: int = Field(..., description="Tomorrow's month (1-12)", ge=1, le=12)

    model_config = {"json_schema_extra": {
        "examples": [{
            "pm25": 45.0,
            "pm25_lag_1": 38.0,
            "pm25_lag_2": 42.0,
            "pm25_lag_7": 55.0,
            "pm25_rolling_3": 41.7,
            "pm25_rolling_7": 44.0,
            "day_of_week": 3,
            "month": 1,
        }]
    }}


class PredictResponse(BaseModel):
    city: str
    predicted_pm25: float
    aqi_category: str
    unit: str = "µg/m³"


# --- Helpers ---
def pm25_to_aqi_category(pm25):
    """Convert PM2.5 to EPA AQI health category.

    Based on EPA 24-hour PM2.5 breakpoints.
    We return the category, not the numeric AQI index.
    """
    if pm25 <= 12.0:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# --- Endpoints ---
@app.get("/health")
def health_check():
    """Quick check — is the API running and model loaded?"""
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "city": CITY,
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict next-day PM2.5 for Skopje.

    Send today's air quality features, get tomorrow's prediction
    with the corresponding EPA health category.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Arrange features in the same order the model was trained on
    features = np.array([[
        request.pm25,
        request.pm25_lag_1,
        request.pm25_lag_2,
        request.pm25_lag_7,
        request.pm25_rolling_3,
        request.pm25_rolling_7,
        request.day_of_week,
        request.month,
    ]])

    prediction = float(model.predict(features)[0])
    prediction = round(max(prediction, 0), 1)  # PM2.5 can't be negative

    return PredictResponse(
        city=CITY,
        predicted_pm25=prediction,
        aqi_category=pm25_to_aqi_category(prediction),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
