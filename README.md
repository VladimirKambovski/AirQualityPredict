# AirQualityPredict

Small end-to-end ML project that predicts next-day PM2.5 in Skopje.

I built this to practice the full pipeline on one clear use case:
- collect data
- preprocess + feature engineering
- train/evaluate a regression model
- serve predictions through a FastAPI endpoint

## What this project does

- Uses OpenAQ hourly PM2.5 measurements (or generated sample data for offline runs)
- Aggregates to daily values
- Builds lag and rolling-average features
- Trains a Random Forest regressor
- Exposes `POST /predict` for next-day PM2.5

## Quick start

```bash
pip install -r requirements.txt
```

```bash
# Option A (recommended first run): no internet/API dependency
python collect_data.py --sample

# Option B: pull real data from OpenAQ
python collect_data.py
```

```bash
python preprocess.py
python train.py
python app.py
```

API docs: `http://localhost:8000/docs`

## Example request

Windows PowerShell:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"pm25\": 45.0, \"pm25_lag_1\": 38.0, \"pm25_lag_2\": 42.0, \"pm25_lag_7\": 55.0, \"pm25_rolling_3\": 41.7, \"pm25_rolling_7\": 44.0, \"day_of_week\": 3, \"month\": 1}"
```

Example response:

```json
{
  "city": "Skopje",
  "predicted_pm25": 42.4,
  "aqi_category": "Unhealthy for Sensitive Groups",
  "unit": "µg/m³"
}
```

## Local Docker run

```bash
docker build -t airqualitypredict .
docker run -p 8000:8000 airqualitypredict
```

## Project layout

```
collect_data.py      # Fetch raw data from OpenAQ or generate sample data
preprocess.py        # Build model-ready features
train.py             # Train, evaluate, and save model.joblib
app.py               # FastAPI inference service
config.py            # Shared constants/paths
models/              # Saved model artifact
data/raw/            # Raw measurements
data/processed/      # Processed training dataset
```

## Model notes

- Model: Random Forest Regressor (100 trees)
- Target: next-day PM2.5 (`pm25_next_day`)
- Split: chronological 80/20 (train on past, test on future)
- Metrics printed in training: RMSE, MAE, R²
