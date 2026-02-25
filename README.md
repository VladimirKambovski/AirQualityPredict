# AirQualityPredict

End-to-end ML pipeline that predicts next-day PM2.5 air quality for Skopje, Macedonia.

Covers the full lifecycle: **data collection → preprocessing → training → evaluation → API deployment**.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect data

```bash
# Option A: Generate sample data (no API needed — works offline)
python collect_data.py --sample

# Option B: Fetch real data from OpenAQ
python collect_data.py
```

### 3. Preprocess

```bash
python preprocess.py
```

### 4. Train and evaluate

```bash
python train.py
```

### 5. Start the API

```bash
python app.py
```

API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 6. Make a prediction

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"pm25\": 45.0, \"pm25_lag_1\": 38.0, \"pm25_lag_2\": 42.0, \"pm25_lag_7\": 55.0, \"pm25_rolling_3\": 41.7, \"pm25_rolling_7\": 44.0, \"day_of_week\": 3, \"month\": 1}"
```

Response:

```json
{
  "city": "Skopje",
  "predicted_pm25": 43.2,
  "aqi_category": "Unhealthy for Sensitive Groups",
  "unit": "µg/m³"
}
```

## Docker

```bash
docker build -t airqualitypredict .
docker run -p 8000:8000 airqualitypredict
```

## AWS Deployment (App Runner)

Simplest path: push a Docker image, get a public URL.

### 1. Create ECR repository

```bash
aws ecr create-repository --repository-name airqualitypredict --region us-east-1
```

### 2. Build, tag, and push

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

docker tag airqualitypredict:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/airqualitypredict:latest

docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/airqualitypredict:latest
```

### 3. Create App Runner service

In AWS Console → App Runner → Create service:
- Source: Container registry (ECR)
- Select your image
- Port: 8000
- Deploy

You'll get a public HTTPS URL in about 2 minutes.

**Alternative:** For zero-traffic cost savings, use Lambda + API Gateway with the same Docker image.

## Project Structure

```
├── collect_data.py     # Fetch data from OpenAQ API (or generate sample)
├── preprocess.py       # Clean data + engineer features
├── train.py            # Train model, evaluate, save
├── app.py              # FastAPI inference endpoint
├── config.py           # All constants and paths
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
├── WALKTHROUGH.md      # Learning journey + interview prep
├── models/             # Saved model (created by train.py)
└── data/               # Raw and processed data (created by pipeline)
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data handling |
| Scikit-learn | Random Forest model |
| FastAPI | Inference API |
| Docker | Containerization |
| AWS App Runner | Deployment |

## Model Details

- **Algorithm:** Random Forest Regressor (100 trees)
- **Target:** Next-day PM2.5 concentration (µg/m³)
- **Features:** Lag values, rolling averages, day of week, month
- **Train/test split:** Chronological 80/20 (no data leakage)
- **Key insight:** Yesterday's PM2.5 is the strongest predictor — air quality is driven by multi-day weather patterns
