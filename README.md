Walbrydge IoT Predictive Maintenance Service
FastAPI • Docker • Machine Learning • SHAP Explainability • NASA Turbofan Sensor Data

Internal-style microservice for predicting upcoming equipment failures in industrial IoT environments.

🚀 Overview

This project simulates a real-world industrial predictive maintenance service used by manufacturing, aviation, and operations engineering teams. It ingests sensor readings from turbofan engines and predicts whether an engine will fail soon.

The service is designed following enterprise engineering patterns, including:

A fully trained ML model (RandomForest)

Data preprocessing pipeline

Scalable FastAPI microservice

Clean visual UI for internal users

Predict + Explainability endpoints

Docker container for easy deployment

Logging for monitoring + drift detection

Health checks for production reliability

This project was created as part of the Walbrydge Operations Platform, representing how modern IoT analytics pipelines are implemented.

🏗 Architecture
                ┌────────────────────┐
                │ NASA Turbofan Data │
                └──────────┬─────────┘
                           ▼
                ┌──────────────────────┐
                │ Feature Engineering  │
                │ Generate RUL labels  │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ ML Training (RF)     │
                │ Save model + scaler  │
                └──────────┬───────────┘
                           ▼
             ┌────────────────────────────┐
             │ FastAPI Inference Service  │
             │  - /predict                │
             │  - /explain (SHAP)         │
             │  - /health                 │
             │  - Custom Landing UI       │
             └──────────┬────────────────┘
                        ▼
             ┌────────────────────────────┐
             │ Dockerized Deployment       │
             └──────────┬────────────────┘
                        ▼
             ┌────────────────────────────┐
             │ Internal Tools / Dashboards │
             └────────────────────────────┘

🖥 Live API Endpoints
Endpoint	Description
GET /	Custom landing page with UI
POST /predict	Send sensor data → returns failure probability
POST /explain	SHAP-based feature importance
GET /health	Health check endpoint
GET /api/docs	Swagger UI
GET /api/openapi.json	OpenAPI schema
📊 Dataset: NASA Turbofan Engine Degradation

FD001 Turbofan Dataset (CMAPSS)

Multiple engines tracked over cycles

21 sensor channels

RUL (Remaining Useful Life) engineered

Binary label generated:

fail_soon = 1 if RUL ≤ threshold

fail_soon = 0 otherwise

This mirrors real predictive maintenance logic used in aviation and manufacturing.

🤖 Machine Learning Pipeline
Model: RandomForestClassifier
Artifacts saved to /models:

model.pkl

scaler.pkl

feature_names.pkl

Feature engineering includes:

Sensor normalization

Cycle-based degradation trends

RUL calculation

Binary classification label

🔍 Explainability (SHAP)

The /explain endpoint returns SHAP values → useful for:

Maintenance engineers

Reliability analysts

Root cause investigations

Building dashboards

🎨 Custom Internal UI

The root endpoint (GET /) has a dark-theme enterprise UI that includes:

✔ Title & descriptions
✔ How the service is used
✔ Sample payload
✔ Endpoint buttons
✔ GitHub link
✔ Internal tags (like real corporate portals)

This gives a professional ML-Platform look.

🐳 Run With Docker
Build image:
docker build -t walbrydge-iot .

Run container:
docker run -p 8000:8000 walbrydge-iot


Open:

Landing Page → http://127.0.0.1:8000/

API Docs → http://127.0.0.1:8000/api/docs

🧪 Local Development (Without Docker)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/feature_engineering.py
python src/train_model.py
uvicorn api.main:app --reload

🔥 Example Prediction Request
{
  "features": {
    "engine_id": 1,
    "cycle": 20,
    "sensor_1": 10.5,
    "sensor_2": 15.7,
    "sensor_3": 30.2
  }
}

Example Response:
{
  "fail_probability": 0.42,
  "fail_soon": 0
}

📝 Logs for Monitoring

Every prediction is logged to:

logs/predictions.jsonl


Used for:

Detecting data drift

Monitoring model inputs

Re-training triggers

Auditing predictions

This is a real enterprise feature.

🛡 Health Check

GET /health returns:

{
  "status": "ok",
  "model_loaded": true
}


Useful for Kubernetes / Docker Swarm / monitoring tools.

🚀 Deployment Notes (Enterprise)

This service can be deployed to:

Azure Container Apps

AWS ECS

Google Cloud Run

Kubernetes

On-prem edge servers

Recommended production setup:

Load balancer in front

Logging to S3 / Azure Blob

Prometheus + Grafana for monitoring

Auto re-training pipeline

Versioned model registry (MLflow)

📈 Scalability Considerations

Stateless microservice → horizontal scaling possible

Docker-ready → fits CI/CD pipelines

Small inference footprint → cheap to run

SHAP calculations can be moved to async workers for speed
