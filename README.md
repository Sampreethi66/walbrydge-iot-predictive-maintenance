# 🌩️ Walbrydge IoT Predictive Maintenance Service

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Machine Learning](https://img.shields.io/badge/ML-Model-orange)
![NASA Dataset](https://img.shields.io/badge/Dataset-NASA_Turbofan-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)

---

## 🔥 Project Overview

The **Walbrydge IoT Predictive Maintenance Service** is a production-ready machine learning microservice designed to predict turbofan engine failures using real IoT sensor data.

This project simulates what Fortune 100 industrial companies (GE, Boeing, Honeywell, Rolls Royce, Siemens) use in real predictive maintenance pipelines.

**The system includes:**
- A trained RandomForest failure prediction model  
- A polished FastAPI service  
- Enterprise-style custom UI  
- SHAP explainability  
- Dockerized deployment  
- Logging for drift monitoring  
- Health checking  
- Professional folder structure  

---

# 🖥️ **Landing Page UI**

A custom-built internal dashboard-style UI for engineers.

1. Docker – Container Running
![Docker Container Running](images/Docker%20Run%20+%20Container%20Running.png)

2. Docker Build Output
![Docker Build](images/Docker%20Build.png)

3. Logs Updating (Prediction Log)
![Logs Updating](images/Logs%20Updating%20(prediction%20log).png)

4. Feature Names / Values Array
![Values array](images/Values%20array.png)

🧠 Explainability (SHAP)
This endpoint helps reliability engineers understand feature contributions.

6. SHAP Explainability Output
![Values array](images/Values%20array.png)

7. Successful Prediction
![Successful Prediction](images/Successful%20Prediction.png)

8. Swagger – Predict Endpoint Expanded
![Swagger Predict Endpoint](images/Swagger%20Predict%20Endpoint%20Expanded.png)

9. Swagger API Docs
![Swagger API Docs](images/Swagger%20API%20Docs.png)

10. FastAPI Custom Landing Page
![Walbrydge IoT Predictive Maintenance Service](images/Walbrydge%20IoT%20Predictive%20Maintenance%20Service.png)

---

# 📘 **API Documentation (Swagger)**

Swagger UI shows all model endpoints clearly.

![Swagger UI](images/swagger.png)

---

# 🧠 Architecture Diagram
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


---

# 📡 **Endpoints Overview**

| Endpoint | Description |
|---------|-------------|
| `GET /` | Custom UI home page |
| `POST /predict` | Predict failure probability |
| `POST /explain` | SHAP explainability |
| `GET /health` | Health status |
| `GET /api/docs` | Swagger Docs |
| `GET /api/openapi.json` | OpenAPI Schema |

---

# 🔮 **Prediction Example**

### Swagger Input
![Swagger Predict](images/swagger_predict.png)

### Successful Prediction Output
![Predict Result](images/predict_result.png)

---

# 🧠 **Explainability (SHAP)**

This endpoint helps reliability engineers understand feature contributions.

![Explain SHAP](images/shap_explain.png)

---

# 📊 **Prediction Logs (For Drift Monitoring)**

Every inference request & its result is logged.

![Logs](images/logs.png)

---

# 🧪 **Local Development**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/feature_engineering.py
python src/train_model.py
uvicorn api.main:app --reload
---

Open:

UI → http://127.0.0.1:8000

Docs → http://127.0.0.1:8000/api/docs

🐳 Run With Docker
Build
docker build -t walbrydge-iot .

----
Run
docker run -p 8000:8000 walbrydge-iot

----
🧱 Project Structure
walbrydge-iot-predictive-maintenance/
│
├── api/
│   └── main.py
├── src/
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── train_model.py
│   └── test_request.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── logs/
│   └── predictions.jsonl
├── images/
├── notebooks/
├── requirements.txt
├── Dockerfile
└── README.md
---

🤖 ML Pipeline
Step 1 — Feature Engineering

Load NASA sensor data

Calculate RUL

Generate binary classification labels

Step 2 — Train RandomForest

Save model artifacts

Save preprocessing scaler

Save feature names

Step 3 — Serve Model via FastAPI

Scale inputs

Predict

Log results

Return explanation

🛡 Health Check Endpoint

GET /health returns:

{
  "status": "ok",
  "model_loaded": true,
  "environment": "docker"
}

---

