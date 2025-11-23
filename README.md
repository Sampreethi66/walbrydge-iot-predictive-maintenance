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

![Landing Page](images/ui_landing.png)

---

# 📘 **API Documentation (Swagger)**

Swagger UI shows all model endpoints clearly.

![Swagger UI](images/swagger.png)

---

# 🧠 Architecture Diagram
            ┌────────────────────┐
   flowchart LR
    A[Sensors Data<br>NASA Turbofan Dataset] -->|ETL & Cleaning| B(Feature Engineering<br>Scaling + Labeling)
    B -->|Train| C(Model Training<br>RandomForest)
    C --> D(SHAP Explainability Engine)

    C -->|Export| E[Models Folder<br>model.pkl<br>scaler.pkl<br>feature_names.pkl]

    E -->|Load| F(FastAPI Service<br>Custom UI)
    D -->|Explain| F

    F -->|Prediction Logs| G[logs/predictions.jsonl]

    F -->|Served via| H((Docker Container))


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

