from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.predict import predict_failure
from src.predict import predict_failure, explain_prediction


class SensorInput(BaseModel):
    features: dict


app = FastAPI(
    title="Walbrydge IoT Predictive Maintenance",
    description="Internal service for predicting upcoming equipment failures using NASA turbofan sensor data.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)


@app.get("/", response_class=HTMLResponse)
def landing_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Walbrydge IoT Predictive Maintenance</title>
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #0f172a;
                color: #e5e7eb;
                margin: 0;
                padding: 0;
            }
            .wrapper {
                max-width: 900px;
                margin: 60px auto;
                padding: 24px;
            }
            .card {
                background: #020617;
                border-radius: 16px;
                padding: 24px 28px;
                box-shadow: 0 18px 40px rgba(15,23,42,0.7);
                border: 1px solid #1e293b;
            }
            h1 {
                font-size: 28px;
                margin-bottom: 8px;
            }
            h2 {
                font-size: 18px;
                margin-top: 24px;
                margin-bottom: 8px;
                color: #93c5fd;
            }
            p {
                font-size: 14px;
                line-height: 1.5;
                color: #e5e7eb;
            }
            .pill {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(56,189,248,0.1);
                color: #7dd3fc;
                font-size: 11px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .meta {
                font-size: 12px;
                color: #9ca3af;
                margin-top: 4px;
            }
            .section-list {
                margin: 0;
                padding-left: 18px;
                font-size: 14px;
                color: #cbd5f5;
            }
            .badge-row {
                margin-top: 12px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .badge {
                font-size: 11px;
                padding: 3px 8px;
                border-radius: 999px;
                border: 1px solid #1e293b;
                background: rgba(15,23,42,0.9);
            }
            a {
                color: #7dd3fc;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .link-row {
                margin-top: 18px;
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
            }
            .link-btn {
                font-size: 13px;
                padding: 8px 14px;
                border-radius: 999px;
                border: 1px solid #1d4ed8;
                background: linear-gradient(90deg,#1d4ed8,#0ea5e9);
                color: white;
                text-decoration: none;
            }
            .link-secondary {
                border-color: #334155;
                background: rgba(15,23,42, 0.6);
                color: #e5e7eb;
            }
            pre {
                background: #020617;
                padding: 12px;
                border-radius: 10px;
                font-size: 12px;
                overflow-x: auto;
                border: 1px solid #1e293b;
            }
        </style>
    </head>
    <body>
        <div class="wrapper">
            <div class="card">
                <span class="pill">Walbrydge Operations · Predictive Maintenance</span>
                <h1>IoT Failure Prediction Service</h1>
                <p class="meta">FastAPI · scikit-learn · Docker · NASA Turbofan Dataset</p>

                <p>
                    This internal service exposes a REST API that scores incoming sensor readings
                    from turbofan engines and predicts whether an asset is at risk of failing in the near term.
                    It is designed as a building block for condition-based maintenance workflows at Walbrydge.
                </p>

                <h2>How this service is used</h2>
                <ul class="section-list">
                    <li>Data engineering jobs stream sensor readings from the plant floor.</li>
                    <li>Each observation is transformed into a feature vector matching the trained model.</li>
                    <li>The service returns a failure risk score and a binary decision (<code>fail_soon</code>).</li>
                    <li>Downstream systems can trigger work orders or alerts based on the score.</li>
                </ul>

                <h2>Sample request payload</h2>
                <pre>{
  "features": {
    "engine_id": 1,
    "cycle": 80,
    "sensor_1": 12.3,
    "sensor_2": 8.4,
    "sensor_3": 19.1
    // ...remaining sensor_N features
  }
}</pre>

                <h2>Endpoints</h2>
                <ul class="section-list">
                    <li><code>POST /predict</code> — score a single reading (<code>features: dict</code>).</li>
                    <li><code>GET /api/docs</code> — interactive API docs (Swagger UI) for testing.</li>
                    <li><code>GET /api/openapi.json</code> — OpenAPI schema for client generation.</li>
                </ul>

                <div class="badge-row">
                    <div class="badge">Binary classification</div>
                    <div class="badge">RandomForest</div>
                    <div class="badge">Remaining Useful Life → fail_soon</div>
                    <div class="badge">Feature scaling</div>
                </div>

                <div class="link-row">
                    <a href="/api/docs" class="link-btn">Open API Docs</a>
                    <a href="https://github.com" class="link-btn link-secondary">View Source (GitHub)</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/predict")
def predict(input_data: SensorInput):
    return predict_failure(input_data.features)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/explain")
def explain(input_data: SensorInput):
    return explain_prediction(input_data.features)
