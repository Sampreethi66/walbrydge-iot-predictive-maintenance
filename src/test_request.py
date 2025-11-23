import requests

payload = {
    "features": {
        "engine_id": 1,
        "cycle": 80,
        "sensor_1": 12.3,
        "sensor_2": 8.4,
        "sensor_3": 19.1
    }
}

try:
    resp = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)
    print("Status:", resp.status_code)
    print("Response:", resp.json())
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to API. Is Docker container or uvicorn running on port 8000?")
