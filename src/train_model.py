import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path
import joblib


def train_model():
    print("🔹 Loading processed data...")
    df = pd.read_csv("data/processed/features_train.csv")
    print(f"✅ Data shape: {df.shape}")

    # Target and features
    if "fail_soon" not in df.columns:
        raise ValueError("Column 'fail_soon' not found in processed data.")

    y = df["fail_soon"]
    X = df.drop(columns=["fail_soon", "RUL", "max_cycle"], errors="ignore")

    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target distribution:\n{y.value_counts()}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔹 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save artifacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model, models_dir / "model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(X.columns.tolist(), models_dir / "feature_names.pkl")

    print("✅ Saved model artifacts:")
    print(f"   - {models_dir / 'model.pkl'}")
    print(f"   - {models_dir / 'scaler.pkl'}")
    print(f"   - {models_dir / 'feature_names.pkl'}")


if __name__ == "__main__":
    train_model()
