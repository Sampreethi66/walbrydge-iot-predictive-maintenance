import pandas as pd
from pathlib import Path

def create_features(input_path, output_path, threshold=30):
    df = pd.read_csv(input_path)

    # Assign column names
    num_sensors = df.shape[1] - 2
    df.columns = ["engine_id", "cycle"] + [f"sensor_{i}" for i in range(1, num_sensors+1)]

    # Compute RUL
    rul = df.groupby("engine_id")["cycle"].max().reset_index()
    rul.columns = ["engine_id", "max_cycle"]

    df = df.merge(rul, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["fail_soon"] = (df["RUL"] <= threshold).astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved processed features to {output_path}")

if __name__ == "__main__":
    create_features(
        "data/raw/nasa_turbofan_train.csv",
        "data/processed/features_train.csv"
    )
