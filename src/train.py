import pandas as pd
import yaml
import pickle

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = Path("data/processed/processed.csv")
MODEL_PATH = Path("models/model.pkl")
PARAMS_PATH = Path("params.yaml")


def train_model():

    print("Loading parameters...")

    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]

    print("\nModel parameters:")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"test_size: {test_size}")
    print(f"random_state: {random_state}")

    print("\nLoading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    target_column = "Delivery_Time_min"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    print("\nTraining Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(MODEL_PATH, "wb") as file:

        pickle.dump(
            {
                "model": model,
                "features": X.columns.tolist(),
                "X_test": X_test,
                "y_test": y_test
            },
            file
        )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    train_model()