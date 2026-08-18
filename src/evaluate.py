import json
import pickle

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics/metrics.json")


def evaluate_model():

    print("Loading trained model...")

    with open(MODEL_PATH, "rb") as file:
        saved_data = pickle.load(file)

    model = saved_data["model"]
    X_test = saved_data["X_test"]
    y_test = saved_data["y_test"]

    print("Generating predictions...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(METRICS_PATH, "w") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\n============================")
    print("       MODEL EVALUATION")
    print("============================")

    print(f"MAE  : {metrics['mae']}")
    print(f"RMSE : {metrics['rmse']}")
    print(f"R²   : {metrics['r2']}")

    print("============================")

    print(
        f"\nMetrics saved to: {METRICS_PATH}"
    )


if __name__ == "__main__":
    evaluate_model()