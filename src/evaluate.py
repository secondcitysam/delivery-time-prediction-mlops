import json
import pickle

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics/metrics.json")


def classify_delivery_time(time):

    if time <= 30:
        return "Fast"

    elif time <= 45:
        return "Normal"

    else:
        return "Slow"


def evaluate_model():

    print("Loading trained model...")

    with open(MODEL_PATH, "rb") as file:
        saved_data = pickle.load(file)

    model = saved_data["model"]
    X_test = saved_data["X_test"]
    y_test = saved_data["y_test"]

    print("Generating predictions...")

    predictions = model.predict(X_test)

    # ==========================================
    # REGRESSION METRICS
    # ==========================================

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

    # ==========================================
    # CLASSIFICATION CONVERSION
    # ==========================================

    actual_categories = [
        classify_delivery_time(value)
        for value in y_test
    ]

    predicted_categories = [
        classify_delivery_time(value)
        for value in predictions
    ]

    # ==========================================
    # CLASSIFICATION METRICS
    # ==========================================

    accuracy = accuracy_score(
        actual_categories,
        predicted_categories
    )

    precision = precision_score(
        actual_categories,
        predicted_categories,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual_categories,
        predicted_categories,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual_categories,
        predicted_categories,
        average="weighted",
        zero_division=0
    )

    # ==========================================
    # STORE ALL METRICS
    # ==========================================

    metrics = {

        # Regression
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),

        # Classification
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
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

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    print("\n================================")
    print("       MODEL EVALUATION")
    print("================================")

    print("\nRegression Metrics")
    print("------------------")

    print(f"MAE  : {metrics['mae']}")
    print(f"RMSE : {metrics['rmse']}")
    print(f"R²   : {metrics['r2']}")

    print("\nClassification Metrics")
    print("----------------------")

    print(f"Accuracy  : {metrics['accuracy']}")
    print(f"Precision : {metrics['precision']}")
    print(f"Recall    : {metrics['recall']}")
    print(f"F1 Score  : {metrics['f1_score']}")

    print("================================")

    print(
        f"\nMetrics saved to: {METRICS_PATH}"
    )


if __name__ == "__main__":
    evaluate_model()