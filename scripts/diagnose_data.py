
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"
MODEL_PATH = ROOT / "models" / "best_delivery_model.joblib"


def main():
    print("=== ShopSage AI | Model Diagnostics ===")

    df = pd.read_csv(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["order_purchase_timestamp", "is_late"]
    )

    df["is_late"] = df["is_late"].astype(int)

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_test = test_df.drop(
        columns=["is_late", "order_purchase_timestamp"]
    )

    y_test = test_df["is_late"]

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    test_df["predicted_probability"] = probabilities
    test_df["predicted_late"] = predictions
    test_df["month"] = (
        test_df["order_purchase_timestamp"].dt.to_period("M")
    )

    print("\n1. DATE RANGES")

    print(
        "Training:",
        train_df["order_purchase_timestamp"].min(),
        "to",
        train_df["order_purchase_timestamp"].max()
    )

    print(
        "Testing:",
        test_df["order_purchase_timestamp"].min(),
        "to",
        test_df["order_purchase_timestamp"].max()
    )

    print("\n2. LATE-DELIVERY RATE")

    print(
        "Training:",
        round(train_df["is_late"].mean() * 100, 2),
        "%"
    )

    print(
        "Testing:",
        round(test_df["is_late"].mean() * 100, 2),
        "%"
    )

    print("\n3. TEST-SET BASELINE")

    print(
        "Random-ranking PR-AUC baseline:",
        round(y_test.mean(), 4)
    )

    print(
        "Model PR-AUC:",
        round(average_precision_score(y_test, probabilities), 4)
    )

    print(
        "Model ROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 4)
    )

    print("\n4. PREDICTION THRESHOLD = 0.5")

    print(
        "Precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4)
    )

    print(
        "Recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4)
    )

    print(
        "Predicted late orders:",
        int(predictions.sum())
    )

    print("\n5. MONTHLY TEST PERFORMANCE")

    monthly = test_df.groupby("month").agg(
        orders=("is_late", "size"),
        actual_late=("is_late", "sum"),
        actual_late_rate=("is_late", "mean"),
        predicted_late=("predicted_late", "sum"),
        average_risk=("predicted_probability", "mean")
    )

    monthly["actual_late_rate"] *= 100
    monthly["average_risk"] *= 100

    print(
        monthly.round(2).to_string()
    )

    print("\n6. DUPLICATE CHECK")

    feature_columns = [
        column for column in df.columns
        if column not in [
            "order_purchase_timestamp",
            "is_late"
        ]
    ]

    duplicate_count = df.duplicated(
        subset=feature_columns + ["is_late"]
    ).sum()

    print(
        "Rows with identical features and target:",
        int(duplicate_count)
    )

    print("\nDiagnostics completed.")


if __name__ == "__main__":
    main()
