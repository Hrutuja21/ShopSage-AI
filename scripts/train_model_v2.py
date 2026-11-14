
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "brier_score": float(brier_score_loss(
            y_true, probabilities
        )),
    }


def main():
    print("=== ShopSage AI | Improved Model Evaluation ===")

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

    # Chronological 70 / 15 / 15 split
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    print("\nSplit sizes:")
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    for name, part in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df),
    ]:
        print(
            f"{name} late-delivery rate: "
            f"{part['is_late'].mean() * 100:.2f}%"
        )
        print(
            f"{name} dates: "
            f"{part['order_purchase_timestamp'].min()} to "
            f"{part['order_purchase_timestamp'].max()}"
        )

    excluded = ["is_late", "order_purchase_timestamp"]

    X_train = train_df.drop(columns=excluded)
    y_train = train_df["is_late"]

    X_val = val_df.drop(columns=excluded)
    y_val = val_df["is_late"]

    X_test = test_df.drop(columns=excluded)
    y_test = test_df["is_late"]

    categorical = X_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    numerical = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numerical,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])

    positive = int(y_train.sum())
    negative = len(y_train) - positive
    weight = negative / max(positive, 1)

    configurations = {
        "XGBoost Unweighted": 1.0,
        "XGBoost Weighted": weight,
    }

    trained = {}
    validation_results = []

    # Train and compare using validation data only
    for name, class_weight in configurations.items():
        print(f"\nTraining {name}...")

        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        val_prob = pipeline.predict_proba(X_val)[:, 1]

        val_metrics = evaluate(y_val, val_prob, 0.5)

        validation_results.append({
            "model": name,
            **val_metrics,
        })

        trained[name] = pipeline

        print("Validation PR-AUC:", round(val_metrics["pr_auc"], 4))
        print("Validation ROC-AUC:", round(val_metrics["roc_auc"], 4))

    results_df = pd.DataFrame(validation_results)

    print("\n=== VALIDATION COMPARISON ===")
    print(results_df.to_string(index=False))

    # Select using validation PR-AUC only
    best_name = results_df.sort_values(
        "pr_auc", ascending=False
    ).iloc[0]["model"]

    best_model = trained[best_name]

    print("\nSelected model:", best_name)

    # Choose threshold using validation F1 only
    val_prob = best_model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)

    threshold_scores = [
        f1_score(
            y_val,
            (val_prob >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_threshold = float(
        thresholds[int(np.argmax(threshold_scores))]
    )

    print("Selected validation threshold:", round(best_threshold, 2))

    # Evaluate final model once on untouched test data
    test_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate(
        y_test,
        test_prob,
        best_threshold,
    )

    test_predictions = (
        test_prob >= best_threshold
    ).astype(int)

    print("\n=== FINAL TEST RESULTS ===")

    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    ))

    print(
        "\nRandom-ranking PR-AUC baseline:",
        round(float(y_test.mean()), 4),
    )

    # Save artifacts
    model_path = MODEL_DIR / "best_delivery_model_v2.joblib"
    joblib.dump(best_model, model_path)

    results_df.to_csv(
        REPORT_DIR / "validation_comparison_v2.csv",
        index=False,
    )

    metadata = {
        "best_model": best_name,
        "selection_metric": "Validation PR-AUC",
        "threshold_selection": "Maximum validation F1",
        "prediction_threshold": best_threshold,
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "testing_rows": len(X_test),
        "test_metrics": test_metrics,
        "features": X_train.columns.tolist(),
    }

    with open(
        REPORT_DIR / "training_metadata_v2.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining and evaluation completed.")
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
