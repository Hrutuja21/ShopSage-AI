
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
    }


def create_pipeline(X_train, class_weight):
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

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def find_best_threshold(y_true, probabilities):
    thresholds = np.arange(0.05, 0.96, 0.01)

    scores = [
        f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    return float(thresholds[int(np.argmax(scores))])


def main():
    print("=== ShopSage AI | Location Feature Ablation ===")

    # Load and prepare dataset
    df = pd.read_csv(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["order_purchase_timestamp", "is_late"]
    )

    df["is_late"] = df["is_late"].astype(int)

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    # Same chronological 70 / 15 / 15 split as V2
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

    excluded = [
        "is_late",
        "order_purchase_timestamp",
    ]

    X_train_full = train_df.drop(columns=excluded)
    X_val_full = val_df.drop(columns=excluded)
    X_test_full = test_df.drop(columns=excluded)

    y_train = train_df["is_late"]
    y_val = val_df["is_late"]
    y_test = test_df["is_late"]

    # Calculate class imbalance from training data only
    positive = int(y_train.sum())
    negative = len(y_train) - positive

    class_weight = negative / max(positive, 1)

    configurations = {
        "Unweighted": 1.0,
        "Weighted": class_weight,
    }

    experiments = {
        "With Location": {
            "train": X_train_full,
            "validation": X_val_full,
            "test": X_test_full,
        },
        "Without Location": {
            "train": X_train_full.drop(
                columns=["customer_state"]
            ),
            "validation": X_val_full.drop(
                columns=["customer_state"]
            ),
            "test": X_test_full.drop(
                columns=["customer_state"]
            ),
        },
    }

    all_results = []
    selected_models = {}

    # Train both feature variants
    for experiment_name, data in experiments.items():

        print(f"\n{'=' * 60}")
        print(experiment_name.upper())
        print(f"{'=' * 60}")

        trained = []

        for config_name, weight in configurations.items():

            print(f"\nTraining {config_name} XGBoost...")

            model = create_pipeline(
                data["train"],
                weight,
            )

            model.fit(data["train"], y_train)

            val_prob = model.predict_proba(
                data["validation"]
            )[:, 1]

            val_pr_auc = average_precision_score(
                y_val,
                val_prob,
            )

            val_roc_auc = roc_auc_score(
                y_val,
                val_prob,
            )

            print("Validation PR-AUC:", round(val_pr_auc, 4))
            print("Validation ROC-AUC:", round(val_roc_auc, 4))

            trained.append({
                "model": model,
                "configuration": config_name,
                "validation_pr_auc": float(val_pr_auc),
                "validation_roc_auc": float(val_roc_auc),
            })

        # Select the best configuration using validation PR-AUC
        best = max(
            trained,
            key=lambda item: item["validation_pr_auc"],
        )

        best_model = best["model"]

        print("\nSelected configuration:", best["configuration"])

        # Select threshold using validation F1
        val_prob = best_model.predict_proba(
            data["validation"]
        )[:, 1]

        best_threshold = find_best_threshold(
            y_val,
            val_prob,
        )

        print("Selected threshold:", round(best_threshold, 2))

        # Evaluate on untouched test data
        test_prob = best_model.predict_proba(
            data["test"]
        )[:, 1]

        test_metrics = evaluate(
            y_test,
            test_prob,
            best_threshold,
        )

        print("\nFinal test results:")

        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        all_results.append({
            "experiment": experiment_name,
            "selected_configuration": best["configuration"],
            "threshold": best_threshold,
            **test_metrics,
        })

        selected_models[experiment_name] = best_model

        if experiment_name == "Without Location":
            model_path = MODEL_DIR / "best_delivery_model_no_state.joblib"
            joblib.dump(best_model, model_path)
            print("\nSaved experimental model:", model_path)

    # Compare final results
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("FINAL LOCATION ABLATION COMPARISON")
    print("=" * 70)

    print(results_df.to_string(index=False))

    # Save comparison report
    report_path = REPORT_DIR / "location_ablation_comparison.csv"

    results_df.to_csv(
        report_path,
        index=False,
    )

    # Save experiment metadata
    metadata = {
        "experiment": "Customer state feature ablation",
        "split": "Chronological 70/15/15",
        "selection_metric": "Validation PR-AUC",
        "threshold_selection": "Maximum validation F1",
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "testing_rows": len(test_df),
        "location_feature": "customer_state",
        "results": results_df.to_dict(orient="records"),
    }

    metadata_path = REPORT_DIR / "location_ablation_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nComparison report:", report_path)
    print("Experiment metadata:", metadata_path)
    print("\nLocation ablation experiment completed.")


if __name__ == "__main__":
    main()
