
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=== ShopSage AI | Model Training ===")

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run scripts.build_features first."
        )

    df = pd.read_csv(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["order_purchase_timestamp", "is_late"]
    )

    df["is_late"] = df["is_late"].astype(int)

    print("\nDataset shape:", df.shape)

    # --------------------------------------------------
    # 2. CHRONOLOGICAL TRAIN / TEST SPLIT
    # --------------------------------------------------

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Raw timestamp is used for ordering, not as a feature.
    X_train = train_df.drop(
        columns=["is_late", "order_purchase_timestamp"]
    )

    y_train = train_df["is_late"]

    X_test = test_df.drop(
        columns=["is_late", "order_purchase_timestamp"]
    )

    y_test = test_df["is_late"]

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both chronological splits must contain both target classes."
        )

    # --------------------------------------------------
    # 3. IDENTIFY FEATURE TYPES
    # --------------------------------------------------

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("\nNumerical features:", numerical_features)
    print("\nCategorical features:", categorical_features)

    # --------------------------------------------------
    # 4. PREPROCESSING
    # --------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------
    # 5. DEFINE MODELS
    # --------------------------------------------------

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    scale_pos_weight = negative_count / max(
        positive_count, 1
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    # --------------------------------------------------
    # 6. TRAIN AND EVALUATE
    # --------------------------------------------------

    for model_name, classifier in models.items():

        print("\n" + "=" * 50)
        print("Training:", model_name)
        print("=" * 50)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": model_name,
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "pr_auc": float(
                average_precision_score(
                    y_test,
                    probabilities
                )
            )
        }

        results.append(metrics)
        trained_models[model_name] = pipeline

        print("\nEvaluation metrics:")

        for metric, value in metrics.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, predictions))

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

    # --------------------------------------------------
    # 7. COMPARE MODELS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n=== MODEL COMPARISON ===")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Select the model with the highest PR-AUC.
    best_model_name = results_df.sort_values(
        "pr_auc",
        ascending=False
    ).iloc[0]["model"]

    best_model = trained_models[best_model_name]

    model_path = MODEL_DIR / "best_delivery_model.joblib"
    joblib.dump(best_model, model_path)

    # --------------------------------------------------
    # 8. SAVE METADATA
    # --------------------------------------------------

    metadata = {
        "best_model": best_model_name,
        "selection_metric": "PR-AUC",
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": X_train.columns.tolist(),
        "metrics": results,
        "prediction_threshold": 0.5
    }

    with open(
        REPORT_DIR / "training_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    # --------------------------------------------------
    # 9. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED")
    print("=" * 50)

    print("Best model:", best_model_name)
    print("Selection metric: PR-AUC")
    print("Saved model:", model_path)
    print("Comparison report:", results_path)

    print("\nNext: Explain the model using SHAP.")


if __name__ == "__main__":
    main()
