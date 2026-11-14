
from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"
MODEL_PATH = ROOT / "models" / "best_delivery_model_v2.joblib"
OUTPUT_DIR = ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== ShopSage AI | SHAP Explainability ===")

    # Load data
    df = pd.read_csv(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["order_purchase_timestamp", "is_late"]
    )

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    # Recreate the original chronological 70/15/15 split
    test_start = int(len(df) * 0.85)
    test_df = df.iloc[test_start:].copy()

    X_test = test_df.drop(
        columns=["is_late", "order_purchase_timestamp"]
    )

    # Load trained V2 model
    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Sample test records for SHAP
    sample_size = min(500, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print("\nTransforming test data...")

    transformed = preprocessor.transform(X_sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    print("Sample rows:", transformed.shape[0])
    print("Transformed features:", transformed.shape[1])

    # Calculate SHAP values
    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(transformed)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, -1]

    # SHAP summary plot
    print("\nCreating SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    summary_path = OUTPUT_DIR / "shap_summary_v2.png"

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # SHAP feature importance plot
    print("\nCreating feature importance plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False
    )

    plt.tight_layout()

    importance_path = OUTPUT_DIR / "shap_importance_v2.png"

    plt.savefig(
        importance_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Save numerical feature importance
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance_path_csv = (
        ROOT / "reports" / "shap_feature_importance_v2.csv"
    )

    importance.to_csv(
        importance_path_csv,
        index=False
    )

    print("\nTOP 15 IMPORTANT FEATURES")

    print(
        importance.head(15).to_string(index=False)
    )

    print("\nSHAP ANALYSIS COMPLETED")
    print("Summary plot:", summary_path)
    print("Importance plot:", OUTPUT_DIR / "shap_importance_v2.png")
    print("Importance CSV:", importance_path_csv)


if __name__ == "__main__":
    main()
