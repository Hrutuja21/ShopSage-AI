from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"
MODEL_PATH = ROOT / "models" / "best_delivery_model_v2.joblib"

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

    return {
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities)
        ),
        "mean_predicted_risk": float(
            probabilities.mean()
        ),
        "actual_late_rate": float(
            np.mean(y_true)
        ),
    }


def create_risk_groups(y_true, probabilities, model_name):
    results = pd.DataFrame({
        "actual": np.asarray(y_true),
        "predicted_risk": np.asarray(probabilities),
    })

    results["risk_group"] = pd.qcut(
        results["predicted_risk"],
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1

    summary = (
        results.groupby("risk_group", observed=True)
        .agg(
            orders=("actual", "count"),
            average_predicted_risk=("predicted_risk", "mean"),
            actual_late_rate=("actual", "mean"),
        )
        .reset_index()
    )

    summary["model"] = model_name

    summary["average_predicted_risk"] *= 100
    summary["actual_late_rate"] *= 100

    return summary


def main():
    print("=== ShopSage AI | Probability Calibration ===")

    # Load dataset
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

    # Reproduce exact chronological 70/15/15 split
    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    val_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    excluded = [
        "is_late",
        "order_purchase_timestamp",
    ]

    X_val = val_df.drop(columns=excluded)
    y_val = val_df["is_late"].to_numpy()

    X_test = test_df.drop(columns=excluded)
    y_test = test_df["is_late"].to_numpy()

    print("\nValidation rows:", len(X_val))
    print("Test rows:", len(X_test))

    # Load the existing V2 model
    model = joblib.load(MODEL_PATH)

    # Get predictions from the existing model
    print("\nGenerating validation predictions...")

    val_prob = model.predict_proba(X_val)[:, 1]

    print("Generating test predictions...")

    test_prob = model.predict_proba(X_test)[:, 1]

    # Fit Platt scaling using validation predictions ONLY
    print("\nFitting probability calibrator on validation data...")

    val_logits = np.log(
        np.clip(val_prob, 1e-7, 1 - 1e-7)
        / (1 - np.clip(val_prob, 1e-7, 1 - 1e-7))
    ).reshape(-1, 1)

    test_logits = np.log(
        np.clip(test_prob, 1e-7, 1 - 1e-7)
        / (1 - np.clip(test_prob, 1e-7, 1 - 1e-7))
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(val_logits, y_val)

    # Apply calibrator to untouched test predictions
    calibrated_test_prob = calibrator.predict_proba(
        test_logits
    )[:, 1]

    # Evaluate both on the same test set
    original_metrics = calculate_metrics(
        y_test,
        test_prob,
    )

    calibrated_metrics = calculate_metrics(
        y_test,
        calibrated_test_prob,
    )

    comparison = pd.DataFrame([
        {
            "model": "Original V2",
            **original_metrics,
        },
        {
            "model": "Platt Calibrated V2",
            **calibrated_metrics,
        },
    ])

    print("\n=== TEST CALIBRATION COMPARISON ===")
    print(comparison.to_string(index=False))

    # Calibration curves on the test set
    original_fraction, original_mean = calibration_curve(
        y_test,
        test_prob,
        n_bins=10,
        strategy="quantile",
    )

    calibrated_fraction, calibrated_mean = calibration_curve(
        y_test,
        calibrated_test_prob,
        n_bins=10,
        strategy="quantile",
    )

    plt.figure(figsize=(9, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        original_mean,
        original_fraction,
        marker="o",
        label="Original V2",
    )

    plt.plot(
        calibrated_mean,
        calibrated_fraction,
        marker="s",
        label="Platt Calibrated V2",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed late-delivery rate")
    plt.title("ShopSage AI | Probability Calibration")
    plt.legend()
    plt.grid(alpha=0.3)

    calibration_path = (
        FIGURE_DIR / "calibration_comparison_v2.png"
    )

    plt.tight_layout()
    plt.savefig(
        calibration_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Risk group comparison
    original_groups = create_risk_groups(
        y_test,
        test_prob,
        "Original V2",
    )

    calibrated_groups = create_risk_groups(
        y_test,
        calibrated_test_prob,
        "Platt Calibrated V2",
    )

    groups = pd.concat(
        [original_groups, calibrated_groups],
        ignore_index=True,
    )

    groups_path = REPORT_DIR / "calibration_risk_groups_v2.csv"
    groups.to_csv(groups_path, index=False)

    # Save comparison metrics
    metrics_path = REPORT_DIR / "calibration_metrics_v2.csv"

    comparison.to_csv(
        metrics_path,
        index=False,
    )

    # Save calibrator separately; original model remains unchanged
    calibrator_path = (
        ROOT / "models" / "delivery_probability_calibrator_v2.joblib"
    )

    joblib.dump(calibrator, calibrator_path)

    print("\n=== RISK GROUP ANALYSIS ===")
    print(groups.to_string(index=False))

    print("\nCalibration completed.")
    print("Calibration plot:", calibration_path)
    print("Metrics:", metrics_path)
    print("Risk groups:", groups_path)
    print("Saved calibrator:", calibrator_path)

    print("\nOriginal V2 model was not modified.")


if __name__ == "__main__":
    main()