
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"
MODEL_PATH = ROOT / "models" / "best_delivery_model_v2.joblib"
METADATA_PATH = ROOT / "reports" / "training_metadata_v2.json"

OUTPUT_DIR = ROOT / "reports"

print("\n=== ShopSage AI | V2 Prediction Error Analysis ===")

# --------------------------------------------------
# LOAD DATA, MODEL AND METADATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

threshold = float(metadata["prediction_threshold"])
features = metadata["features"]

print(f"\nPrediction threshold: {threshold}")
print(f"Number of model features: {len(features)}")

# --------------------------------------------------
# RECREATE CHRONOLOGICAL TEST SPLIT
# --------------------------------------------------

df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

test_df = df.iloc[validation_end:].copy()

X_test = test_df[features]
y_test = test_df["is_late"].astype(int)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)

results = test_df.copy()
results["actual"] = y_test.to_numpy()
results["predicted"] = predictions
results["predicted_risk"] = probabilities

results["prediction_result"] = np.select(
    [
        (results["actual"] == 1) & (results["predicted"] == 1),
        (results["actual"] == 0) & (results["predicted"] == 1),
        (results["actual"] == 1) & (results["predicted"] == 0),
        (results["actual"] == 0) & (results["predicted"] == 0),
    ],
    [
        "True Positive",
        "False Positive",
        "False Negative",
        "True Negative",
    ],
    default="Unknown",
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("\n1. CONFUSION MATRIX")
print("-------------------")
print(f"True Positives : {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives : {tn}")

print("\n2. CLASSIFICATION REPORT")
print("------------------------")
print(classification_report(
    y_test,
    predictions,
    target_names=["On-time", "Late"],
    zero_division=0
))

# --------------------------------------------------
# MONTHLY ERROR ANALYSIS
# --------------------------------------------------

results["month"] = (
    results["order_purchase_timestamp"].dt.to_period("M").astype(str)
)

monthly = results.groupby("month").agg(
    total_orders=("actual", "size"),
    actual_late=("actual", "sum"),
    predicted_late=("predicted", "sum"),
    true_positive=("prediction_result", lambda x: (x == "True Positive").sum()),
    false_positive=("prediction_result", lambda x: (x == "False Positive").sum()),
    false_negative=("prediction_result", lambda x: (x == "False Negative").sum()),
    true_negative=("prediction_result", lambda x: (x == "True Negative").sum()),
    average_predicted_risk=("predicted_risk", "mean"),
).reset_index()

monthly["actual_late_rate"] = (
    monthly["actual_late"] / monthly["total_orders"]
)

monthly["predicted_late_rate"] = (
    monthly["predicted_late"] / monthly["total_orders"]
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_error_analysis_v2.csv",
    index=False
)

print("\n3. MONTHLY ERROR ANALYSIS")
print("-------------------------")
print(monthly.to_string(index=False))

# --------------------------------------------------
# STATE ERROR ANALYSIS
# --------------------------------------------------

state_col = "customer_state"

if state_col in results.columns:
    state = results.groupby(state_col).agg(
        total_orders=("actual", "size"),
        actual_late=("actual", "sum"),
        predicted_late=("predicted", "sum"),
        true_positive=("prediction_result", lambda x: (x == "True Positive").sum()),
        false_positive=("prediction_result", lambda x: (x == "False Positive").sum()),
        false_negative=("prediction_result", lambda x: (x == "False Negative").sum()),
        average_predicted_risk=("predicted_risk", "mean"),
    ).reset_index()

    state["actual_late_rate"] = (
        state["actual_late"] / state["total_orders"]
    )

    state["predicted_late_rate"] = (
        state["predicted_late"] / state["total_orders"]
    )

    state = state.sort_values(
        "actual_late_rate",
        ascending=False
    )

    state.to_csv(
        OUTPUT_DIR / "state_error_analysis_v2.csv",
        index=False
    )

    print("\n4. STATE ERROR ANALYSIS")
    print("-----------------------")
    print(state.to_string(index=False))

# --------------------------------------------------
# FALSE POSITIVE AND FALSE NEGATIVE EXAMPLES
# --------------------------------------------------

example_columns = [
    "order_id",
    "order_purchase_timestamp",
    "customer_state",
    "actual",
    "predicted",
    "predicted_risk",
    "prediction_result",
]

example_columns = [
    col for col in example_columns if col in results.columns
]

false_negatives = (
    results[results["prediction_result"] == "False Negative"]
    .sort_values("predicted_risk", ascending=False)
    .head(100)
)

false_positives = (
    results[results["prediction_result"] == "False Positive"]
    .sort_values("predicted_risk", ascending=False)
    .head(100)
)

examples = pd.concat(
    [false_negatives, false_positives],
    ignore_index=True
)

examples[example_columns].to_csv(
    OUTPUT_DIR / "error_examples_v2.csv",
    index=False
)

# --------------------------------------------------
# SAVE COMPLETE TEST PREDICTIONS
# --------------------------------------------------

results[example_columns + ["month"]].to_csv(
    OUTPUT_DIR / "test_predictions_v2.csv",
    index=False
)

print("\n5. ERROR EXAMPLES")
print("-----------------")
print(f"False negative examples saved: {len(false_negatives)}")
print(f"False positive examples saved: {len(false_positives)}")

print("\nFiles saved:")
print("- reports/monthly_error_analysis_v2.csv")
print("- reports/state_error_analysis_v2.csv")
print("- reports/error_examples_v2.csv")
print("- reports/test_predictions_v2.csv")

print("\nError analysis completed successfully.")
