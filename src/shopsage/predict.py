
"""Prediction service for the ShopSage AI delivery-risk model."""

from functools import lru_cache
from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_delivery_model_v2.joblib"
METADATA_PATH = PROJECT_ROOT / "reports" / "training_metadata_v2.json"


@lru_cache(maxsize=1)
def load_prediction_assets():
    """Load the trained model and its training metadata once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_features = metadata.get("features", [])
    threshold = float(metadata.get("prediction_threshold", 0.1))

    if not required_features:
        raise ValueError("No model features found in metadata.")

    if not 0 < threshold < 1:
        raise ValueError(f"Invalid prediction threshold: {threshold}")

    model_features = list(model.feature_names_in_)

    if model_features != required_features:
        raise ValueError(
            "Model feature order does not match training metadata."
        )

    return model, metadata


def prepare_input(order_data: dict) -> pd.DataFrame:
    """Validate and arrange one order's features."""

    if not isinstance(order_data, dict):
        raise TypeError("Order input must be a dictionary.")

    model, metadata = load_prediction_assets()
    required_features = metadata["features"]

    missing = [
        feature for feature in required_features
        if feature not in order_data
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    # Ignore additional keys and preserve exact training order.
    row = {
        feature: order_data[feature]
        for feature in required_features
    }

    return pd.DataFrame([row], columns=required_features)


def predict_order(order_data: dict) -> dict:
    """
    Predict delivery risk for one order.

    The returned score is a model-generated ranking score.
    It should not be interpreted as a perfectly calibrated probability.
    """

    model, metadata = load_prediction_assets()
    threshold = float(metadata["prediction_threshold"])

    input_df = prepare_input(order_data)

    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The loaded model does not contain class 1.")

    late_index = classes.index(1)
    score = float(probabilities[late_index])

    return {
        "risk_score": score,
        "threshold": threshold,
        "is_flagged": bool(score >= threshold),
        "prediction": "Review for late-delivery risk"
        if score >= threshold
        else "Not flagged by the current threshold",
    }
