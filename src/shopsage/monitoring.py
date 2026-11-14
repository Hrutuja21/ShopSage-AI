
"""Prediction logging and monitoring utilities for ShopSage AI."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / "reports" / "prediction_logs.csv"


def log_prediction(
    features: dict,
    result: dict,
) -> str:
    """Save a prediction and its model output for monitoring."""

    if not isinstance(features, dict):
        raise TypeError("features must be a dictionary.")

    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary.")

    required_result_fields = [
        "risk_score",
        "threshold",
        "is_flagged",
        "prediction",
    ]

    missing = [
        field for field in required_result_fields
        if field not in result
    ]

    if missing:
        raise ValueError(f"Missing prediction fields: {missing}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    prediction_id = str(uuid4())

    record = {
        "prediction_id": prediction_id,
        "prediction_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_version": "v2",
        **features,
        "risk_score": float(result["risk_score"]),
        "threshold": float(result["threshold"]),
        "is_flagged": bool(result["is_flagged"]),
        "prediction": result["prediction"],
        "actual_is_late": None,
    }

    new_record = pd.DataFrame([record])

    if LOG_PATH.exists():
        new_record.to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_record.to_csv(
            LOG_PATH,
            index=False,
        )

    return prediction_id


def load_prediction_logs() -> pd.DataFrame:
    """Load saved prediction records."""

    if not LOG_PATH.exists():
        return pd.DataFrame()

    logs = pd.read_csv(LOG_PATH)

    if "prediction_timestamp" in logs.columns:
        logs["prediction_timestamp"] = pd.to_datetime(
            logs["prediction_timestamp"],
            errors="coerce",
            utc=True,
        )

    if "risk_score" in logs.columns:
        logs["risk_score"] = pd.to_numeric(
            logs["risk_score"],
            errors="coerce",
        )

    if "is_flagged" in logs.columns:
        logs["is_flagged"] = (
            logs["is_flagged"]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    if "actual_is_late" in logs.columns:
        logs["actual_is_late"] = pd.to_numeric(
            logs["actual_is_late"],
            errors="coerce",
        )

    return logs


def update_actual_outcome(
    prediction_id: str,
    actual_is_late: int,
) -> None:
    """Update the actual delivery outcome for a prediction."""

    if actual_is_late not in (0, 1):
        raise ValueError(
            "Actual outcome must be 0 (On Time) or 1 (Late)."
        )

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "Prediction log does not exist."
        )

    logs = pd.read_csv(LOG_PATH)

    if "prediction_id" not in logs.columns:
        raise ValueError(
            "Prediction ID column is missing."
        )

    if "actual_is_late" not in logs.columns:
        raise ValueError(
            "Actual outcome column is missing."
        )

    matching_rows = (
        logs["prediction_id"].astype(str) == str(prediction_id)
    )

    if not matching_rows.any():
        raise ValueError(
            "Prediction ID not found."
        )

    if logs.loc[
        matching_rows, "actual_is_late"
    ].notna().any():
        raise ValueError(
            "Actual outcome has already been recorded."
        )

    logs.loc[
        matching_rows, "actual_is_late"
    ] = actual_is_late

    logs.to_csv(
        LOG_PATH,
        index=False,
    )
