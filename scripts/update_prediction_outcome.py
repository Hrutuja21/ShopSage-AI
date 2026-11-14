
"""Update actual delivery outcomes for logged ShopSage predictions."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "reports" / "prediction_logs.csv"


def main():
    if not LOG_PATH.exists():
        print("No prediction logs found. Make a prediction in the dashboard first.")
        return

    logs = pd.read_csv(LOG_PATH)

    required_columns = {"prediction_id", "actual_is_late"}

    if not required_columns.issubset(logs.columns):
        print("Prediction log is missing required columns.")
        return

    pending = logs[logs["actual_is_late"].isna()]

    if pending.empty:
        print("No predictions are waiting for actual outcomes.")
        return

    print("\n=== ShopSage AI | Outcome Tracking ===")
    print(f"\nPending predictions: {len(pending)}")

    display_columns = [
        column for column in [
            "prediction_id",
            "prediction_timestamp",
            "risk_score",
            "is_flagged",
        ]
        if column in pending.columns
    ]

    print("\nPending prediction records:")
    print(pending[display_columns].to_string(index=False))

    prediction_id = input(
        "\nEnter the Prediction ID to update: "
    ).strip()

    matching = logs["prediction_id"].astype(str) == prediction_id

    if not matching.any():
        print("Prediction ID not found.")
        return

    if logs.loc[matching, "actual_is_late"].notna().any():
        print("This prediction already has an actual outcome.")
        return

    answer = input(
        "\nWas the order actually delivered late? (yes/no): "
    ).strip().lower()

    if answer in {"yes", "y"}:
        actual_value = 1
    elif answer in {"no", "n"}:
        actual_value = 0
    else:
        print("Invalid answer. Enter yes or no.")
        return

    logs.loc[matching, "actual_is_late"] = actual_value
    logs.to_csv(LOG_PATH, index=False)

    print("\nOutcome updated successfully!")
    print(f"Prediction ID: {prediction_id}")
    print(f"Actual outcome: {'Late' if actual_value == 1 else 'On time'}")


if __name__ == "__main__":
    main()
