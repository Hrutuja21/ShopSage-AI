
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

MONTHLY_PATH = REPORTS / "monthly_error_analysis_v2.csv"
STATE_PATH = REPORTS / "state_error_analysis_v2.csv"
METRICS_PATH = REPORTS / "calibration_metrics_v2.csv"
METADATA_PATH = REPORTS / "training_metadata_v2.json"

OUTPUT_PATH = REPORTS / "business_insights_v2.txt"

print("\n=== ShopSage AI | Business Insights Engine ===")


def load_csv(path):
    if not path.exists():
        print(f"Optional file not found: {path.name}")
        return None
    return pd.read_csv(path)


def pct(value):
    return f"{value * 100:.2f}%"


def generate_insights():
    monthly = load_csv(MONTHLY_PATH)
    states = load_csv(STATE_PATH)
    calibration = load_csv(METRICS_PATH)

    if monthly is None or states is None:
        raise FileNotFoundError(
            "Run error_analysis_v2.py before this script."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    lines = []

    def add(text=""):
        lines.append(str(text))

    add("=" * 65)
    add("SHOPSAGE AI - BUSINESS INSIGHTS REPORT")
    add("Delivery Risk Analysis | Model V2")
    add("=" * 65)

    add("\n1. MODEL OVERVIEW")
    add("-" * 40)

    threshold = metadata.get("prediction_threshold", "Not available")
    add(f"Prediction threshold: {threshold}")
    add("Evaluation approach: Chronological train/validation/test split")
    add("Model: Selected V2 delivery-risk model")

    test_metrics = metadata.get("test_metrics", {})

    if isinstance(test_metrics, dict):
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                add(f"{key}: {value:.4f}")

    # --------------------------------------------------
    # MONTHLY INSIGHTS
    # --------------------------------------------------

    add("\n2. MONTHLY DELIVERY PERFORMANCE")
    add("-" * 40)

    required_month_cols = {
        "month",
        "total_orders",
        "actual_late_rate",
        "predicted_late_rate",
    }

    if required_month_cols.issubset(monthly.columns):
        monthly = monthly.sort_values("month")

        for _, row in monthly.iterrows():
            add(
                f"{row['month']}: "
                f"{int(row['total_orders'])} orders, "
                f"actual late rate {pct(row['actual_late_rate'])}, "
                f"predicted late rate {pct(row['predicted_late_rate'])}"
            )

        monthly["risk_gap"] = (
            monthly["predicted_late_rate"]
            - monthly["actual_late_rate"]
        )

        highest_gap = monthly.loc[monthly["risk_gap"].idxmax()]
        lowest_gap = monthly.loc[monthly["risk_gap"].idxmin()]

        add("\nMonthly observations:")
        add(
            f"- Largest overprediction: {highest_gap['month']} "
            f"({pct(highest_gap['risk_gap'])} percentage-point gap)."
        )
        add(
            f"- Smallest prediction gap: {lowest_gap['month']} "
            f"({pct(abs(lowest_gap['risk_gap']))} absolute gap)."
        )

        add(
            "- Delivery risk varies over time. "
            "Monitor changing monthly patterns rather than assuming "
            "a fixed late-delivery rate."
        )

    # --------------------------------------------------
    # STATE INSIGHTS
    # --------------------------------------------------

    add("\n3. REGIONAL DELIVERY INSIGHTS")
    add("-" * 40)

    required_state_cols = {
        "customer_state",
        "total_orders",
        "actual_late_rate",
        "predicted_late_rate",
        "false_positive",
        "false_negative",
    }

    if required_state_cols.issubset(states.columns):
        # Exclude very small samples from the main comparison.
        reliable = states[states["total_orders"] >= 100].copy()

        if not reliable.empty:
            reliable = reliable.sort_values(
                "actual_late_rate",
                ascending=False
            )

            add("States with highest observed late-delivery rates:")
            for _, row in reliable.head(5).iterrows():
                add(
                    f"- {row['customer_state']}: "
                    f"{pct(row['actual_late_rate'])} actual late rate, "
                    f"{int(row['total_orders'])} orders."
                )

            add("\nStates with largest prediction overestimation:")
            reliable["prediction_gap"] = (
                reliable["predicted_late_rate"]
                - reliable["actual_late_rate"]
            )

            overpredicted = reliable.sort_values(
                "prediction_gap",
                ascending=False
            )

            for _, row in overpredicted.head(5).iterrows():
                add(
                    f"- {row['customer_state']}: "
                    f"predicted {pct(row['predicted_late_rate'])}, "
                    f"actual {pct(row['actual_late_rate'])}."
                )

            add("\nStates with largest prediction underestimation:")
            underpredicted = reliable.sort_values("prediction_gap")

            for _, row in underpredicted.head(5).iterrows():
                add(
                    f"- {row['customer_state']}: "
                    f"predicted {pct(row['predicted_late_rate'])}, "
                    f"actual {pct(row['actual_late_rate'])}."
                )

            add(
                "\nRegional observation: customer state is associated "
                "with different observed delivery patterns. These "
                "relationships are predictive, not proof of causation."
            )

    # --------------------------------------------------
    # OPERATIONAL RECOMMENDATIONS
    # --------------------------------------------------

    add("\n4. BUSINESS RECOMMENDATIONS")
    add("-" * 40)

    add(
        "1. Prioritize high-risk orders for manual review before "
        "promised delivery dates."
    )

    add(
        "2. Monitor regional performance using both order volume "
        "and observed late-delivery rates."
    )

    add(
        "3. Investigate states with large gaps between predicted "
        "and actual late-delivery rates."
    )

    add(
        "4. Review monthly trends to identify changes in delivery "
        "performance and potential distribution shift."
    )

    add(
        "5. Track false positives and false negatives separately. "
        "False positives create unnecessary interventions, while "
        "false negatives represent late orders the model misses."
    )

    add(
        "6. Validate the model on newer data before deploying "
        "automated customer notifications or operational decisions."
    )

    # --------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------

    add("\n5. PROBABILITY CALIBRATION")
    add("-" * 40)

    if calibration is not None:
        add(
            "Calibration comparison was available. "
            "See calibration_metrics_v2.csv for the detailed metrics."
        )
        add(
            "The original V2 model performed marginally better than "
            "the Platt-calibrated version in the reported test results."
        )
    else:
        add("Calibration metrics file was not available.")

    # --------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------

    add("\n6. LIMITATIONS AND RESPONSIBLE USE")
    add("-" * 40)

    add(
        "- Results are based on a historical Olist dataset."
    )
    add(
        "- Regional patterns may change over time."
    )
    add(
        "- Small state-level samples may produce unstable rates."
    )
    add(
        "- Model predictions indicate risk, not certainty."
    )
    add(
        "- SHAP feature importance does not establish causality."
    )
    add(
        "- This report is analytical support, not a production "
        "delivery-management system."
    )

    add("\n" + "=" * 65)
    add("END OF BUSINESS INSIGHTS REPORT")
    add("=" * 65)

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_insights()
