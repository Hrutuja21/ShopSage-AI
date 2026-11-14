
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"
OUTPUT_PATH = ROOT / "reports" / "state_split_analysis.csv"


def main():
    print("=== ShopSage AI | State Analysis Across Time Splits ===")

    df = pd.read_csv(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "order_purchase_timestamp",
            "is_late",
            "customer_state"
        ]
    )

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    # Reproduce the same chronological 70/15/15 split
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    splits = {
        "Training": df.iloc[:train_end],
        "Validation": df.iloc[train_end:validation_end],
        "Testing": df.iloc[validation_end:]
    }

    results = []

    for split_name, split_df in splits.items():

        print(f"\n{'=' * 55}")
        print(f"{split_name.upper()} PERIOD")
        print(f"{'=' * 55}")

        print("Orders:", len(split_df))
        print(
            "Late-delivery rate:",
            round(split_df["is_late"].mean() * 100, 2),
            "%"
        )

        state_stats = (
            split_df.groupby("customer_state")
            .agg(
                total_orders=("is_late", "count"),
                late_orders=("is_late", "sum")
            )
            .reset_index()
        )

        state_stats["late_rate"] = (
            state_stats["late_orders"]
            / state_stats["total_orders"]
            * 100
        )

        state_stats["split"] = split_name

        # Display states with at least 50 orders
        reliable = state_stats[
            state_stats["total_orders"] >= 50
        ].sort_values("late_rate", ascending=False)

        print("\nHighest late-delivery rates (minimum 50 orders):")
        print(reliable.head(10).to_string(index=False))

        # Always inspect the SHAP-important states
        print("\nSHAP-important states:")
        important_states = state_stats[
            state_stats["customer_state"].isin(
                ["AL", "MA", "PR", "RR", "MG", "PA", "DF", "RJ"]
            )
        ]

        print(important_states.to_string(index=False))

        results.append(state_stats)

    final_df = pd.concat(results, ignore_index=True)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("\nState split analysis completed.")


if __name__ == "__main__":
    main()
