
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "olist_features.csv"

df = pd.read_csv(DATA_PATH)

print("=== ShopSage AI | Customer State Analysis ===")

# State-level order counts and late-delivery rates
state_analysis = (
    df.groupby("customer_state")
    .agg(
        total_orders=("is_late", "count"),
        late_orders=("is_late", "sum"),
        late_rate=("is_late", "mean")
    )
    .reset_index()
)

state_analysis["late_rate"] *= 100

state_analysis = state_analysis.sort_values(
    "total_orders",
    ascending=False
)

print("\n1. STATE-LEVEL ORDER DISTRIBUTION")
print(state_analysis.to_string(index=False))

print("\n2. STATES WITH HIGHEST LATE-DELIVERY RATES")
print(
    state_analysis.sort_values(
        "late_rate",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\n3. STATES WITH LOWEST LATE-DELIVERY RATES")
print(
    state_analysis.sort_values(
        "late_rate",
        ascending=True
    ).head(10).to_string(index=False)
)

output_path = ROOT / "reports" / "state_analysis.csv"

state_analysis.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("\nAnalysis completed.")
