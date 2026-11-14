from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "olist" / "olist_orders_dataset.csv"

if not DATA_PATH.exists():
    sys.exit(
        f"Dataset not found: {DATA_PATH}\n"
        "Download Olist from Kaggle and extract the CSV files into data/raw/olist/."
    )

orders = pd.read_csv(DATA_PATH)
print("=== ShopSage AI | Olist orders inspection ===")
print(f"Shape: {orders.shape}")
print("\nColumns:")
print(orders.columns.tolist())
print("\nMissing values:")
print(orders.isna().sum())

from src.shopsage.preprocessing import load_orders, add_late_delivery_target

orders = load_orders(DATA_PATH)
labelled = add_late_delivery_target(orders)

print(f"\nValid delivered orders: {len(labelled):,}")
print("\nLate-delivery label counts:")
print(labelled["is_late"].value_counts().sort_index())
print(f"\nLate-delivery rate: {labelled['is_late'].mean() * 100:.2f}%")
