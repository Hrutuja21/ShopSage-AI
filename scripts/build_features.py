
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "olist"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():

    print("=== ShopSage AI | Improved Feature Engineering ===")

    # --------------------------------------------------
    # 1. LOAD DATASETS
    # --------------------------------------------------

    print("\nLoading Olist datasets...")

    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
    customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
    items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
    payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
    products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

    print("Orders:", orders.shape)
    print("Customers:", customers.shape)
    print("Items:", items.shape)
    print("Payments:", payments.shape)
    print("Products:", products.shape)

    # --------------------------------------------------
    # 2. PARSE DATE COLUMNS
    # --------------------------------------------------

    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_columns:
        orders[col] = pd.to_datetime(
            orders[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # 3. CREATE TARGET VARIABLE
    # --------------------------------------------------

    print("\nCreating late-delivery target...")

    data = orders[
        (orders["order_status"] == "delivered")
        & orders["order_purchase_timestamp"].notna()
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    data["is_late"] = (
        data["order_delivered_customer_date"]
        > data["order_estimated_delivery_date"]
    ).astype(int)

    # --------------------------------------------------
    # 4. PURCHASE-TIME FEATURES
    # --------------------------------------------------

    print("Creating purchase-time features...")

    purchase_time = data["order_purchase_timestamp"]
    estimated_time = data["order_estimated_delivery_date"]

    data["purchase_year"] = purchase_time.dt.year
    data["purchase_month"] = purchase_time.dt.month
    data["purchase_day"] = purchase_time.dt.day
    data["purchase_hour"] = purchase_time.dt.hour
    data["purchase_weekday"] = purchase_time.dt.dayofweek

    data["is_weekend"] = (
        purchase_time.dt.dayofweek >= 5
    ).astype(int)

    # Cyclical calendar features
    data["purchase_month_sin"] = np.sin(
        2 * np.pi * purchase_time.dt.month / 12
    )

    data["purchase_month_cos"] = np.cos(
        2 * np.pi * purchase_time.dt.month / 12
    )

    data["purchase_weekday_sin"] = np.sin(
        2 * np.pi * purchase_time.dt.dayofweek / 7
    )

    data["purchase_weekday_cos"] = np.cos(
        2 * np.pi * purchase_time.dt.dayofweek / 7
    )

    # --------------------------------------------------
    # 5. ESTIMATED DELIVERY FEATURES
    # --------------------------------------------------

    print("Creating promised delivery features...")

    # These are based on the estimated date available
    # when the order is placed, not the actual delivery.

    data["promised_delivery_days"] = (
        estimated_time.dt.normalize()
        - purchase_time.dt.normalize()
    ).dt.days

    data["estimated_delivery_month"] = estimated_time.dt.month

    data["estimated_delivery_weekday"] = (
        estimated_time.dt.dayofweek
    )

    data["estimated_delivery_is_weekend"] = (
        estimated_time.dt.dayofweek >= 5
    ).astype(int)

    # --------------------------------------------------
    # 6. CUSTOMER FEATURES
    # --------------------------------------------------

    print("Adding customer features...")

    customer_features = customers[
        [
            "customer_id",
            "customer_state",
            "customer_zip_code_prefix",
        ]
    ].copy()

    data = data.merge(
        customer_features,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------
    # 7. PRODUCT FEATURES
    # --------------------------------------------------

    print("Creating product features...")

    item_product = items.merge(
        products[
            [
                "product_id",
                "product_category_name",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    item_product["product_volume_cm3"] = (
        item_product["product_length_cm"]
        * item_product["product_height_cm"]
        * item_product["product_width_cm"]
    )

    item_features = item_product.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        unique_products=("product_id", "nunique"),
        unique_sellers=("seller_id", "nunique"),
        unique_categories=("product_category_name", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        average_item_price=("price", "mean"),
        average_item_weight=("product_weight_g", "mean"),
        average_product_volume=("product_volume_cm3", "mean"),
    ).reset_index()

    # Most common product category in each order
    primary_category = (
        item_product.groupby("order_id")["product_category_name"]
        .agg(
            lambda x: (
                x.mode().iloc[0]
                if not x.mode().empty
                else "unknown"
            )
        )
        .rename("primary_product_category")
        .reset_index()
    )

    item_features = item_features.merge(
        primary_category,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    data = data.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # 8. PAYMENT FEATURES
    # --------------------------------------------------

    print("Creating payment features...")

    payment_features = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_types=("payment_type", "nunique"),
    ).reset_index()

    primary_payment = (
        payments.groupby("order_id")["payment_type"]
        .agg(
            lambda x: (
                x.mode().iloc[0]
                if not x.mode().empty
                else "unknown"
            )
        )
        .rename("primary_payment_type")
        .reset_index()
    )

    payment_features = payment_features.merge(
        primary_payment,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    data = data.merge(
        payment_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # 9. REMOVE IDENTIFIERS AND LEAKAGE COLUMNS
    # --------------------------------------------------

    print("Removing identifiers and actual delivery information...")

    excluded_columns = [
        "order_id",
        "customer_id",
        "order_status",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "order_delivered_carrier_date",
        "order_approved_at",
    ]

    data = data.drop(
        columns=[
            col for col in excluded_columns
            if col in data.columns
        ]
    )

    # --------------------------------------------------
    # 10. DATA QUALITY CHECKS
    # --------------------------------------------------

    print("\nRunning data quality checks...")

    if data["is_late"].isna().any():
        raise ValueError("Target contains missing values.")

    if not set(data["is_late"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    if data["order_purchase_timestamp"].isna().any():
        raise ValueError("Purchase timestamp contains missing values.")

    if data.duplicated().any():
        print("Warning: Duplicate rows detected.")

    print("\nMissing values by feature:")
    print(data.isnull().sum())

    # --------------------------------------------------
    # 11. SAVE PROCESSED DATASET
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "olist_features.csv"

    data.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------
    # 12. FINAL SUMMARY
    # --------------------------------------------------

    print("\n===================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("===================================")

    print("\nProcessed dataset shape:", data.shape)

    print(
        "Late-delivery rate:",
        round(data["is_late"].mean() * 100, 2),
        "%"
    )

    print("\nTarget distribution:")
    print(data["is_late"].value_counts().sort_index())

    print("\nNew delivery-related features:")

    print(
        data[
            [
                "promised_delivery_days",
                "estimated_delivery_month",
                "estimated_delivery_weekday",
                "estimated_delivery_is_weekend",
            ]
        ].head()
    )

    print("\nFeature columns:")
    print(data.columns.tolist())

    print("\nSaved successfully:")
    print(output_path)


if __name__ == "__main__":
    main()
