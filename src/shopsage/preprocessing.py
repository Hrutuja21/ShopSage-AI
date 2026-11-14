import pandas as pd


def load_orders(path: str) -> pd.DataFrame:
    """Load Olist orders and parse relevant timestamps."""
    orders = pd.read_csv(path)
    date_columns = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ]
    for column in date_columns:
        if column in orders.columns:
            orders[column] = pd.to_datetime(orders[column], errors="coerce")
    return orders


def add_late_delivery_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Create a binary late-delivery label for delivered orders only."""
    required = {
        "order_status",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    }
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = orders.loc[
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    result["is_late"] = (
        result["order_delivered_customer_date"]
        > result["order_estimated_delivery_date"]
    ).astype("int8")
    return result
