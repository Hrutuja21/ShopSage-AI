
"""Input feature engineering for ShopSage AI delivery predictions."""

import numpy as np
import pandas as pd


ORDER_DETAIL_FEATURES = [
    "customer_state",
    "customer_zip_code_prefix",
    "item_count",
    "unique_products",
    "unique_sellers",
    "unique_categories",
    "total_price",
    "total_freight",
    "average_item_price",
    "average_item_weight",
    "average_product_volume",
    "primary_product_category",
    "total_payment_value",
    "max_installments",
    "payment_types",
    "primary_payment_type",
]


def build_order_features(
    purchase_timestamp,
    estimated_delivery_date,
    order_details: dict,
) -> dict:
    """
    Build the exact 30 features expected by the trained V2 model.

    purchase_timestamp:
        Date and time when the customer places the order.

    estimated_delivery_date:
        Promised delivery date available when the order is placed.

    order_details:
        Customer, cart, product and payment information.
    """

    if not isinstance(order_details, dict):
        raise TypeError("order_details must be a dictionary.")

    missing = [
        name for name in ORDER_DETAIL_FEATURES
        if name not in order_details
    ]

    if missing:
        raise ValueError(f"Missing order details: {missing}")

    purchase = pd.Timestamp(purchase_timestamp)
    estimated = pd.Timestamp(estimated_delivery_date)

    if pd.isna(purchase) or pd.isna(estimated):
        raise ValueError("Purchase and delivery dates must be valid.")

    if estimated.normalize() < purchase.normalize():
        raise ValueError(
            "Estimated delivery date cannot be before purchase date."
        )

    weekday = int(purchase.dayofweek)
    estimated_weekday = int(estimated.dayofweek)
    month = int(purchase.month)

    features = {
        "purchase_year": int(purchase.year),
        "purchase_month": month,
        "purchase_day": int(purchase.day),
        "purchase_hour": int(purchase.hour),
        "purchase_weekday": weekday,
        "is_weekend": int(weekday >= 5),

        "purchase_month_sin": float(
            np.sin(2 * np.pi * month / 12)
        ),
        "purchase_month_cos": float(
            np.cos(2 * np.pi * month / 12)
        ),
        "purchase_weekday_sin": float(
            np.sin(2 * np.pi * weekday / 7)
        ),
        "purchase_weekday_cos": float(
            np.cos(2 * np.pi * weekday / 7)
        ),

        "promised_delivery_days": int(
            (estimated.normalize() - purchase.normalize()).days
        ),
        "estimated_delivery_month": int(estimated.month),
        "estimated_delivery_weekday": estimated_weekday,
        "estimated_delivery_is_weekend": int(
            estimated_weekday >= 5
        ),
    }

    features.update({
        name: order_details[name]
        for name in ORDER_DETAIL_FEATURES
    })

    return features
