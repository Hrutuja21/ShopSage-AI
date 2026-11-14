import pandas as pd

from src.shopsage.preprocessing import add_late_delivery_target


def test_late_delivery_target():
    data = pd.DataFrame({
        "order_status": ["delivered", "delivered", "canceled"],
        "order_delivered_customer_date": pd.to_datetime(
            ["2024-01-05", "2024-01-03", None]
        ),
        "order_estimated_delivery_date": pd.to_datetime(
            ["2024-01-04", "2024-01-04", "2024-01-04"]
        ),
    })

    result = add_late_delivery_target(data)

    assert len(result) == 2
    assert result["is_late"].tolist() == [1, 0]
