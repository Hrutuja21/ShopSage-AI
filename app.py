
"""ShopSage AI - E-Commerce Analytics, Delivery Risk Prediction and Monitoring."""

from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.shopsage.features import build_order_features
from src.shopsage.predict import predict_order
from src.shopsage.monitoring import (
    log_prediction,
    load_prediction_logs,
    update_actual_outcome,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "olist_features.csv"


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="ShopSage AI",
    page_icon="S",
    layout="wide",
)


# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_historical_data():
    """Load the processed historical Olist dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "order_purchase_timestamp",
        "is_late",
        "customer_state",
        "total_price",
        "total_freight",
        "primary_product_category",
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="coerce",
    )

    data["is_late"] = pd.to_numeric(
        data["is_late"],
        errors="coerce",
    )

    return data


def safe_number(data, column, default=0):
    """Get a numeric default from historical data."""

    if column not in data.columns:
        return default

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return default

    return float(values.median())


def get_options(data, column, fallback):
    """Get dropdown values from historical data."""

    if column not in data.columns:
        return fallback

    values = (
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values) if values else fallback


# ---------------------------------------------------------
# 2. HISTORICAL ANALYTICS
# ---------------------------------------------------------

def show_overview(data):

    st.subheader("Historical E-Commerce Overview")

    valid = data.dropna(
        subset=["order_purchase_timestamp", "is_late"]
    ).copy()

    total_orders = len(valid)

    late_rate = (
        valid["is_late"].mean() * 100
        if total_orders else 0
    )

    average_order_value = valid["total_price"].mean()
    average_freight = valid["total_freight"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Historical Orders", f"{total_orders:,}")
    c2.metric("Observed Late Rate", f"{late_rate:.2f}%")

    c3.metric(
        "Average Order Value",
        f"{average_order_value:.2f}",
    )

    c4.metric(
        "Average Freight",
        f"{average_freight:.2f}",
    )

    st.caption(
        "These figures describe the historical Olist dataset, "
        "not live store activity."
    )

    if valid.empty:
        st.warning("No valid historical records are available.")
        return

    monthly = (
        valid.assign(
            month=valid["order_purchase_timestamp"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("month")
        .agg(
            orders=("is_late", "size"),
            late_rate=("is_late", "mean"),
        )
        .reset_index()
    )

    monthly["late_rate"] *= 100

    st.markdown("#### Monthly Delivery Performance")

    fig_month = px.line(
        monthly,
        x="month",
        y="late_rate",
        markers=True,
        labels={
            "month": "Purchase Month",
            "late_rate": "Late Delivery Rate (%)",
        },
        title="Observed Monthly Late Delivery Rate",
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:

        state_data = (
            valid.groupby("customer_state")
            .agg(
                orders=("is_late", "size"),
                late_rate=("is_late", "mean"),
            )
            .reset_index()
        )

        state_data = state_data[
            state_data["orders"] >= 100
        ].copy()

        state_data["late_rate"] *= 100

        state_data = state_data.sort_values(
            "late_rate",
            ascending=False,
        ).head(12)

        fig_state = px.bar(
            state_data,
            x="customer_state",
            y="late_rate",
            hover_data=["orders"],
            labels={
                "customer_state": "Customer State",
                "late_rate": "Late Rate (%)",
                "orders": "Orders",
            },
            title="States with Highest Observed Late Rates",
        )

        st.plotly_chart(
            fig_state,
            use_container_width=True,
        )

    with right:

        category_data = (
            valid.groupby("primary_product_category")
            .size()
            .reset_index(name="orders")
            .sort_values("orders", ascending=False)
            .head(12)
        )

        fig_category = px.bar(
            category_data,
            x="orders",
            y="primary_product_category",
            orientation="h",
            labels={
                "orders": "Number of Orders",
                "primary_product_category": "Product Category",
            },
            title="Most Common Product Categories",
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True,
        )

    st.markdown("#### Historical Dataset Sample")

    st.dataframe(
        valid[
            [
                "order_purchase_timestamp",
                "customer_state",
                "primary_product_category",
                "total_price",
                "total_freight",
                "is_late",
            ]
        ].head(10),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# 3. DELIVERY RISK PREDICTION
# ---------------------------------------------------------

def show_prediction_form(data):

    st.subheader("Predict Delivery Risk for an Order")

    st.write(
        "Enter the order details available when a customer places "
        "an order. ShopSage calculates a model risk score and "
        "indicates whether the order should be reviewed."
    )

    st.info(
        "The model was trained on historical Brazilian Olist orders "
        "from 2016-2018. Its score is a ranking score, not a perfectly "
        "calibrated probability. Use it as decision support, not as "
        "a guarantee."
    )

    states = get_options(
        data,
        "customer_state",
        ["SP", "RJ", "MG"],
    )

    categories = get_options(
        data,
        "primary_product_category",
        ["other"],
    )

    payment_options = get_options(
        data,
        "primary_payment_type",
        ["credit_card", "boleto", "voucher", "debit_card"],
    )

    with st.form("delivery_prediction_form"):

        st.markdown("#### 1. Order Dates")

        col1, col2, col3 = st.columns(3)

        with col1:
            purchase_date = st.date_input(
                "Purchase Date",
                value=date(2018, 8, 1),
            )

        with col2:
            purchase_time = st.time_input(
                "Purchase Time",
                value=time(10, 0),
            )

        with col3:
            estimated_date = st.date_input(
                "Promised Delivery Date",
                value=date(2018, 8, 18),
            )

        st.markdown("#### 2. Customer Details")

        col1, col2 = st.columns(2)

        with col1:
            customer_state = st.selectbox(
                "Customer State",
                states,
            )

        with col2:
            customer_zip = st.number_input(
                "Customer ZIP Code Prefix",
                min_value=0,
                max_value=99999,
                value=int(
                    safe_number(
                        data,
                        "customer_zip_code_prefix",
                        10000,
                    )
                ),
                step=1,
            )

        st.markdown("#### 3. Cart and Product Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            item_count = st.number_input(
                "Number of Items",
                min_value=1,
                value=max(
                    1,
                    int(safe_number(data, "item_count", 1)),
                ),
                step=1,
            )

            unique_products = st.number_input(
                "Unique Products",
                min_value=1,
                value=max(
                    1,
                    int(safe_number(data, "unique_products", 1)),
                ),
                step=1,
            )

        with col2:

            unique_sellers = st.number_input(
                "Unique Sellers",
                min_value=1,
                value=max(
                    1,
                    int(safe_number(data, "unique_sellers", 1)),
                ),
                step=1,
            )

            unique_categories = st.number_input(
                "Unique Categories",
                min_value=1,
                value=max(
                    1,
                    int(safe_number(data, "unique_categories", 1)),
                ),
                step=1,
            )

        with col3:

            primary_category = st.selectbox(
                "Primary Product Category",
                categories,
            )

            total_price = st.number_input(
                "Total Product Price",
                min_value=0.0,
                value=max(
                    0.0,
                    safe_number(data, "total_price", 100),
                ),
                step=10.0,
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            total_freight = st.number_input(
                "Total Freight",
                min_value=0.0,
                value=max(
                    0.0,
                    safe_number(data, "total_freight", 20),
                ),
                step=5.0,
            )

        with col2:

            average_item_price = st.number_input(
                "Average Item Price",
                min_value=0.0,
                value=max(
                    0.0,
                    safe_number(data, "average_item_price", 50),
                ),
                step=5.0,
            )

        with col3:

            average_item_weight = st.number_input(
                "Average Item Weight",
                min_value=0.0,
                value=max(
                    0.0,
                    safe_number(data, "average_item_weight", 500),
                ),
                step=50.0,
            )

        average_weight_unknown = st.checkbox(
            "Average item weight is unknown"
        )

        average_product_volume = st.number_input(
            "Average Product Volume",
            min_value=0.0,
            value=max(
                0.0,
                safe_number(data, "average_product_volume", 1000),
            ),
            step=100.0,
        )

        average_volume_unknown = st.checkbox(
            "Average product volume is unknown"
        )

        st.markdown("#### 4. Payment Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            total_payment_value = st.number_input(
                "Total Payment Value",
                min_value=0.0,
                value=max(
                    0.0,
                    safe_number(data, "total_payment_value", 100),
                ),
                step=10.0,
            )

        with col2:

            max_installments = st.number_input(
                "Maximum Installments",
                min_value=1,
                value=max(
                    1,
                    int(safe_number(data, "max_installments", 1)),
                ),
                step=1,
            )

        with col3:

            payment_types = st.number_input(
                "Number of Payment Types",
                min_value=1,
                value=max(
                    1,
                    int(safe_number(data, "payment_types", 1)),
                ),
                step=1,
            )

        primary_payment_type = st.selectbox(
            "Primary Payment Type",
            payment_options,
        )

        submitted = st.form_submit_button(
            "Predict Delivery Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        purchase_timestamp = datetime.combine(
            purchase_date,
            purchase_time,
        )

        estimated_delivery = datetime.combine(
            estimated_date,
            time(0, 0),
        )

        if estimated_date < purchase_date:

            st.error(
                "Promised delivery date cannot be before "
                "the purchase date."
            )
            return

        order_details = {
            "customer_state": customer_state,
            "customer_zip_code_prefix": int(customer_zip),
            "item_count": int(item_count),
            "unique_products": int(unique_products),
            "unique_sellers": int(unique_sellers),
            "unique_categories": int(unique_categories),
            "total_price": float(total_price),
            "total_freight": float(total_freight),
            "average_item_price": float(average_item_price),
            "average_item_weight": (
                None if average_weight_unknown
                else float(average_item_weight)
            ),
            "average_product_volume": (
                None if average_volume_unknown
                else float(average_product_volume)
            ),
            "primary_product_category": primary_category,
            "total_payment_value": float(total_payment_value),
            "max_installments": int(max_installments),
            "payment_types": int(payment_types),
            "primary_payment_type": primary_payment_type,
        }

        try:

            features = build_order_features(
                purchase_timestamp,
                estimated_delivery,
                order_details,
            )

            result = predict_order(features)

            prediction_id = log_prediction(
                features,
                result,
            )

            st.markdown("---")
            st.markdown("### Prediction Result")

            score = result["risk_score"]
            threshold = result["threshold"]

            col1, col2 = st.columns(2)

            col1.metric(
                "Model Risk Score",
                f"{score * 100:.2f}%",
            )

            col2.metric(
                "Review Threshold",
                f"{threshold * 100:.2f}%",
            )

            st.progress(
                min(max(score, 0.0), 1.0)
            )

            if result["is_flagged"]:

                st.warning(
                    "Review recommended: this order is flagged "
                    "by the current model threshold."
                )

            else:

                st.success(
                    "This order is not flagged by the current "
                    "model threshold."
                )

            st.success("Prediction saved successfully.")

            st.caption(
                f"Prediction ID: {prediction_id}"
            )

            st.caption(
                "The score is a model-generated ranking score. "
                "It is not a guaranteed delivery outcome or a "
                "calibrated real-world probability."
            )

            with st.expander("View model input features"):
                st.json(features)

        except FileNotFoundError as error:

            st.error(str(error))

            st.error(
                "Check that the trained V2 model and metadata "
                "exist inside the project folders."
            )

        except Exception as error:

            st.error(
                f"Prediction could not be completed: {error}"
            )


# ---------------------------------------------------------
# 4. MODEL MONITORING
# ---------------------------------------------------------

def show_monitoring():

    st.subheader("Prediction Monitoring")

    # -----------------------------------------------------
    # ACTUAL DELIVERY OUTCOME UPDATE
    # -----------------------------------------------------

    st.markdown("### Update Actual Delivery Outcome")

    outcome_logs = load_prediction_logs()

    if not outcome_logs.empty:

        outcome_logs = outcome_logs.copy()

        outcome_logs["actual_is_late"] = pd.to_numeric(
            outcome_logs["actual_is_late"],
            errors="coerce",
        )

        outcome_logs["risk_score"] = pd.to_numeric(
            outcome_logs["risk_score"],
            errors="coerce",
        )

        pending_logs = outcome_logs[
            outcome_logs["actual_is_late"].isna()
        ].copy()

        if not pending_logs.empty:

            prediction_options = {}

            for _, row in pending_logs.iterrows():

                prediction_id = str(row["prediction_id"])

                customer_state = row.get(
                    "customer_state",
                    "Unknown",
                )

                if pd.isna(customer_state):
                    customer_state = "Unknown"

                risk_score = row["risk_score"]

                risk_label = (
                    f"{risk_score:.3f}"
                    if pd.notna(risk_score)
                    else "N/A"
                )

                label = (
                    f"{prediction_id} | "
                    f"{customer_state} | "
                    f"Risk: {risk_label}"
                )

                prediction_options[label] = prediction_id

            with st.form("actual_outcome_form"):

                selected_label = st.selectbox(
                    "Select a pending prediction",
                    options=list(prediction_options.keys()),
                )

                actual_outcome = st.radio(
                    "What was the actual delivery outcome?",
                    options=[0, 1],
                    format_func=lambda value: (
                        "On Time" if value == 0 else "Late"
                    ),
                    horizontal=True,
                )

                submitted = st.form_submit_button(
                    "Save Actual Outcome",
                    type="primary",
                    use_container_width=True,
                )

                if submitted:

                    try:

                        selected_id = prediction_options[
                            selected_label
                        ]

                        update_actual_outcome(
                            selected_id,
                            int(actual_outcome),
                        )

                        st.success(
                            "Actual delivery outcome saved successfully."
                        )

                        st.rerun()

                    except Exception as error:

                        st.error(
                            f"Unable to update outcome: {error}"
                        )

        else:

            st.success(
                "All recorded predictions have verified outcomes."
            )

    else:

        st.info(
            "No prediction records are available yet. "
            "Generate a prediction to begin monitoring."
        )

    st.divider()

    # -----------------------------------------------------
    # LOAD MONITORING DATA
    # -----------------------------------------------------

    logs = load_prediction_logs()

    if logs.empty:

        st.info(
            "No predictions recorded yet. Submit an order in the "
            "Delivery Risk Prediction tab to start monitoring."
        )
        return

    logs = logs.copy()

    logs["risk_score"] = pd.to_numeric(
        logs["risk_score"],
        errors="coerce",
    )

    logs["actual_is_late"] = pd.to_numeric(
        logs["actual_is_late"],
        errors="coerce",
    )

    logs["is_flagged"] = (
        logs["is_flagged"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        })
    )

    logs["prediction_timestamp"] = pd.to_datetime(
        logs["prediction_timestamp"],
        errors="coerce",
        utc=True,
    )

    total = len(logs)

    valid_flags = logs["is_flagged"].dropna()

    flagged = int((valid_flags == 1).sum())

    flag_rate = (
        flagged / len(valid_flags) * 100
        if len(valid_flags)
        else 0
    )

    average_score = logs["risk_score"].mean() * 100

    verified_mask = (
        logs["actual_is_late"].isin([0, 1])
        & logs["is_flagged"].isin([0, 1])
    )

    actual_data = logs.loc[
        verified_mask
    ].copy()

    verified_count = len(actual_data)
    pending_count = total - verified_count

    # -----------------------------------------------------
    # MONITORING OVERVIEW
    # -----------------------------------------------------

    st.markdown("### Monitoring Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Predictions", f"{total:,}")
    c2.metric("Flagged Orders", f"{flagged:,}")
    c3.metric("Verified Outcomes", f"{verified_count:,}")
    c4.metric("Pending Outcomes", f"{pending_count:,}")

    c5, c6 = st.columns(2)

    c5.metric("Flag Rate", f"{flag_rate:.2f}%")

    c6.metric(
        "Average Risk Score",
        (
            f"{average_score:.2f}%"
            if pd.notna(average_score)
            else "N/A"
        ),
    )

    st.caption(
        "Flag rate is the percentage of predictions classified "
        "for review. Average risk score is the mean model score."
    )

    st.divider()

    # -----------------------------------------------------
    # RISK SCORE DISTRIBUTION
    # -----------------------------------------------------

    st.markdown("### Risk Score Distribution")

    risk_data = logs.dropna(
        subset=["risk_score"]
    ).copy()

    if not risk_data.empty:

        fig_risk = px.histogram(
            risk_data,
            x="risk_score",
            nbins=20,
            labels={
                "risk_score": "Model Risk Score",
                "count": "Number of Predictions",
            },
            title="Distribution of Recorded Risk Scores",
        )

        fig_risk.update_layout(
            xaxis_tickformat=".0%",
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True,
        )

    else:

        st.info("No valid risk scores are available.")

    st.divider()

    # -----------------------------------------------------
    # PREDICTION ACTIVITY
    # -----------------------------------------------------

    st.markdown("### Prediction Activity")

    activity_logs = logs.dropna(
        subset=["prediction_timestamp"]
    ).copy()

    if not activity_logs.empty:

        activity = (
            activity_logs.assign(
                prediction_date=activity_logs[
                    "prediction_timestamp"
                ].dt.date
            )
            .groupby("prediction_date")
            .size()
            .reset_index(name="predictions")
        )

        fig_activity = px.line(
            activity,
            x="prediction_date",
            y="predictions",
            markers=True,
            labels={
                "prediction_date": "Prediction Date",
                "predictions": "Number of Predictions",
            },
            title="Daily Prediction Volume",
        )

        st.plotly_chart(
            fig_activity,
            use_container_width=True,
        )

    else:

        st.info("Prediction timestamps are not available.")

    st.divider()

    # -----------------------------------------------------
    # ACTUAL OUTCOME EVALUATION
    # -----------------------------------------------------

    st.markdown("### Actual Outcome Evaluation")

    if actual_data.empty:

        st.info(
            "No verified outcomes are available yet. "
            "Update actual delivery outcomes to calculate "
            "monitoring performance."
        )

    else:

        actual = actual_data[
            "actual_is_late"
        ].astype(int)

        predicted = actual_data[
            "is_flagged"
        ].astype(int)

        accuracy = accuracy_score(actual, predicted)

        precision = precision_score(
            actual,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{accuracy:.3f}")
        m2.metric("Precision", f"{precision:.3f}")
        m3.metric("Recall", f"{recall:.3f}")
        m4.metric("F1 Score", f"{f1:.3f}")

        cm = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        st.markdown("#### Confusion Matrix")

        matrix_data = pd.DataFrame(
            cm,
            index=[
                "Actual On Time",
                "Actual Late",
            ],
            columns=[
                "Predicted Not Flagged",
                "Predicted Flagged",
            ],
        )

        fig_cm = px.imshow(
            matrix_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Model Prediction",
                "y": "Actual Outcome",
                "color": "Orders",
            },
            title="Actual Outcomes vs Model Predictions",
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True,
        )

        st.markdown("#### Prediction Error Analysis")

        e1, e2, e3, e4 = st.columns(4)

        e1.metric("True Negatives", f"{tn:,}")
        e2.metric("True Positives", f"{tp:,}")
        e3.metric("False Positives", f"{fp:,}")
        e4.metric("False Negatives", f"{fn:,}")

        st.caption(
            "False positives are on-time orders flagged for review. "
            "False negatives are late orders that were not flagged."
        )

        late_count = int((actual == 1).sum())
        on_time_count = int((actual == 0).sum())

        if (
            verified_count < 30
            or late_count < 5
            or on_time_count < 5
        ):

            st.warning(
                f"Only {verified_count} verified outcomes are available "
                f"({late_count} late and {on_time_count} on time). "
                "These metrics are preliminary and should not be "
                "treated as reliable evidence of model performance. "
                "Collect more verified outcomes."
            )

        else:

            st.success(
                "The monitoring dataset contains a larger set of "
                "verified outcomes. Continue collecting data and "
                "monitor performance over time."
            )

    st.divider()

    # -----------------------------------------------------
    # RECENT PREDICTIONS
    # -----------------------------------------------------

    st.markdown("### Recent Predictions")

    display_columns = [
        "prediction_id",
        "prediction_timestamp",
        "customer_state",
        "risk_score",
        "threshold",
        "is_flagged",
        "prediction",
        "actual_is_late",
    ]

    available_columns = [
        column for column in display_columns
        if column in logs.columns
    ]

    recent = logs.sort_values(
        "prediction_timestamp",
        ascending=False,
        na_position="last",
    )

    st.dataframe(
        recent[available_columns].head(50),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Actual outcome: 1 = Late, 0 = On Time, blank = Pending."
    )

    st.info(
        "Monitoring metrics are calculated only from predictions "
        "with verified actual outcomes. They are separate from the "
        "historical test-set metrics shown in Model Information."
    )


# ---------------------------------------------------------
# 5. MODEL INFORMATION
# ---------------------------------------------------------

def show_model_information():

    st.subheader("Delivery Risk Model - Version 2")

    st.write(
        "ShopSage uses a trained XGBoost model to rank orders "
        "according to their estimated late-delivery risk."
    )

    metrics = {
        "PR-AUC": "0.1604",
        "ROC-AUC": "0.7272",
        "Precision": "0.1602",
        "Recall": "0.4556",
        "F1 Score": "0.2370",
        "Brier Score": "0.0589",
    }

    metric_columns = st.columns(3)

    for index, (name, value) in enumerate(metrics.items()):

        metric_columns[index % 3].metric(
            name,
            value,
        )

    st.markdown("#### Model Evaluation Notes")

    st.write(
        "- Evaluation used a chronological 70/15/15 "
        "train-validation-test split."
    )

    st.write(
        "- The final test set contained 14,471 historical orders."
    )

    st.write(
        "- The review threshold of 0.10 was selected using validation data."
    )

    st.write(
        "- The model detected approximately 45.56% of late orders "
        "in the historical test set."
    )

    st.write(
        "- Precision was approximately 16.02%, so many flagged "
        "orders were not actually late."
    )

    st.warning(
        "This is a historical research and decision-support model. "
        "It is not a production delivery guarantee. Its performance "
        "may change on newer data or in other markets."
    )


# ---------------------------------------------------------
# 6. MAIN APPLICATION
# ---------------------------------------------------------

def main():

    st.title("ShopSage AI")

    st.caption(
        "Intelligent E-Commerce Analytics and Delivery Risk Prediction"
    )

    try:

        data = load_historical_data()

    except Exception as error:

        st.error(
            f"Could not load the historical dataset: {error}"
        )

        st.stop()

    date_min = data[
        "order_purchase_timestamp"
    ].min()

    date_max = data[
        "order_purchase_timestamp"
    ].max()

    st.caption(
        f"Historical data period: {date_min:%d %b %Y} "
        f"to {date_max:%d %b %Y} | "
        f"{len(data):,} processed orders"
    )

    (
        overview_tab,
        prediction_tab,
        model_tab,
        monitoring_tab,
    ) = st.tabs(
        [
            "Historical Analytics",
            "Delivery Risk Prediction",
            "Model Information",
            "Model Monitoring",
        ]
    )

    with overview_tab:
        show_overview(data)

    with prediction_tab:
        show_prediction_form(data)

    with model_tab:
        show_model_information()

    with monitoring_tab:
        show_monitoring()


if __name__ == "__main__":
    main()
