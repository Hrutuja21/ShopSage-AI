# ShopSage AI

**An AI-powered e-commerce analytics and delivery-risk prediction platform.**

ShopSage AI uses historical e-commerce data and machine learning to analyze business performance, identify potentially delayed orders, and support operational decision-making through an interactive Streamlit dashboard.

## Project Overview

Built using the Brazilian E-Commerce Public Dataset by Olist, ShopSage AI combines historical business analytics, predictive modelling, model evaluation, explainability, and prediction monitoring in one application.

## Key Features

- **Historical Analytics:** Explore 96,470 e-commerce orders and analyze delivery performance, order trends, freight, and business KPIs.
- **Delivery Risk Prediction:** Identify potentially delayed orders using an XGBoost model.
- **Model Evaluation:** Compare PR-AUC, ROC-AUC, precision, recall, F1-score, and Brier score.
- **Explainable AI:** Understand important model features using SHAP.
- **Prediction Logging:** Save predictions for subsequent monitoring.
- **Model Monitoring:** Review pending predictions and update actual delivery outcomes when verified.
- **Interactive Dashboard:** Explore analytics, predictions, model information, and monitoring through Streamlit.

## Model Performance

Selected model: **XGBoost V2 (Unweighted)**

Evaluation strategy: Chronological train/validation/test split.

| Metric | Test Result |
|---|---:|
| PR-AUC | 0.1604 |
| ROC-AUC | 0.7272 |
| Precision | 0.1602 |
| Recall | 0.4556 |
| F1-score | 0.2370 |
| Brier Score | 0.0589 |
| Decision Threshold | 0.10 |

**Important:** The model score is a model-generated ranking score, not a calibrated real-world probability or a guarantee of delivery outcome. The decision threshold is configurable through model metadata.

## Technology Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib
- **Testing:** Pytest
- **Development:** Jupyter Notebook, Git

## Project Structure

```text
ShopSage-AI/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   └── best_delivery_model_v2.joblib
├── notebooks/
│   └── README.md
├── reports/
│   ├── business_insights_v2.py
│   ├── business_insights_v2.txt
│   ├── training_metadata_v2.json
│   ├── model_comparison.csv
│   ├── calibration_metrics_v2.csv
│   ├── calibration_risk_groups_v2.csv
│   ├── figures/
│   └── other analysis reports
├── scripts/
│   ├── build_features.py
│   ├── check_data.py
│   ├── diagnose_data.py
│   ├── train_model.py
│   ├── train_model_v2.py
│   ├── explain_model_v2.py
│   └── update_prediction_outcome.py
├── src/
│   └── shopsage/
│       ├── __init__.py
│       ├── ai_agent.py
│       ├── config.py
│       ├── explain.py
│       ├── features.py
│       ├── monitoring.py
│       ├── predict.py
│       ├── preprocessing.py
│       └── train.py
└── tests/
    ├── test_app_import.py
    └── test_data.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hrutuja21/ShopSage-AI.git
cd ShopSage-AI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Setup

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

1. Download the dataset from Kaggle.
2. Extract its CSV files into `data/raw/olist/`.
3. Use the included data inspection and feature-engineering scripts to prepare the processed dataset.
4. Place the resulting feature dataset at:

```text
data/processed/olist_features.csv
```

The raw and processed datasets are excluded from GitHub to keep the repository lightweight. They must be prepared locally before using historical analytics.

## Model Artifacts

The repository includes the selected V2 model and its training metadata:

```text
models/best_delivery_model_v2.joblib
reports/training_metadata_v2.json
```

These artifacts are required for delivery-risk prediction. The historical dataset is not included.

## Run the Application

After installing dependencies and preparing the dataset:

```bash
streamlit run app.py
```

The dashboard provides four main sections:

1. Historical Analytics
2. Delivery Risk Prediction
3. Model Information
4. Model Monitoring

## Run Tests

```bash
python -m pytest -v
```

## Modelling Approach

The project follows a chronological evaluation strategy to better represent real-world prediction scenarios.

- Historical order information is used to construct the delivery-delay target.
- Features are designed to represent information available at prediction time.
- Actual delivery timestamps are not used as prediction inputs.
- Model performance is evaluated on a held-out chronological test set.
- Prediction monitoring uses verified actual outcomes rather than assumed delivery results.

## Future Improvements

- Integrate PostgreSQL for persistent analytics storage.
- Expand the natural-language business analytics assistant.
- Develop API endpoints for model inference.
- Improve prediction calibration and risk segmentation.
- Deploy the dashboard to a cloud platform.

## Dataset Attribution

Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The dataset is distributed under the CC BY-NC-SA 4.0 license. Review its terms and provide appropriate attribution when using or redistributing derived material.

## Author

**Hrutuja Bhosale**

B.Tech – Computer Science and Engineering (Artificial Intelligence)

Pune, Maharashtra, India