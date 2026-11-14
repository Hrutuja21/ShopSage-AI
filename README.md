# ShopSage AI



**An AI-powered e-commerce analytics and delivery-risk prediction platform.**



ShopSage AI uses historical e-commerce data and machine learning to analyze business performance, identify potentially delayed orders, and support operational decision-making through an interactive Streamlit dashboard.



## Project Overview



ShopSage AI is built using the Brazilian E-Commerce Public Dataset by Olist. It combines historical business analytics, predictive modelling, model evaluation, explainability, and prediction monitoring in one application.



### Key Features



- **Historical Analytics:** Explore 96,470 e-commerce orders and analyze delivery performance, order trends, freight, and business KPIs.

- **Delivery Risk Prediction:** Predict potential delivery delays using an XGBoost machine learning model.

- **Model Evaluation:** Compare model performance using PR-AUC, ROC-AUC, precision, recall, F1-score, and Brier score.

- **Explainable AI:** Analyze important model features using SHAP.

- **Prediction Logging:** Save model predictions for subsequent monitoring.

- **Model Monitoring:** Review pending predictions and update actual delivery outcomes when verified.

- **Interactive Dashboard:** Explore analytics, predictions, model information, and monitoring through Streamlit.



## Model Performance



The selected model is **XGBoost V2 (Unweighted)**, evaluated using a chronological train/validation/test split.



| Metric | Test Result |

|---|---:|

| PR-AUC | 0.1604 |

| ROC-AUC | 0.7272 |

| Precision | 0.1602 |

| Recall | 0.4556 |

| F1-score | 0.2370 |

| Brier Score | 0.0589 |

| Decision Threshold | 0.10 |



**Important:** The model score is a model-generated ranking score, not a calibrated real-world probability or a guarantee of delivery outcome. The decision threshold is configurable through the model metadata.



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

â”‚

â”œâ”€â”€ app.py

â”œâ”€â”€ requirements.txt

â”œâ”€â”€ README.md

â”œâ”€â”€ .env.example

â”œâ”€â”€ .gitignore

â”‚

â”œâ”€â”€ data/

â”‚   â”œâ”€â”€ raw/

â”‚   â”œâ”€â”€ interim/

â”‚   â””â”€â”€ processed/

â”‚

â”œâ”€â”€ models/

â”‚   â””â”€â”€ best_delivery_model_v2.joblib

â”‚

â”œâ”€â”€ notebooks/

â”‚

â”œâ”€â”€ reports/

â”‚   â”œâ”€â”€ training_metadata_v2.json

â”‚   â”œâ”€â”€ model_comparison.csv

â”‚   â”œâ”€â”€ calibration_metrics_v2.csv

â”‚   â”œâ”€â”€ business_insights_v2.txt

â”‚   â””â”€â”€ figures/

â”‚

â”œâ”€â”€ scripts/

â”‚   â”œâ”€â”€ check_data.py

â”‚   â”œâ”€â”€ build_features.py

â”‚   â”œâ”€â”€ diagnose_data.py

â”‚   â”œâ”€â”€ train_model.py

â”‚   â”œâ”€â”€ train_model_v2.py

â”‚   â””â”€â”€ explain_model_v2.py

â”‚

â”œâ”€â”€ src/

â”‚   â””â”€â”€ shopsage/

â”‚       â”œâ”€â”€ config.py

â”‚       â”œâ”€â”€ features.py

â”‚       â”œâ”€â”€ predict.py

â”‚       â”œâ”€â”€ monitoring.py

â”‚       â”œâ”€â”€ preprocessing.py

â”‚       â”œâ”€â”€ explain.py

â”‚       â””â”€â”€ train.py

â”‚

â””â”€â”€ tests/

```



## Installation and Setup



### 1. Clone the repository



```bash

git clone https://github.com/YOUR-USERNAME/ShopSage-AI.git

cd ShopSage-AI

```



### 2. Create a virtual environment



Windows PowerShell:



```powershell

py -3.13 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



macOS/Linux:



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



1\. Download the dataset from Kaggle.

2\. Extract its CSV files into `data/raw/olist/`.

3\. Use the included data inspection and feature-engineering scripts to prepare the processed dataset.

4\. Place the resulting feature dataset at:



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



These artifacts are required for the delivery-risk prediction functionality. The historical dataset is not included.



## Run the Application



After installing dependencies and preparing the dataset:



```bash

streamlit run app.py

```



The dashboard provides four main sections:



1\. Historical Analytics

2\. Delivery Risk Prediction

3\. Model Information

4\. Model Monitoring



## Run Tests



```bash

python -m pytest -v

```



## Modelling Approach



The project follows a chronological evaluation strategy to better represent real-world prediction scenarios.



Key modelling considerations:



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



B.Tech â€“ Computer Science and Engineering (Artificial Intelligence)



Pune, Maharashtra, India
