# Online Retail Customer Intelligence 

An interactive Streamlit application built from the supplied Online Retail II notebook outputs. It includes **Home**, **Sales Analytics**, **Customer Segments**, **Prediction**, **About**, and **How to Use** pages.

## Package contents

- `app.py` — Streamlit application
- `models/` — trained purchase-prediction pipeline and RFM scaler/K-Means artifacts
- `analytics/` — seven exported CSV tables from the notebook
- `requirements.txt` — Python dependencies

The original Excel workbooks and notebook are not required to run this app. They are intentionally not bundled because the app reads the exported tables and fitted artifacts only.

## Run locally

Use Python 3.11 or 3.12 and create a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app uses the already-trained model. **Do not run a training script** to launch it.

## Deploy on Streamlit Community Cloud

1. Push the contents of this project folder to a GitHub repository.
2. Keep `app.py`, `requirements.txt`, `models/`, and `analytics/` at the repository root.
3. In Streamlit Community Cloud, connect the repository and select `app.py` as the main file.
4. Deploy.

The model artifact was serialized with scikit-learn 1.9.0; the requirements file pins that version for compatibility. Keep the model file private if your deployment or data-governance requirements call for it.

## Prediction inputs

`frequency`, `monetary`, `total_units`, `average_line_revenue`, `unique_products`, `unique_countries`, `recency_days`, `customer_tenure_days`, and `average_order_value`.

Use the same units and feature definitions as the notebook. The target is at least one purchase in the following 90 days, not a permanent churn label. The notebook's test results show limitations in identifying non-purchasers; use the score as exploratory decision support and validate the model before operational use.
