# Customer Churn Prediction System

Production-style end-to-end machine learning system to predict whether a telecom customer will churn.

## Business Problem

Customer churn directly impacts recurring revenue and acquisition cost efficiency.  
The goal is to identify high-risk customers early enough to trigger retention actions (discount, support outreach, plan change, contract conversion).

## Dataset

- Source: Telco Customer Churn (public IBM/Kaggle-style dataset)
- Loaded automatically from configured public URLs (with fallback):
  - `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- Target column: `Churn` (`Yes` = churn, `No` = retained)

## Project Structure

```text
src/
  data/          # dataset loading + EDA
  features/      # feature engineering + preprocessing
  models/        # training, evaluation, threshold tuning, inference predictor
  api/           # FastAPI app + request/response schemas
  utils/         # logging + persistence helpers
  config/        # centralized config and paths
  artifacts/     # saved model, preprocessor, threshold, metrics
notebooks/       # notebook workspace
tests/           # unit and API tests
```

## ML Pipeline

### 1) Exploratory Data Analysis

Run:

```bash
python -m src.main --task eda
```

Outputs:
- `docs/eda_plots/class_distribution.png`
- `docs/eda_plots/numerical_correlation.png`
- categorical churn ratio plots
- `docs/eda_summary.md`

Business-oriented patterns to look for:
- Churn concentration in month-to-month contracts indicates lower commitment risk.
- Payment method and internet-service combinations can signal dissatisfaction/pricing sensitivity.
- Lower tenure cohorts often have disproportionately higher churn.

### 2) Preprocessing

- Missing values:
  - Numeric: median imputation
  - Categorical: most-frequent imputation
- Type cleanup:
  - `TotalCharges` coerced to numeric
- Encoding:
  - One-hot encoding for categorical columns
- Scaling:
  - Standard scaling for numeric columns
- Implemented with `ColumnTransformer` + `Pipeline`

### 3) Feature Engineering

Derived features:
- `tenure_group`: customer lifecycle buckets
- `charges_ratio`: `TotalCharges / MonthlyCharges` proxy for tenure-consistency and billing quality
- `service_count`: number of active services
- `contract_risk`: mapped contract risk category

Why they help:
- Tenure and contract terms capture commitment and switching friction.
- Service breadth can reflect account stickiness.
- Charge patterns can expose billing anomalies or low engagement.

### 4) Model Training

Compared models:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Training script:

```bash
python -m src.main --task train
```

### 5) Evaluation Metrics

Saved in `src/artifacts/metrics.json`:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Why accuracy is not enough:
- Churn is often imbalanced.
- A model can be "accurate" by over-predicting non-churn while missing true churners.
- Recall/precision tradeoff is usually the operational KPI for retention teams.

### 6) Threshold Tuning

- Validation-set threshold scan across configurable grid (`0.20` to `0.80` default)
- Best threshold selected using F1 with recall/precision tie-break preference
- Persisted to `src/artifacts/threshold.joblib`

Tradeoff:
- Lower threshold -> catch more churners (higher recall), but more false alarms
- Higher threshold -> fewer false alarms (higher precision), but more missed churners

### 7) Explainability

- Tree models: feature importance from `feature_importances_`
- Logistic regression: absolute coefficient magnitude
- Plot saved to `src/artifacts/feature_importance.png`
- Top factors saved in `metrics.json`

## Inference API (FastAPI)

Start API:

```bash
python -m src.main --task serve --host 127.0.0.1 --port 8000
```

Endpoints:
- `POST /predict`
- `POST /predict_proba`
- `GET /health`
- `GET /` (simple web UI)

### Web UI

After starting the API, open:
- `http://127.0.0.1:8000/`

The UI provides a customer input form and calls `/predict` under the hood.

### Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 6,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 537.0
}
```

### Example Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.74,
  "threshold_used": 0.4
}
```

## Input Validation and Robustness

- Pydantic request schema in `src/api/schemas.py`
- Graceful malformed input handling (HTTP 400)
- Startup check for model artifacts (HTTP 503 if missing)
- Inference request logging in `logs/inference.log`

## Tests

Run:

```bash
pytest -q
```

Included tests:
- preprocessing transformation behavior
- engineered feature creation
- prediction response schema format
- API endpoint contract (`/predict`, `/predict_proba`)

## Reproducibility

- Fixed random seed in config (`random_state=42`)
- Config-driven parameters in `src/config/settings.py`
- Separated train-time and inference-time logic

## Quickstart

```bash
pip install -r requirements.txt
python -m src.main --task eda
python -m src.main --task train
python -m src.main --task serve
pytest -q
```

## Run Commands Section

- EDA: `python -m src.main --task eda`
- Train: `python -m src.main --task train`
- Serve API: `python -m src.main --task serve`
- Test: `pytest -q`
