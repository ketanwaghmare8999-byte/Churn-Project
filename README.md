# Customer Churn Prediction Pipeline

End-to-end supervised ML project: clean data → engineer features → train & tune three models →
pick the winner by **business impact** → serve it through a **FastAPI** REST endpoint.

## Project structure
```
src/features.py   cleaning, 5 engineered features, scaling + one-hot preprocessing
src/train.py      tuning (CV), evaluation, cost-based threshold, model selection, saving
src/service.py    framework-free inference wrapper
src/api.py        FastAPI app (/predict, /predict/batch, /health, /model-info)
tests/            service + API tests
scripts/make_sample_data.py   synthetic data with the Telco schema (testing only)
```

## Quick start
```bash
pip install -r requirements.txt

# 1. Get the real dataset (IBM Telco Customer Churn, ~7,043 rows):
#    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#    save as data/WA_Fn-UseC_-Telco-Customer-Churn.csv
#    (no download? `python scripts/make_sample_data.py` makes a synthetic stand-in)

# 2. Train (writes models/churn_model.joblib + models/metadata.json)
python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
#    optional: --offer-cost 20 --customer-value 500 --success-rate 0.3

# 3. Serve
uvicorn src.api:app --reload          # Swagger docs at http://127.0.0.1:8000/docs

# 4. Test
pytest -q
```

## Example request
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":3,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":95.5,"TotalCharges":280.3}'
# {"churn_probability":0.93,"churn_prediction":true,"risk_level":"high"}
```

## Design decisions
| Requirement | How it is met |
|---|---|
| Cleaning | `TotalCharges` blanks → NaN → median imputation; duplicates/ID dropped; target → 0/1 |
| One-hot encoding | `OneHotEncoder(handle_unknown="ignore")` so unseen categories never crash the API |
| ≥3 engineered features | `tenure_group`, `avg_monthly_spend`, `charge_ratio` (price-change signal), `num_addon_services`, `is_autopay` |
| Scaling | `StandardScaler` on numeric columns, inside the pipeline |
| ≥3 models + tuning | Logistic Regression (grid), Random Forest and Gradient Boosting (randomized), 5-fold stratified CV, scored on AUC-ROC |
| Metrics | AUC-ROC, PR-AUC, precision, recall, F1, and **net campaign profit** |
| Business-driven selection | Winner = highest cross-validated profit per customer, not highest accuracy |
| Deployment | Pydantic-validated FastAPI, model loaded once at startup, auto-generated docs |

**Avoiding leakage / skew:** the whole preprocessing chain is part of one scikit-learn `Pipeline`, so
it is fit only on training folds and the identical object is used at serving time. The decision threshold
and the winning model are chosen from out-of-fold predictions on the training set; the 20% test set is
used only for final reporting.

**Business case (editable via CLI flags):** each flagged customer costs $20 (offer); a contacted churner
accepts 30% of the time and is worth $500. Net benefit = TP × 0.30 × $500 − flagged × $20, so contacting
anyone with churn probability above ≈13% is profitable. This is why the chosen threshold is well below 0.5 and
recall is prioritised over precision. Replace these numbers with your company's real figures.
