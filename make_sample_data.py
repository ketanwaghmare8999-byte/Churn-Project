"""Generate a synthetic dataset with the IBM Telco Churn schema (for testing only).

Use the real file instead: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000
yn = lambda p: np.where(rng.random(n) < p, "Yes", "No")

internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
has_net = internet != "No"
addon = lambda: np.where(has_net, yn(0.4), "No internet service")
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
tenure = rng.integers(0, 73, n)
monthly = np.round(20 + 30 * (internet == "DSL") + 55 * (internet == "Fiber optic")
                   + rng.normal(0, 5, n), 2).clip(18)
df = pd.DataFrame({
    "customerID": [f"C{i:05d}" for i in range(n)],
    "gender": rng.choice(["Male", "Female"], n),
    "SeniorCitizen": (rng.random(n) < 0.16).astype(int),
    "Partner": yn(0.48), "Dependents": yn(0.3), "tenure": tenure,
    "PhoneService": yn(0.9), "MultipleLines": yn(0.4), "InternetService": internet,
    "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
    "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(),
    "Contract": contract, "PaperlessBilling": yn(0.6),
    "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                 "Bank transfer (automatic)", "Credit card (automatic)"], n),
    "MonthlyCharges": monthly,
})
total = np.round(df["MonthlyCharges"] * df["tenure"] * rng.uniform(0.95, 1.05, n), 2)
df["TotalCharges"] = total.astype(str)
df.loc[df["tenure"] == 0, "TotalCharges"] = " "          # mimic the real data quirk

logit = (-0.6 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
         + 0.6 * (internet == "Fiber optic") - 0.035 * tenure + 0.012 * (monthly - 65)
         + 0.4 * (df["PaymentMethod"] == "Electronic check")
         - 0.5 * (df["TechSupport"] == "Yes") + rng.normal(0, 0.6, n))
df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
df.to_csv("data/sample_telco_churn.csv", index=False)
print(f"Wrote {n} rows, churn rate = {(df.Churn == 'Yes').mean():.1%}")
