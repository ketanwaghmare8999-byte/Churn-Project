"""Data cleaning, feature engineering and the shared preprocessing pipeline.

Everything that transforms raw customer data lives here so that training and
serving use *exactly* the same code (no train/serve skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
AUTOPAY_METHODS = {"Bank transfer (automatic)", "Credit card (automatic)"}

TENURE_BINS = [-1, 12, 24, 48, 60, np.inf]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-60m", "60m+"]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "charge_ratio", "num_addon_services", "is_autopay",
]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", *ADDON_SERVICES, "Contract", "PaperlessBilling",
    "PaymentMethod", "tenure_group",
]
RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", *ADDON_SERVICES, "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw CSV and fix known data-quality problems."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # TotalCharges contains blank strings for brand-new customers -> NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates(subset=ID_COLUMN).drop(columns=ID_COLUMN)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df.dropna(subset=[TARGET]).astype({TARGET: int})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from raw columns (stateless, safe for serving)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 1. Tenure bucket: churn risk is highly non-linear in customer age
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)

    # 2. Historical average monthly spend (falls back to current charge if new)
    safe_tenure = df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = (df["TotalCharges"] / safe_tenure).fillna(df["MonthlyCharges"])

    # 3. Price-change signal: current bill vs. historical average
    df["charge_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(0, np.nan)

    # 4. Engagement: number of add-on services subscribed
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)

    # 5. Automatic payment (a known retention driver)
    df["is_autopay"] = df["PaymentMethod"].isin(AUTOPAY_METHODS).astype(int)
    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering -> imputation -> scaling -> one-hot encoding."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(add_features, validate=False)),
        ("columns", columns),
    ])
