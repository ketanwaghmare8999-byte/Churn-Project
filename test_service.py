"""Inference tests (need a trained model: run `python -m src.train` first)."""
import pytest

from src.service import MODEL_PATH, ChurnModel

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")

RISKY = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 100.0, "TotalCharges": 200.0,
}
SAFE = {**RISKY, "tenure": 70, "Contract": "Two year", "TechSupport": "Yes",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 60.0, "TotalCharges": 4200.0}


@pytest.fixture(scope="module")
def model():
    return ChurnModel.load()


def test_output_shape_and_ranges(model):
    out = model.predict([RISKY])[0]
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert isinstance(out["churn_prediction"], bool)
    assert out["risk_level"] in {"low", "medium", "high"}


def test_risky_customer_scores_higher_than_safe(model):
    risky, safe = model.predict([RISKY, SAFE])
    assert risky["churn_probability"] > safe["churn_probability"]


def test_new_customer_without_total_charges(model):
    out = model.predict([{**RISKY, "tenure": 0, "TotalCharges": None}])[0]
    assert 0.0 <= out["churn_probability"] <= 1.0
