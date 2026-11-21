"""REST API for churn prediction.

Run:   uvicorn src.api:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.service import ChurnModel

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """Raw customer attributes, matching the Telco churn dataset columns."""
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0,
                                          description="Omit for brand-new customers")

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 280.3,
    }}}


class Prediction(BaseModel):
    churn_probability: float = Field(description="Estimated probability of churning, 0-1")
    churn_prediction: bool = Field(description="True if probability >= business-optimised threshold")
    risk_level: Literal["low", "medium", "high"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ChurnModel.load()   # load once at startup, not per request
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Returns churn probability and a retention-campaign flag for telecom customers.",
    version="1.0.0",
    lifespan=lifespan,
)


def _model(request: Request) -> ChurnModel:
    return request.app.state.model


@app.get("/health", tags=["meta"])
def health(request: Request) -> dict:
    return {"status": "ok", "model": _model(request).metadata["model_name"]}


@app.get("/model-info", tags=["meta"])
def model_info(request: Request) -> dict:
    m = _model(request).metadata
    return {"model": m["model_name"], "threshold": m["threshold"],
            "business_case": m["business_case"],
            "test_metrics": m["results"][m["model_name"]]["test"]}


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(customer: Customer, request: Request) -> dict:
    """Score a single customer."""
    return _model(request).predict([customer.model_dump()])[0]


@app.post("/predict/batch", response_model=list[Prediction], tags=["prediction"])
def predict_batch(customers: list[Customer], request: Request) -> list[dict]:
    """Score up to 1,000 customers in one call."""
    if not 0 < len(customers) <= 1000:
        raise HTTPException(status_code=422, detail="Send between 1 and 1000 customers.")
    return _model(request).predict([c.model_dump() for c in customers])
