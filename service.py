"""Framework-independent inference logic (loaded once, reused by the API)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.features import RAW_FEATURES

MODEL_PATH = Path("models/churn_model.joblib")
METADATA_PATH = Path("models/metadata.json")
HIGH_RISK_PROBABILITY = 0.60


@dataclass
class ChurnModel:
    pipeline: object
    threshold: float
    metadata: dict

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH) -> "ChurnModel":
        metadata = json.loads(metadata_path.read_text())
        return cls(joblib.load(model_path), metadata["threshold"], metadata)

    def predict(self, records: list[dict]) -> list[dict]:
        """Score raw customer records; the saved pipeline does all preprocessing."""
        frame = pd.DataFrame.from_records(records)[RAW_FEATURES]
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            {
                "churn_probability": round(float(p), 4),
                "churn_prediction": bool(p >= self.threshold),
                "risk_level": self._risk_level(p),
            }
            for p in probabilities
        ]

    def _risk_level(self, p: float) -> str:
        if p >= HIGH_RISK_PROBABILITY:
            return "high"
        return "medium" if p >= self.threshold else "low"
