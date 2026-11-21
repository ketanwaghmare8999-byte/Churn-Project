"""End-to-end API tests (skipped automatically if FastAPI or the model is missing)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402
from src.service import MODEL_PATH  # noqa: E402
from tests.test_service import RISKY  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # context manager runs the lifespan (model load)
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    body = client.post("/predict", json=RISKY).json()
    assert set(body) == {"churn_probability", "churn_prediction", "risk_level"}


def test_batch(client):
    assert len(client.post("/predict/batch", json=[RISKY, RISKY]).json()) == 2


def test_invalid_payload_rejected(client):
    assert client.post("/predict", json={**RISKY, "Contract": "Lifetime"}).status_code == 422
