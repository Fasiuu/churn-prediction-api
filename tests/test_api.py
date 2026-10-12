import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Customer

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]

LOYAL = {**EXAMPLE, "tenure": 60, "Contract": "Two year", "PaymentMethod": "Credit card (automatic)",
         "InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "Yes", "MonthlyCharges": 55.0,
         "TotalCharges": 3300.0, "Partner": "Yes", "Dependents": "Yes", "PaperlessBilling": "No"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_version"] == "1.0.0"
    assert 0 < info["threshold"] < 1
    assert info["metrics_holdout"]["test_roc_auc"] > 0.8


def test_predict_example(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"}
    assert len(body["top_factors"]) == 3
    assert r.headers["x-request-id"]


def test_new_month_to_month_customer_is_riskier_than_loyal_one(client):
    risky = client.post("/predict", json=EXAMPLE).json()
    loyal = client.post("/predict", json=LOYAL).json()
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["will_churn"] and not loyal["will_churn"]


def test_invalid_category_is_rejected(client):
    bad = {**EXAMPLE, "Contract": "Weekly"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_missing_field_is_rejected(client):
    bad = copy.deepcopy(EXAMPLE)
    del bad["tenure"]
    assert client.post("/predict", json=bad).status_code == 422


def test_inconsistent_input_is_rejected(client):
    bad = {**EXAMPLE, "InternetService": "No"}          # but add-ons still say "No"/"Yes"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "No internet service" in r.text


def test_batch(client):
    r = client.post("/predict/batch", json={"customers": [EXAMPLE, LOYAL]})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_api_matches_pipeline_on_real_rows(client):
    """The API should give the same probabilities as calling the saved pipeline directly."""
    from app.main import state
    df = pd.read_csv("data/Telco-Customer-Churn.csv").head(20)
    rows = df.drop(columns=["customerID", "Churn"]).copy()
    rows["TotalCharges"] = pd.to_numeric(rows["TotalCharges"], errors="coerce").fillna(0.0)
    payload = rows.to_dict(orient="records")
    api = [p["churn_probability"] for p in client.post("/predict/batch", json={"customers": payload}).json()["predictions"]]
    direct = state["model"].pipeline.predict_proba(rows)[:, 1]
    assert max(abs(a - b) for a, b in zip(api, direct, strict=True)) < 1e-3


def test_metrics_endpoint(client):
    client.get("/health")
    text = client.get("/metrics").text
    assert "http_requests_total" in text and "churn_predictions_total" in text
