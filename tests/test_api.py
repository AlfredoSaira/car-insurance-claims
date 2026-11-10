"""API tests. They need models/model.bin (run train.py first)."""

from fastapi.testclient import TestClient

from predict import app

client = TestClient(app)

YOUNG_RISKY = {"Area": "E", "VehPower": 7, "VehAge": 0, "DrivAge": 19, "BonusMalus": 120,
               "VehBrand": "B1", "VehGas": "Regular", "Density": 4000, "Region": "R11"}
EXPERIENCED_SAFE = {"Area": "A", "VehPower": 5, "VehAge": 8, "DrivAge": 55, "BonusMalus": 50,
                    "VehBrand": "B12", "VehGas": "Diesel", "Density": 30, "Region": "R24"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_numbers():
    r = client.post("/predict", json=YOUNG_RISKY)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["expected_claims_per_year"] < 2
    assert 0 < body["claim_probability"] < 1
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_risky_profile_scores_higher_than_safe_profile():
    risky, safe = client.post("/predict_batch", json=[YOUNG_RISKY, EXPERIENCED_SAFE]).json()
    assert risky["expected_claims_per_year"] > 2 * safe["expected_claims_per_year"]
    assert risky["risk_tier"] == "high"
    assert safe["risk_tier"] == "low"


def test_invalid_input_is_rejected():
    r = client.post("/predict", json={**YOUNG_RISKY, "DrivAge": 12, "VehGas": "Electric"})
    assert r.status_code == 422
