"""Web service: predict the yearly claim frequency of a car insurance policy.

    uv run uvicorn predict:app --host 0.0.0.0 --port 9696
    # then open http://localhost:9696/docs
"""

import math
import os
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import prepare_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "models" / "model.bin"))

with open(MODEL_PATH, "rb") as f:
    BUNDLE = pickle.load(f)
PIPELINE = BUNDLE["pipeline"]
TIERS = BUNDLE["tiers"]


class Policy(BaseModel):
    """One policy, with the same fields as the freMTPL2 dataset."""

    Area: Literal["A", "B", "C", "D", "E", "F"] = Field(
        description="Density class of the city, A (rural) to F (urban centre)")
    VehPower: int = Field(ge=1, le=20, description="Vehicle power class (4-15)")
    VehAge: int = Field(ge=0, le=100, description="Vehicle age in years")
    DrivAge: int = Field(ge=18, le=100, description="Driver age in years")
    BonusMalus: int = Field(ge=50, le=350, description="Bonus-malus level: 50 = best record, >100 = penalised")
    VehBrand: str = Field(pattern=r"^B\d{1,2}$", description="Anonymised car brand, B1-B14")
    VehGas: Literal["Regular", "Diesel"]
    Density: int = Field(ge=1, le=30000, description="Inhabitants per km2 where the driver lives")
    Region: str = Field(pattern=r"^R\d{2}$", description="French region code, e.g. R11 (Ile-de-France)")

    model_config = {"json_schema_extra": {"examples": [{
        "Area": "C", "VehPower": 7, "VehAge": 2, "DrivAge": 23, "BonusMalus": 90,
        "VehBrand": "B12", "VehGas": "Regular", "Density": 1200, "Region": "R11"}]}}


class Prediction(BaseModel):
    expected_claims_per_year: float = Field(description="Predicted claim frequency for one policy-year")
    claim_probability: float = Field(description="Probability of at least one claim in a year (Poisson)")
    risk_tier: Literal["low", "medium", "high"]
    relative_to_portfolio: float = Field(description="Expected claims divided by the portfolio average")


app = FastAPI(title="Car insurance claim frequency",
              description="Predicts how many claims a motor insurance policy will file per year.")


def score(policies: list[Policy]) -> list[Prediction]:
    raw = pd.DataFrame([p.model_dump() for p in policies])
    freqs = PIPELINE.predict(prepare_features(raw))
    out = []
    for lam in map(float, freqs):
        tier = "low" if lam <= TIERS["low_max"] else "medium" if lam <= TIERS["medium_max"] else "high"
        out.append(Prediction(expected_claims_per_year=round(lam, 4),
                              claim_probability=round(1 - math.exp(-lam), 4),
                              risk_tier=tier,
                              relative_to_portfolio=round(lam / BUNDLE["portfolio_frequency"], 2)))
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "xgboost-poisson", "test_metrics": BUNDLE["test_metrics"]}


@app.post("/predict", response_model=Prediction)
def predict(policy: Policy) -> Prediction:
    return score([policy])[0]


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(policies: list[Policy]) -> list[Prediction]:
    return score(policies)
