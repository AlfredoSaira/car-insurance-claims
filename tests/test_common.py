import numpy as np
import pandas as pd

from common import gini_score, poisson_deviance, prepare_features, split

RAW = pd.DataFrame([{
    "Area": "C", "VehPower": 15, "VehAge": 40, "DrivAge": 99, "BonusMalus": 230,
    "VehBrand": "B12", "VehGas": "Diesel", "Density": 0, "Region": "R11"}])


def test_prepare_features_caps_and_log():
    out = prepare_features(RAW).iloc[0]
    assert out["VehPower"] == 12
    assert out["VehAge"] == 20
    assert out["DrivAge"] == 90
    assert out["BonusMalus"] == 150
    assert out["LogDensity"] == 0.0  # density clipped to 1 before the log


def test_gini_perfect_and_random():
    rng = np.random.default_rng(0)
    y = rng.poisson(0.1, 5000).astype(float)
    w = np.ones_like(y)
    assert abs(gini_score(y, y, w) - 1) < 1e-9
    assert abs(gini_score(y, rng.random(5000), w)) < 0.1


def test_poisson_deviance_zero_for_exact_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert poisson_deviance(y, y, np.ones(3)) < 1e-12


def test_split_sizes_are_60_20_20():
    df = pd.DataFrame({"x": range(1000)})
    train, val, test = split(df)
    assert (len(train), len(val), len(test)) == (600, 200, 200)
