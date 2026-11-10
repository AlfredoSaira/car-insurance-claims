"""Shared data preparation, model builders and metrics.

Used by the notebook, train.py, predict.py and the tests, so the service
transforms a policy exactly the way the model saw it during training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "freMTPL2freq.csv.gz"
SEED = 42

CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
NUMERICAL = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
FEATURES = CATEGORICAL + NUMERICAL

# Caps decided in the EDA: a few extreme values carry almost no exposure but
# would let a model (especially the GLM) fit noise.
CAPS = {"VehPower": 12, "VehAge": 20, "DrivAge": 90, "BonusMalus": 150}
MAX_CLAIMS = 4  # 9 policies with 5-16 claims in a few months look like data errors


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw policies and add the target columns."""
    df = pd.read_csv(path)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=MAX_CLAIMS)
    df["Exposure"] = df["Exposure"].clip(upper=1.0)  # a policy-year is at most 1
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw policy fields into model features (works for 1 or N rows)."""
    out = pd.DataFrame(index=df.index)
    for col in CATEGORICAL:
        out[col] = df[col].astype(str)
    for col, cap in CAPS.items():
        out[col] = df[col].clip(upper=cap).astype(float)
    out["LogDensity"] = np.log(df["Density"].astype(float).clip(lower=1))
    return out[FEATURES]


def split(df: pd.DataFrame):
    """60/20/20 train/validation/test split, fixed seed."""
    full_train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    train, val = train_test_split(full_train, test_size=0.25, random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_glm(alpha: float = 1e-4) -> Pipeline:
    """Poisson GLM: the classic actuarial model. Splines let it bend the
    non-linear effects (young drivers, BonusMalus) while staying linear."""
    pre = ColumnTransformer(
        [
            ("cat", one_hot(), CATEGORICAL),
            ("num", Pipeline([("spline", SplineTransformer(n_knots=5, degree=3)),
                              ("scale", StandardScaler())]), NUMERICAL),
        ]
    )
    return Pipeline([("pre", pre),
                     ("model", PoissonRegressor(alpha=alpha, solver="newton-cholesky", max_iter=300))])


def tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([("cat", one_hot(), CATEGORICAL)], remainder="passthrough")


def poisson_deviance(y_true, y_pred, weight) -> float:
    """Exposure-weighted mean Poisson deviance (lower is better)."""
    return mean_poisson_deviance(y_true, np.clip(y_pred, 1e-9, None), sample_weight=weight)


def d2_score(y_true, y_pred, weight) -> float:
    """Share of the Poisson deviance explained vs. predicting the mean
    frequency for everyone (like R2, but for counts). Higher is better."""
    mean = np.average(y_true, weights=weight)
    null = poisson_deviance(y_true, np.full(len(y_true), mean), weight)
    return 1 - poisson_deviance(y_true, y_pred, weight) / null


def gini_score(y_true, y_pred, weight) -> float:
    """Normalized Gini of the ordered Lorenz curve: how well the model ranks
    policies from safest to riskiest (0 = random, 1 = perfect). Insurers use
    it because pricing is about ranking risk."""
    order = np.argsort(y_pred)
    w = np.asarray(weight)[order]
    claims = (np.asarray(y_true) * np.asarray(weight))[order]
    cum_w = np.concatenate([[0], np.cumsum(w) / w.sum()])
    cum_c = np.concatenate([[0], np.cumsum(claims) / claims.sum()])
    gini = 1 - 2 * np.trapezoid(cum_c, cum_w)

    perfect = np.argsort(np.asarray(y_true))
    w_p = np.asarray(weight)[perfect]
    c_p = (np.asarray(y_true) * np.asarray(weight))[perfect]
    cum_wp = np.concatenate([[0], np.cumsum(w_p) / w_p.sum()])
    cum_cp = np.concatenate([[0], np.cumsum(c_p) / c_p.sum()])
    return gini / (1 - 2 * np.trapezoid(cum_cp, cum_wp))


def evaluate(y_true, y_pred, weight) -> dict:
    return {
        "poisson_deviance": round(float(poisson_deviance(y_true, y_pred, weight)), 5),
        "d2": round(float(d2_score(y_true, y_pred, weight)), 5),
        "gini": round(float(gini_score(y_true, y_pred, weight)), 5),
    }
