"""Train the final claim-frequency model and save it to models/model.bin.

The model family and hyper-parameters come from the tuning in
notebook.ipynb, which writes them to models/best_params.json.

    uv run python train.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from common import evaluate, load_data, prepare_features, split, tree_preprocessor

MODELS = Path(__file__).parent / "models"
PARAMS_PATH = MODELS / "best_params.json"
MODEL_PATH = MODELS / "model.bin"

# Fallback if the notebook has not been run (same values it selected).
DEFAULT_PARAMS = {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 50,
                  "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 900}


def build_model(params: dict) -> Pipeline:
    model = xgb.XGBRegressor(objective="count:poisson", tree_method="hist",
                             n_jobs=-1, random_state=42, **params)
    return Pipeline([("pre", tree_preprocessor()), ("model", model)])


def main() -> None:
    params = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else DEFAULT_PARAMS
    print("Hyper-parameters:", params)

    df = load_data()
    train, val, test = split(df)
    full_train = pd.concat([train, val], ignore_index=True)  # final fit on train + validation

    pipeline = build_model(params)
    pipeline.fit(prepare_features(full_train), full_train["Frequency"],
                 model__sample_weight=full_train["Exposure"])

    test_pred = pipeline.predict(prepare_features(test))
    metrics = evaluate(test["Frequency"], test_pred, test["Exposure"])
    print("Test metrics:", metrics)

    # Risk tiers: cut-offs on the predicted yearly frequency of the training
    # portfolio (50% of policies are "low", the top 10% are "high").
    train_pred = pipeline.predict(prepare_features(full_train))
    tiers = {"low_max": float(np.quantile(train_pred, 0.5)),
             "medium_max": float(np.quantile(train_pred, 0.9))}

    MODELS.mkdir(exist_ok=True)
    bundle = {"pipeline": pipeline, "tiers": tiers, "params": params,
              "test_metrics": {k: float(v) for k, v in metrics.items()},
              "portfolio_frequency": float(np.average(full_train["Frequency"], weights=full_train["Exposure"]))}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model to {MODEL_PATH} (tiers: {tiers})")


if __name__ == "__main__":
    main()
