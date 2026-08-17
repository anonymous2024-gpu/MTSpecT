from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


@dataclass
class TargetResult:
    target: str
    n_train: int
    n_test: int
    rmse: float
    mae: float
    r2: float


def _make_model(params: dict) -> Pipeline:
    rf = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=int(params.get("seed", 42)),
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", rf),
        ]
    )


def train_and_evaluate_per_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    targets: list[str],
    train_params: dict,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows: list[TargetResult] = []
    fitted_models: dict[str, Pipeline] = {}

    for target in targets:
        train_t = train_df.dropna(subset=[target])
        test_t = test_df.dropna(subset=[target])

        if train_t.empty or test_t.empty:
            continue

        X_train = train_t[feature_columns]
        y_train = train_t[target]
        X_test = test_t[feature_columns]
        y_test = test_t[target]

        model = _make_model(train_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        rows.append(
            TargetResult(
                target=target,
                n_train=len(train_t),
                n_test=len(test_t),
                rmse=rmse,
                mae=mae,
                r2=r2,
            )
        )
        fitted_models[target] = model

    result_df = pd.DataFrame([row.__dict__ for row in rows])
    return result_df, fitted_models
