from __future__ import annotations

import pandas as pd


def select_feature_columns(
    df: pd.DataFrame,
    targets: list[str],
    exclude_columns: list[str],
) -> list[str]:
    blocked = set(exclude_columns) | set(targets)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    features = [column for column in numeric if column not in blocked]
    if not features:
        raise ValueError("No numeric feature columns found after exclusions.")
    return features
