from __future__ import annotations

import pandas as pd


def apply_basic_qc(
    df: pd.DataFrame,
    targets: list[str],
    min_targets_per_row: int = 1,
) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned.dropna(subset=["sample_date", "acquisition_date"])  # date parsing failures
    cleaned = cleaned[(cleaned["lat"].between(-90, 90)) & (cleaned["lon"].between(-180, 180))]

    for target in targets:
        if target in cleaned.columns:
            cleaned[target] = pd.to_numeric(cleaned[target], errors="coerce")
            cleaned.loc[cleaned[target] < 0, target] = pd.NA

    present_count = cleaned[targets].notna().sum(axis=1)
    cleaned = cleaned[present_count >= min_targets_per_row]

    return cleaned.reset_index(drop=True)
