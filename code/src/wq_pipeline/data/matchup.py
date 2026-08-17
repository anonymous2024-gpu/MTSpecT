from __future__ import annotations

import pandas as pd


def create_matchups(
    in_situ_df: pd.DataFrame,
    eo_df: pd.DataFrame,
    date_tolerance_days: int = 3,
    require_same_station: bool = True,
) -> pd.DataFrame:
    if require_same_station:
        merged = in_situ_df.merge(eo_df, on="station_id", how="inner", suffixes=("", "_eo"))
    else:
        in_situ_tmp = in_situ_df.assign(_k=1)
        eo_tmp = eo_df.assign(_k=1)
        merged = in_situ_tmp.merge(eo_tmp, on="_k", how="inner", suffixes=("", "_eo")).drop(columns=["_k"])

    merged["abs_day_diff"] = (merged["sample_date"] - merged["acquisition_date"]).abs().dt.days
    matched = merged[merged["abs_day_diff"] <= int(date_tolerance_days)].copy()

    matched = matched.sort_values(["sample_id", "abs_day_diff", "eo_id"]).drop_duplicates(subset=["sample_id"], keep="first")
    return matched.reset_index(drop=True)
