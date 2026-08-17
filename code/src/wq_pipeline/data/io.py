from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_IN_SITU_COLUMNS = {
    "sample_id",
    "station_id",
    "lake_id",
    "sample_date",
    "lat",
    "lon",
}
REQUIRED_EO_COLUMNS = {"eo_id", "station_id", "acquisition_date"}


def _validate_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def load_in_situ(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_IN_SITU_COLUMNS, "in_situ")
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    return df


def load_eo_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_EO_COLUMNS, "eo_features")
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce")
    return df


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
