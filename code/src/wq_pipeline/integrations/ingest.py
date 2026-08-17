from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_table(path: Path, fmt: str | None = None) -> pd.DataFrame:
    file_format = (fmt or path.suffix.lower().lstrip(".")).lower()
    if file_format == "csv":
        return pd.read_csv(path)
    if file_format in {"tsv", "txt"}:
        return pd.read_csv(path, sep="\t")
    if file_format in {"parquet", "pq"}:
        return pd.read_parquet(path)
    if file_format == "json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {file_format}")


def _rename_columns(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    inverse = {v: k for k, v in column_map.items() if v in df.columns}
    return df.rename(columns=inverse)


def _ensure_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns after mapping: {missing}")


def standardize_in_situ_table(
    input_path: Path,
    output_path: Path,
    column_map: dict[str, str],
    targets: list[str],
    fmt: str | None = None,
) -> dict[str, Any]:
    raw = _read_table(input_path, fmt=fmt)
    mapped = _rename_columns(raw, column_map)

    required = ["sample_id", "station_id", "lake_id", "sample_date", "lat", "lon"]
    _ensure_columns(mapped, required, "in_situ")

    for target in targets:
        if target not in mapped.columns:
            mapped[target] = pd.NA

    mapped["sample_date"] = pd.to_datetime(mapped["sample_date"], errors="coerce")
    mapped = mapped.dropna(subset=["sample_date"]).copy()

    final_columns = required + targets
    standardized = mapped[final_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, index=False)

    return {
        "rows": int(len(standardized)),
        "columns": final_columns,
        "targets_non_null": {t: int(standardized[t].notna().sum()) for t in targets},
    }


def standardize_eo_table(
    input_path: Path,
    output_path: Path,
    column_map: dict[str, str],
    fmt: str | None = None,
) -> dict[str, Any]:
    raw = _read_table(input_path, fmt=fmt)
    mapped = _rename_columns(raw, column_map)

    required = ["eo_id", "station_id", "acquisition_date"]
    _ensure_columns(mapped, required, "eo")

    mapped["acquisition_date"] = pd.to_datetime(mapped["acquisition_date"], errors="coerce")
    mapped = mapped.dropna(subset=["acquisition_date"]).copy()

    keep_cols = required + mapped.select_dtypes(include=["number"]).columns.tolist()
    keep_cols = list(dict.fromkeys(keep_cols))
    standardized = mapped[keep_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, index=False)

    return {
        "rows": int(len(standardized)),
        "columns": keep_cols,
    }


def write_ingest_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
