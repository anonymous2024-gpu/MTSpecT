from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class VeslaODataConfig:
    service_root: str
    start_date: str
    end_date: str
    top: int
    max_pages: int
    apply_date_filter: bool


def _build_url(root: str, entity: str, query: dict[str, str]) -> str:
    base = root.rstrip("/") + "/" + entity.lstrip("/")
    if not query:
        return base
    parts = [f"{key}={value}" for key, value in query.items() if value]
    return base + "?" + "&".join(parts)


def _fetch_odata_pages(url: str, max_pages: int, timeout: int = 120) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    next_url = url
    page_count = 0

    while next_url and page_count < max_pages:
        response = requests.get(next_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("value", [])
        if rows:
            all_rows.extend(rows)
        next_url = payload.get("odata.nextLink")
        page_count += 1

    return pd.DataFrame(all_rows)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_parameter_name(name: str | None) -> str | None:
    if name is None or pd.isna(name):
        return None
    text = str(name).strip().lower()
    if not text:
        return None

    if "chlorophyll" in text:
        return "chl_a"
    if "secchi" in text:
        return "secchi_depth"
    if "turbidity" in text:
        return "turbidity"
    if "dissolved organic carbon" in text or re.search(r"\bdoc\b", text):
        return "doc"
    if "total organic carbon" in text or re.search(r"\btoc\b", text):
        return "toc"
    if "total phosphorous" in text or "total phosphorus" in text:
        return "tp"
    if "total nitrogen" in text:
        return "tn"
    if "dissolved oxygen" in text:
        return "dissolved_oxygen"
    if "conductivity" in text:
        return "conductivity"
    if text == "ph":
        return "ph"
    if "temperature" in text:
        return "temperature"
    if "suspended solids" in text:
        return "suspended_solids"
    if "nitrite nitrate as nitrogen" in text:
        return "nitrite_nitrate_n"
    if "ammonium as nitrogen" in text:
        return "ammonium_n"
    if "phosphate as phosphorous" in text or "phosphate as phosphorus" in text:
        return "phosphate_p"
    return None


def _choose_parameter_label(df: pd.DataFrame) -> pd.Series:
    columns = [
        "AnalyteName",
        "DeterminationName",
        "AnalyteNameFI",
        "DeterminationNameFI",
        "Analyte",
        "AnalysisMethod",
    ]
    label = pd.Series([None] * len(df), index=df.index, dtype="object")
    for column in columns:
        if column in df.columns:
            candidate = df[column].astype("string")
            label = label.fillna(candidate)
    return label.astype("string")


def _build_parameter_catalog(long_df: pd.DataFrame) -> pd.DataFrame:
    catalog = (
        long_df.groupby(["parameter", "target_name"], dropna=False)
        .agg(
            n_records=("value", "count"),
            n_stations=("station_id", pd.Series.nunique),
            first_date=("sample_date", "min"),
            last_date=("sample_date", "max"),
        )
        .reset_index()
        .sort_values(["n_records", "n_stations"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return catalog


def fetch_vesla_odata_dataset(
    output_in_situ_csv: Path,
    output_long_csv: Path,
    output_site_csv: Path,
    output_catalog_csv: Path,
    output_determination_csv: Path,
    output_report_json: Path,
    cfg: VeslaODataConfig,
) -> dict[str, Any]:
    root = cfg.service_root.rstrip("/")

    result_url = _build_url(root, "Result_Wide", {})
    site_url = _build_url(root, "Site_Wide", {})
    determination_url = _build_url(root, "Determination", {})

    results_df = _fetch_odata_pages(result_url, max_pages=cfg.max_pages)
    sites_df = _fetch_odata_pages(site_url, max_pages=5)
    determination_df = _fetch_odata_pages(determination_url, max_pages=10)

    if results_df.empty:
        raise ValueError("No Result_Wide rows returned from VESLA OData.")

    sites = sites_df.copy()
    sites["station_id"] = sites["Site_Id"].astype(str)
    sites["lat"] = sites.get("CoordEUREFFIN_WGS84_Lat", pd.Series([None] * len(sites))).map(_parse_float)
    sites["lon"] = sites.get("CoordEUREFFIN_WGS84_Long", pd.Series([None] * len(sites))).map(_parse_float)
    sites["lake_id"] = sites.get("LakeCode", pd.Series([None] * len(sites))).fillna(sites.get("WaterbodyCode", pd.Series([None] * len(sites))))
    sites["lake_id"] = sites["lake_id"].fillna(sites.get("Lake", pd.Series([None] * len(sites))))
    sites["lake_id"] = sites["lake_id"].fillna(sites["station_id"])
    site_requested = [
        "station_id",
        "Name",
        "lake_id",
        "lat",
        "lon",
        "EnvironmentType",
        "Municipal",
        "WaterManagementAreaCode",
        "WaterManagementArea",
        "WaterbasinCode",
        "Waterbasin",
        "WaterbodyType",
        "WaterbodyType_Id",
    ]
    site_available = [column for column in site_requested if column in sites.columns]
    site_min = sites[site_available].copy()

    long_df = results_df.copy()
    long_df["station_id"] = long_df["Site_Id"].astype(str)
    long_df["sample_date"] = pd.to_datetime(long_df["Time"], errors="coerce", utc=True).dt.tz_localize(None)
    if cfg.apply_date_filter:
        start_dt = pd.to_datetime(cfg.start_date)
        end_dt = pd.to_datetime(cfg.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        long_df = long_df[(long_df["sample_date"] >= start_dt) & (long_df["sample_date"] <= end_dt)].copy()
    long_df["value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df["parameter"] = _choose_parameter_label(long_df)
    long_df["target_name"] = long_df["parameter"].map(_normalize_parameter_name)

    long_df = long_df.dropna(subset=["sample_date", "value", "parameter"])
    long_df = long_df.merge(site_min, on="station_id", how="left")

    long_requested = [
        "station_id",
        "sample_date",
        "parameter",
        "target_name",
        "value",
        "Unit",
        "SampleDepth_m",
        "SampleDepthUpper_m",
        "SampleDepthLower_m",
        "Flag",
        "AnalyteCode",
        "DeterminationCode",
        "Name",
        "lake_id",
        "lat",
        "lon",
        "EnvironmentType",
        "Municipal",
        "WaterManagementAreaCode",
        "WaterManagementArea",
        "WaterbasinCode",
        "Waterbasin",
        "WaterbodyType",
        "WaterbodyType_Id",
    ]
    long_available = [column for column in long_requested if column in long_df.columns]
    long_export = long_df[long_available].rename(columns={"Unit": "unit"})

    canonical_df = long_export.dropna(subset=["target_name"]).copy()
    wide = (
        canonical_df.pivot_table(
            index=["station_id", "sample_date"],
            columns="target_name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide = wide.merge(site_min[["station_id", "lake_id", "lat", "lon"]], on="station_id", how="left")
    wide["sample_id"] = [f"VESLA_{idx:010d}" for idx in range(len(wide))]
    wide["sample_date"] = pd.to_datetime(wide["sample_date"]).astype(str)

    ordered_columns = ["sample_id", "station_id", "lake_id", "sample_date", "lat", "lon"]
    target_cols = [column for column in wide.columns if column not in ordered_columns]
    target_cols = sorted([column for column in target_cols if column not in {"sample_date", "station_id", "lake_id", "lat", "lon", "sample_id"}])
    in_situ = wide[ordered_columns + target_cols].copy()

    catalog = _build_parameter_catalog(long_export)

    output_in_situ_csv.parent.mkdir(parents=True, exist_ok=True)
    output_long_csv.parent.mkdir(parents=True, exist_ok=True)
    output_site_csv.parent.mkdir(parents=True, exist_ok=True)
    output_catalog_csv.parent.mkdir(parents=True, exist_ok=True)
    output_determination_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.parent.mkdir(parents=True, exist_ok=True)

    in_situ.to_csv(output_in_situ_csv, index=False)
    long_export.to_csv(output_long_csv, index=False)
    site_min.to_csv(output_site_csv, index=False)
    catalog.to_csv(output_catalog_csv, index=False)
    if not determination_df.empty:
        determination_export = determination_df.copy()
        for name_col in ["AnalyteName", "AnalyteNameFI", "DeterminationName", "DeterminationNameFI", "Analyte", "AnalysisMethod"]:
            if name_col not in determination_export.columns:
                determination_export[name_col] = pd.NA
        determination_export["target_name"] = _choose_parameter_label(determination_export).map(_normalize_parameter_name)
        determination_export.to_csv(output_determination_csv, index=False)
        determination_target_counts = (
            determination_export["target_name"].dropna().value_counts().sort_values(ascending=False).to_dict()
        )
    else:
        pd.DataFrame(
            columns=[
                "Determination_Id",
                "AnalyteName",
                "AnalyteNameFI",
                "DeterminationName",
                "DeterminationNameFI",
                "Analyte",
                "AnalysisMethod",
                "target_name",
            ]
        ).to_csv(
            output_determination_csv, index=False
        )
        determination_target_counts = {}

    report = {
        "service_root": root,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "max_pages": cfg.max_pages,
        "top": cfg.top,
        "n_result_rows": int(len(results_df)),
        "n_long_rows": int(len(long_export)),
        "n_sites": int(site_min["station_id"].nunique()),
        "n_determinations": int(len(determination_df)),
        "n_samples_wide": int(len(in_situ)),
        "n_target_columns": len(target_cols),
        "targets": target_cols,
        "target_non_null": {column: int(in_situ[column].notna().sum()) for column in target_cols},
        "determination_target_counts": determination_target_counts,
        "sample_date_min": str(long_export["sample_date"].min()) if not long_export.empty else None,
        "sample_date_max": str(long_export["sample_date"].max()) if not long_export.empty else None,
    }
    output_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
