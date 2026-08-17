from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import requests


OAPIF_BASE = "https://vm4072.kaj.pouta.csc.fi/ddas/oapif/collections/fin_sykewaterquality"


@dataclass(frozen=True)
class FinlandFetchConfig:
    start_date: str
    end_date: str
    station_limit: int | None
    parameters: list[str]
    depth_layer: str


DEFAULT_PARAMETER_MAP = {
    "Chlorophyll a": "chl_a",
    "Turbidity": "turbidity",
    "Secchi depth": "secchi_depth",
    "Dissolved organic carbon": "doc",
    "Total phosphorus": "tp",
    "Total nitrogen": "tn",
    "Total organic carbon": "toc",
}


def _normalize_parameter_name(name: str | None) -> str | None:
    if not name:
        return None
    text = str(name).strip().lower()

    if "chlorophyll" in text:
        return "chl_a"
    if "secchi" in text:
        return "secchi_depth"
    if "dissolved organic carbon" in text or re.search(r"\bdoc\b", text):
        return "doc"
    if "total organic carbon" in text or re.search(r"\btoc\b", text):
        return "toc"
    if "turbidity" in text:
        return "turbidity"
    if "total phosphorous" in text or "total phosphorus" in text:
        return "tp"
    if "total nitrogen" in text:
        return "tn"
    return None


def _to_iso(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    return dt.strftime("%Y-%m-%dT00:00:00Z")


def _fetch_station_index(timeout: int = 120) -> list[dict]:
    url = f"{OAPIF_BASE}/items"
    response = requests.get(url, params={"f": "json"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("features", []))


def _extract_station_meta(feature: dict) -> dict:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]
    props = feature.get("properties") or {}
    station_id = str(feature.get("id"))
    return {
        "station_id": station_id,
        "lon": coordinates[0],
        "lat": coordinates[1],
        "name": props.get("name"),
        "lakename": props.get("lakename"),
        "municipality": props.get("municipality"),
    }


def _fetch_station_all_series(
    station_id: str,
    start_iso: str,
    end_iso: str,
    timeout: int = 120,
) -> tuple[dict, list[dict]]:
    url = f"{OAPIF_BASE}/items/{station_id}"
    params = {
        "f": "json",
        "datetime": f"{start_iso}/{end_iso}",
    }
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    geometry = payload.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]
    props = payload.get("properties") or {}

    station_meta = {
        "station_id": str(payload.get("id", station_id)),
        "lon": coordinates[0],
        "lat": coordinates[1],
        "name": props.get("name"),
        "lakename": props.get("lakename"),
        "municipality": props.get("municipality"),
    }

    values = props.get("property") or []
    rows: list[dict] = []
    for row in values:
        rows.append(
            {
                "station_id": station_meta["station_id"],
                "sample_date": row.get("date"),
                "parameter": row.get("parameter"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "layer": row.get("layer"),
            }
        )

    return station_meta, rows


def _pivot_to_canonical(
    long_df: pd.DataFrame,
    station_meta_df: pd.DataFrame,
    parameter_map: dict[str, str],
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["sample_id", "station_id", "lake_id", "sample_date", "lat", "lon", *sorted(set(parameter_map.values()))])

    use_df = long_df.copy()
    use_df["parameter"] = use_df["parameter"].astype(str)
    use_df["target_name"] = use_df["parameter"].map(_normalize_parameter_name)
    use_df["sample_date"] = pd.to_datetime(use_df["sample_date"], errors="coerce").dt.date
    use_df = use_df.dropna(subset=["sample_date", "value", "target_name"])

    wide = (
        use_df.pivot_table(
            index=["station_id", "sample_date"],
            columns="target_name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    merged = wide.merge(station_meta_df, on="station_id", how="left")
    merged["lake_id"] = merged["lakename"].fillna(merged["station_id"])
    merged["sample_id"] = [f"FI_{idx:08d}" for idx in range(len(merged))]
    merged["sample_date"] = pd.to_datetime(merged["sample_date"]).astype(str)

    target_columns = sorted(set(parameter_map.values()))
    keep = ["sample_id", "station_id", "lake_id", "sample_date", "lat", "lon", *target_columns]
    for column in target_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[keep]


def fetch_finland_syke_dataset(
    output_in_situ_csv: Path,
    output_raw_long_csv: Path,
    output_station_csv: Path,
    cfg: FinlandFetchConfig,
) -> dict:
    start_iso = _to_iso(cfg.start_date)
    end_iso = _to_iso(cfg.end_date)

    index_features = _fetch_station_index()
    station_meta = [_extract_station_meta(feature) for feature in index_features]

    station_meta = [
        row
        for row in station_meta
        if pd.notna(row.get("lat")) and pd.notna(row.get("lon"))
    ]

    if cfg.station_limit and cfg.station_limit > 0:
        station_meta = station_meta[: cfg.station_limit]

    station_df = pd.DataFrame(station_meta)

    all_rows: list[dict] = []
    station_rows: list[dict] = []

    for station_id in station_df["station_id"].astype(str).tolist():
        try:
            station_meta_row, rows = _fetch_station_all_series(
                station_id=station_id,
                start_iso=start_iso,
                end_iso=end_iso,
            )
        except requests.RequestException:
            continue

        if rows:
            all_rows.extend(rows)
            station_rows.append(station_meta_row)

    long_df = pd.DataFrame(all_rows)
    station_out_df = pd.DataFrame(station_rows).drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    if not station_out_df.empty:
        station_out_df = station_out_df.merge(
            station_df[["station_id", "lat", "lon", "name", "lakename", "municipality"]].rename(
                columns={
                    "lat": "lat_index",
                    "lon": "lon_index",
                    "name": "name_index",
                    "lakename": "lakename_index",
                    "municipality": "municipality_index",
                }
            ),
            on="station_id",
            how="left",
        )
        station_out_df["lat"] = station_out_df["lat"].fillna(station_out_df["lat_index"])
        station_out_df["lon"] = station_out_df["lon"].fillna(station_out_df["lon_index"])
        station_out_df["name"] = station_out_df["name"].fillna(station_out_df["name_index"])
        station_out_df["lakename"] = station_out_df["lakename"].fillna(station_out_df["lakename_index"])
        station_out_df["municipality"] = station_out_df["municipality"].fillna(station_out_df["municipality_index"])
        station_out_df = station_out_df.drop(columns=["lat_index", "lon_index", "name_index", "lakename_index", "municipality_index"])

    in_situ_df = _pivot_to_canonical(long_df, station_out_df, DEFAULT_PARAMETER_MAP)

    output_in_situ_csv.parent.mkdir(parents=True, exist_ok=True)
    output_raw_long_csv.parent.mkdir(parents=True, exist_ok=True)
    output_station_csv.parent.mkdir(parents=True, exist_ok=True)

    in_situ_df.to_csv(output_in_situ_csv, index=False)
    long_df.to_csv(output_raw_long_csv, index=False)
    station_out_df.to_csv(output_station_csv, index=False)

    return {
        "n_stations_indexed": int(len(station_df)),
        "n_stations_with_data": int(len(station_out_df)),
        "n_long_rows": int(len(long_df)),
        "n_samples_wide": int(len(in_situ_df)),
        "targets_non_null": {column: int(in_situ_df[column].notna().sum()) for column in in_situ_df.columns if column not in {"sample_id", "station_id", "lake_id", "sample_date", "lat", "lon"}},
    }
