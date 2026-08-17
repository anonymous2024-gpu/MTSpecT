from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests


@dataclass(frozen=True)
class S2MetadataConfig:
    stac_url: str
    collection: str
    tolerance_days: int
    limit: int
    max_rows: int | None


def _search_s2_item(
    stac_url: str,
    collection: str,
    lon: float,
    lat: float,
    sample_date: pd.Timestamp,
    tolerance_days: int,
    limit: int,
    timeout: int = 90,
) -> dict | None:
    start = (sample_date - timedelta(days=tolerance_days)).strftime("%Y-%m-%dT00:00:00Z")
    end = (sample_date + timedelta(days=tolerance_days)).strftime("%Y-%m-%dT23:59:59Z")

    body = {
        "collections": [collection],
        "limit": limit,
        "datetime": f"{start}/{end}",
        "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "sortby": [{"field": "datetime", "direction": "asc"}],
    }

    response = requests.post(stac_url.rstrip("/") + "/search", json=body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features", [])
    if not features:
        return None

    best = None
    best_diff = None
    for feature in features:
        dt_str = feature.get("properties", {}).get("datetime")
        if not dt_str:
            continue
        item_dt = pd.to_datetime(dt_str, errors="coerce", utc=True)
        if pd.isna(item_dt):
            continue
        diff = abs((item_dt.tz_localize(None) - sample_date).days)
        if best is None or diff < best_diff:
            best = feature
            best_diff = diff

    return best


def build_s2_metadata_features(
    in_situ_csv: Path,
    output_eo_csv: Path,
    cfg: S2MetadataConfig,
) -> dict:
    in_situ = pd.read_csv(in_situ_csv)
    required = ["sample_id", "station_id", "sample_date", "lat", "lon"]
    missing = [col for col in required if col not in in_situ.columns]
    if missing:
        raise ValueError(f"in_situ missing required columns for S2 metadata: {missing}")

    in_situ["sample_date"] = pd.to_datetime(in_situ["sample_date"], errors="coerce")
    use = in_situ.dropna(subset=["sample_date", "lat", "lon"]).copy()
    if cfg.max_rows is not None and cfg.max_rows > 0:
        use = use.head(int(cfg.max_rows)).copy()

    query_cols = ["station_id", "sample_date", "lat", "lon"]
    query_df = use[query_cols].drop_duplicates(ignore_index=True)

    query_results: dict[tuple[str, str, str, str], dict] = {}
    query_matched = 0
    for _, row in query_df.iterrows():
        sample_date = pd.Timestamp(row["sample_date"])
        query_key = (
            str(row["station_id"]),
            sample_date.date().isoformat(),
            f"{float(row['lat']):.7f}",
            f"{float(row['lon']):.7f}",
        )

        try:
            item = _search_s2_item(
                stac_url=cfg.stac_url,
                collection=cfg.collection,
                lon=float(row["lon"]),
                lat=float(row["lat"]),
                sample_date=sample_date,
                tolerance_days=cfg.tolerance_days,
                limit=cfg.limit,
            )
        except requests.RequestException:
            item = None

        record = {
            "acquisition_date": pd.NA,
            "abs_day_diff": pd.NA,
            "s2_cloud_cover": pd.NA,
            "s2_solar_zenith": pd.NA,
            "s2_solar_azimuth": pd.NA,
            "s2_view_zenith": pd.NA,
            "s2_view_azimuth": pd.NA,
            "s2_mgrs_tile": pd.NA,
            "s2_item_id": pd.NA,
        }

        if item is not None:
            props = item.get("properties", {})
            acq = pd.to_datetime(props.get("datetime"), errors="coerce")
            if not pd.isna(acq):
                record["acquisition_date"] = acq.date().isoformat()
                record["abs_day_diff"] = abs((acq.tz_localize(None) - sample_date).days) if acq.tzinfo else abs((acq - sample_date).days)
            record["s2_cloud_cover"] = props.get("eo:cloud_cover")
            record["s2_solar_zenith"] = props.get("s2:mean_solar_zenith")
            record["s2_solar_azimuth"] = props.get("s2:mean_solar_azimuth")
            record["s2_view_zenith"] = props.get("s2:mean_view_zenith")
            record["s2_view_azimuth"] = props.get("s2:mean_view_azimuth")
            record["s2_mgrs_tile"] = props.get("s2:mgrs_tile")
            record["s2_item_id"] = item.get("id")
            query_matched += 1

        query_results[query_key] = record

    rows = []
    matched = 0
    for _, row in use.iterrows():
        sample_date = pd.Timestamp(row["sample_date"])
        query_key = (
            str(row["station_id"]),
            sample_date.date().isoformat(),
            f"{float(row['lat']):.7f}",
            f"{float(row['lon']):.7f}",
        )
        record = query_results.get(query_key, {})
        out = {
            "eo_id": f"S2_{row['sample_id']}",
            "station_id": row["station_id"],
            "acquisition_date": record.get("acquisition_date", pd.NA),
            "abs_day_diff": record.get("abs_day_diff", pd.NA),
            "s2_cloud_cover": record.get("s2_cloud_cover", pd.NA),
            "s2_solar_zenith": record.get("s2_solar_zenith", pd.NA),
            "s2_solar_azimuth": record.get("s2_solar_azimuth", pd.NA),
            "s2_view_zenith": record.get("s2_view_zenith", pd.NA),
            "s2_view_azimuth": record.get("s2_view_azimuth", pd.NA),
            "s2_mgrs_tile": record.get("s2_mgrs_tile", pd.NA),
            "s2_item_id": record.get("s2_item_id", pd.NA),
        }

        if pd.notna(out["acquisition_date"]):
            matched += 1

        rows.append(out)

    eo = pd.DataFrame(rows)
    output_eo_csv.parent.mkdir(parents=True, exist_ok=True)
    eo.to_csv(output_eo_csv, index=False)

    return {
        "n_input_rows": int(len(use)),
        "n_output_rows": int(len(eo)),
        "n_matched_items": int(matched),
        "match_rate": float(matched / max(1, len(eo))),
        "n_unique_query_points": int(len(query_df)),
        "query_reduction_ratio": float(len(query_df) / max(1, len(use))),
        "n_unique_matched_items": int(query_matched),
        "collection": cfg.collection,
        "stac_url": cfg.stac_url,
    }
