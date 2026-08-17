from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class S2TemporalConfig:
    stac_url: str
    collection: str
    temporal_offsets_days: list[int]
    offset_tolerance_days: int
    max_cloud_cover: float | None
    max_rows: int | None
    bands: list[str]
    timeout_seconds: int
    progress_every_rows: int = 25


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _post_stac_search(
    stac_url: str,
    collection: str,
    lon: float,
    lat: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    limit: int,
    timeout: int,
) -> list[dict[str, Any]]:
    body = {
        "collections": [collection],
        "limit": int(limit),
        "datetime": f"{start_dt.strftime('%Y-%m-%dT00:00:00Z')}/{end_dt.strftime('%Y-%m-%dT23:59:59Z')}",
        "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "sortby": [{"field": "datetime", "direction": "asc"}],
    }
    response = requests.post(stac_url.rstrip("/") + "/search", json=body, timeout=int(timeout))
    response.raise_for_status()
    payload = response.json()
    return payload.get("features", [])


def _select_best_item(
    items: list[dict[str, Any]],
    target_dt: pd.Timestamp,
    max_cloud_cover: float | None,
) -> dict[str, Any] | None:
    best = None
    best_key: tuple[float, float] | None = None

    for item in items:
        props = item.get("properties", {})
        dt_val = pd.to_datetime(props.get("datetime"), errors="coerce", utc=True)
        if pd.isna(dt_val):
            continue
        cloud = props.get("eo:cloud_cover")
        cloud_val = float(cloud) if cloud is not None and pd.notna(cloud) else np.nan
        if max_cloud_cover is not None and np.isfinite(cloud_val) and cloud_val > float(max_cloud_cover):
            continue

        dt_naive = dt_val.tz_localize(None) if dt_val.tzinfo else dt_val
        day_diff = float(abs((dt_naive - target_dt).days))
        cloud_rank = cloud_val if np.isfinite(cloud_val) else 9999.0
        key = (day_diff, cloud_rank)
        if best is None or key < best_key:
            best = item
            best_key = key

    return best


def _sign_href_if_needed(href: str) -> str:
    if "planetarycomputer.microsoft.com" not in href.lower() and "blob.core.windows.net" not in href.lower():
        return href
    try:
        import planetary_computer as pc
    except ImportError:
        return href
    try:
        return pc.sign(href)
    except Exception:
        return href


def _sample_asset_value(href: str, lon: float, lat: float) -> float:
    import rasterio
    from rasterio.warp import transform

    signed_href = _sign_href_if_needed(href)
    with rasterio.open(signed_href) as ds:
        src_crs = ds.crs
        if src_crs is None:
            return float("nan")
        xs, ys = transform("EPSG:4326", src_crs, [float(lon)], [float(lat)])
        sample = next(ds.sample([(xs[0], ys[0])]))
        if sample is None or len(sample) == 0:
            return float("nan")
        val = float(sample[0])
        if not np.isfinite(val):
            return float("nan")
        nodata = ds.nodata
        if nodata is not None and np.isfinite(nodata) and abs(val - float(nodata)) < 1e-12:
            return float("nan")
        return val


def build_temporal_point_features(input_csv: Path, output_csv: Path, cfg: S2TemporalConfig) -> dict[str, Any]:
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise ImportError("rasterio is required for temporal Sentinel-2 feature extraction.") from exc

    df = pd.read_csv(input_csv, low_memory=False)
    required = ["sample_id", "sample_date", "lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    work = df.dropna(subset=["sample_date", "lat", "lon"]).copy()

    if cfg.max_rows is not None and cfg.max_rows > 0:
        work = work.head(int(cfg.max_rows)).copy()

    offsets = [int(v) for v in cfg.temporal_offsets_days]
    bands = [str(b).upper() for b in cfg.bands]

    rows: list[dict[str, Any]] = []
    matched_any = 0
    total_rows = int(len(work))
    start_time = time.perf_counter()
    processed_rows = 0
    progress_every = max(1, int(getattr(cfg, "progress_every_rows", 25)))
    stac_cache: dict[tuple[str, float, float, str, str], list[dict[str, Any]]] = {}

    for _, row in work.iterrows():
        sample_date = pd.Timestamp(row["sample_date"])
        lon = float(row["lon"])
        lat = float(row["lat"])

        out = row.to_dict()
        out["temporal_feature_status"] = "ok"
        out["temporal_feature_error"] = ""

        row_matched = False
        for t_idx, offset in enumerate(offsets):
            target_date = sample_date + timedelta(days=int(offset))
            start = target_date - timedelta(days=int(cfg.offset_tolerance_days))
            end = target_date + timedelta(days=int(cfg.offset_tolerance_days))

            step_prefix = f"img_t{t_idx}"
            out[f"{step_prefix}_offset_days"] = int(offset)
            out[f"{step_prefix}_acq_date"] = pd.NA
            out[f"{step_prefix}_day_diff"] = pd.NA
            out[f"{step_prefix}_cloud"] = pd.NA
            out[f"{step_prefix}_item_id"] = pd.NA

            for band in bands:
                out[f"{step_prefix}_{band.lower()}"] = np.nan

            try:
                cache_key = (
                    str(cfg.collection),
                    round(float(lon), 6),
                    round(float(lat), 6),
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                )
                if cache_key in stac_cache:
                    candidates = stac_cache[cache_key]
                else:
                    candidates = _post_stac_search(
                        stac_url=cfg.stac_url,
                        collection=cfg.collection,
                        lon=lon,
                        lat=lat,
                        start_dt=start,
                        end_dt=end,
                        limit=50,
                        timeout=cfg.timeout_seconds,
                    )
                    stac_cache[cache_key] = candidates
            except Exception as exc:
                out["temporal_feature_status"] = "stac_error"
                out["temporal_feature_error"] = str(exc)
                continue

            best = _select_best_item(candidates, target_date, cfg.max_cloud_cover)
            if best is None:
                continue

            props = best.get("properties", {})
            best_dt = pd.to_datetime(props.get("datetime"), errors="coerce", utc=True)
            best_dt_naive = best_dt.tz_localize(None) if (not pd.isna(best_dt) and best_dt.tzinfo) else best_dt

            out[f"{step_prefix}_item_id"] = best.get("id")
            out[f"{step_prefix}_cloud"] = props.get("eo:cloud_cover")
            if not pd.isna(best_dt_naive):
                out[f"{step_prefix}_acq_date"] = best_dt_naive.date().isoformat()
                out[f"{step_prefix}_day_diff"] = int(abs((best_dt_naive - target_date).days))

            assets = best.get("assets", {})
            for band in bands:
                href = None
                if band in assets:
                    href = assets[band].get("href")
                if href is None:
                    alt_key = band.lower()
                    if alt_key in assets:
                        href = assets[alt_key].get("href")
                if href is None:
                    continue
                try:
                    out[f"{step_prefix}_{band.lower()}"] = _sample_asset_value(href, lon=lon, lat=lat)
                    row_matched = True
                except Exception:
                    out[f"{step_prefix}_{band.lower()}"] = np.nan

        if row_matched:
            matched_any += 1
        elif out["temporal_feature_status"] == "ok":
            out["temporal_feature_status"] = "no_match"

        rows.append(out)
        processed_rows += 1

        elapsed_seconds = time.perf_counter() - start_time
        rows_per_sec = (processed_rows / elapsed_seconds) if elapsed_seconds > 0 else 0.0
        remaining_rows = max(0, total_rows - processed_rows)
        eta_seconds = (remaining_rows / rows_per_sec) if rows_per_sec > 0 else 0.0
        percent = (100.0 * processed_rows / total_rows) if total_rows > 0 else 100.0

        if processed_rows % progress_every == 0 or processed_rows == total_rows:
            print(
                (
                    f"[temporal] {processed_rows}/{total_rows} "
                    f"({percent:6.2f}%) | "
                    f"elapsed {_format_duration(elapsed_seconds)} | "
                    f"eta {_format_duration(eta_seconds)} | "
                    f"{rows_per_sec:7.3f} rows/s"
                ),
                flush=True,
            )

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "n_rows_input": int(len(df)),
        "n_rows_processed": int(len(work)),
        "n_rows_output": int(len(out_df)),
        "n_rows_with_any_temporal_match": int(matched_any),
        "temporal_offsets_days": offsets,
        "progress_every_rows": int(progress_every),
        "stac_query_cache_size": int(len(stac_cache)),
        "bands": bands,
        "collection": cfg.collection,
        "stac_url": cfg.stac_url,
    }
