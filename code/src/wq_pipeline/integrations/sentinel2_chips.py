from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class S2ChipConfig:
    stac_url: str
    collection: str
    tolerance_days: int
    limit: int
    max_cloud_cover: float | None
    max_solar_zenith: float | None
    chip_size: int
    overlap_stride_pixels: int
    emit_chip_images: bool
    chip_image_format: str
    chip_image_quality: int | None
    chip_image_rgb_bands: list[str]
    chip_image_stretch_percentiles: tuple[float, float]
    bands: list[str]
    max_rows: int | None
    max_nodata_ratio: float
    timeout_seconds: int


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


def _read_chip_array(
    href: str,
    lon: float,
    lat: float,
    chip_size: int,
    x_offset_px: int = 0,
    y_offset_px: int = 0,
) -> tuple[np.ndarray, float]:
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window

    signed = _sign_href_if_needed(href)
    with rasterio.open(signed) as ds:
        if ds.crs is None:
            arr = np.full((chip_size, chip_size), np.nan, dtype=np.float32)
            return arr, 1.0

        xs, ys = transform("EPSG:4326", ds.crs, [float(lon)], [float(lat)])
        px, py = ds.index(xs[0], ys[0])
        px = int(px + int(x_offset_px))
        py = int(py + int(y_offset_px))
        half = chip_size // 2
        window = Window(col_off=int(px - half), row_off=int(py - half), width=int(chip_size), height=int(chip_size))

        arr = ds.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None and np.isfinite(nodata):
            arr[np.isclose(arr, float(nodata))] = np.nan

        nan_ratio = float(np.isnan(arr).mean())
        return arr, nan_ratio


def _candidate_offsets(stride_px: int) -> list[tuple[int, int]]:
    if int(stride_px) <= 0:
        return [(0, 0)]
    s = int(stride_px)
    return [
        (0, 0),
        (-s, 0),
        (s, 0),
        (0, -s),
        (0, s),
        (-s, -s),
        (-s, s),
        (s, -s),
        (s, s),
    ]


def _normalize_rgb_uint8(stack: np.ndarray, band_names: list[str], rgb_bands: list[str], p_low: float, p_high: float) -> np.ndarray:
    band_index = {str(name).upper(): i for i, name in enumerate(band_names)}
    planes: list[np.ndarray] = []
    for band in rgb_bands:
        idx = band_index.get(str(band).upper())
        if idx is None:
            raise ValueError(f"RGB band '{band}' not available in stack bands: {band_names}")
        plane = stack[idx].astype(np.float32)
        finite = np.isfinite(plane)
        if not finite.any():
            out = np.zeros_like(plane, dtype=np.uint8)
            planes.append(out)
            continue
        low = np.nanpercentile(plane, float(p_low))
        high = np.nanpercentile(plane, float(p_high))
        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        scaled = (plane - low) / (high - low)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~finite] = 0.0
        out = (scaled * 255.0).round().astype(np.uint8)
        planes.append(out)
    return np.stack(planes, axis=-1)


def _write_chip_image(image_path: Path, rgb_uint8: np.ndarray, image_format: str, quality: int | None) -> None:
    import matplotlib.pyplot as plt

    image_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(image_format).lower()
    if fmt in {"jpg", "jpeg"}:
        pil_kwargs = {"quality": int(quality)} if quality is not None else None
        plt.imsave(image_path, rgb_uint8, format="jpg", pil_kwargs=pil_kwargs)
    elif fmt == "png":
        plt.imsave(image_path, rgb_uint8, format="png")
    else:
        raise ValueError(f"Unsupported chip image format: {image_format}")


def build_s2_image_chips(
    input_csv: Path,
    chips_dir: Path,
    output_index_csv: Path,
    cfg: S2ChipConfig,
) -> dict[str, Any]:
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise ImportError("rasterio is required for Sentinel-2 chip extraction.") from exc

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

    chips_dir.mkdir(parents=True, exist_ok=True)
    output_index_csv.parent.mkdir(parents=True, exist_ok=True)

    bands = [str(b).upper() for b in cfg.bands]
    stac_cache: dict[tuple[float, float, str, str, str], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    quality_counts = {
        "ok": 0,
        "no_match": 0,
        "cloud_reject": 0,
        "solar_reject": 0,
        "nodata_reject": 0,
        "stac_error": 0,
        "chip_error": 0,
    }

    for _, row in work.iterrows():
        sample_id = str(row["sample_id"])
        sample_date = pd.Timestamp(row["sample_date"])
        lon = float(row["lon"])
        lat = float(row["lat"])
        start = sample_date - timedelta(days=int(cfg.tolerance_days))
        end = sample_date + timedelta(days=int(cfg.tolerance_days))

        record = {
            "sample_id": sample_id,
            "sample_date": sample_date.date().isoformat(),
            "lat": lat,
            "lon": lon,
            "chip_path": "",
            "chip_image_path": "",
            "chip_status": "",
            "chip_reason": "",
            "s2_item_id": pd.NA,
            "acquisition_date": pd.NA,
            "abs_day_diff": pd.NA,
            "s2_cloud_cover": pd.NA,
            "s2_solar_zenith": pd.NA,
            "chip_nodata_ratio": pd.NA,
            "chip_size": int(cfg.chip_size),
            "chip_overlap_stride_pixels": int(cfg.overlap_stride_pixels),
            "chip_offset_x_px": 0,
            "chip_offset_y_px": 0,
            "chip_overlap_candidates": 0,
            "chip_bands": ",".join(bands),
        }

        cache_key = (
            round(lon, 6),
            round(lat, 6),
            str(cfg.collection),
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        try:
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
                    limit=int(cfg.limit),
                    timeout=int(cfg.timeout_seconds),
                )
                stac_cache[cache_key] = candidates
        except Exception as exc:
            record["chip_status"] = "stac_error"
            record["chip_reason"] = str(exc)
            quality_counts["stac_error"] += 1
            rows.append(record)
            continue

        best = _select_best_item(candidates, sample_date, cfg.max_cloud_cover)
        if best is None:
            record["chip_status"] = "no_match"
            record["chip_reason"] = "no_stac_item"
            quality_counts["no_match"] += 1
            rows.append(record)
            continue

        props = best.get("properties", {})
        acq = pd.to_datetime(props.get("datetime"), errors="coerce", utc=True)
        acq_naive = acq.tz_localize(None) if (not pd.isna(acq) and acq.tzinfo) else acq
        cloud = props.get("eo:cloud_cover")
        cloud_val = float(cloud) if cloud is not None and pd.notna(cloud) else np.nan
        solar_zen = props.get("s2:mean_solar_zenith")
        solar_zen_val = float(solar_zen) if solar_zen is not None and pd.notna(solar_zen) else np.nan

        record["s2_item_id"] = best.get("id")
        record["s2_cloud_cover"] = cloud_val if np.isfinite(cloud_val) else pd.NA
        record["s2_solar_zenith"] = solar_zen_val if np.isfinite(solar_zen_val) else pd.NA
        if not pd.isna(acq_naive):
            record["acquisition_date"] = acq_naive.date().isoformat()
            record["abs_day_diff"] = int(abs((acq_naive - sample_date).days))

        if cfg.max_cloud_cover is not None and np.isfinite(cloud_val) and cloud_val > float(cfg.max_cloud_cover):
            record["chip_status"] = "rejected"
            record["chip_reason"] = "cloud_reject"
            quality_counts["cloud_reject"] += 1
            rows.append(record)
            continue

        if cfg.max_solar_zenith is not None and np.isfinite(solar_zen_val) and solar_zen_val > float(cfg.max_solar_zenith):
            record["chip_status"] = "rejected"
            record["chip_reason"] = "solar_reject"
            quality_counts["solar_reject"] += 1
            rows.append(record)
            continue

        assets = best.get("assets", {})
        asset_hrefs: dict[str, str] = {}
        for band in bands:
            href = None
            if band in assets:
                href = assets[band].get("href")
            if href is None and band.lower() in assets:
                href = assets[band.lower()].get("href")
            if href is None:
                record["chip_status"] = "chip_error"
                record["chip_reason"] = f"missing_asset_{band}"
                quality_counts["chip_error"] += 1
                rows.append(record)
                asset_hrefs = {}
                break
            asset_hrefs[band] = str(href)

        if not asset_hrefs:
            continue

        stride_px = int(cfg.overlap_stride_pixels)
        offsets = _candidate_offsets(stride_px)
        best_stack: np.ndarray | None = None
        best_nodata_ratio = np.inf
        best_offset = (0, 0)
        chip_error = None

        for ox, oy in offsets:
            chip_planes: list[np.ndarray] = []
            offset_error = None
            for band in bands:
                href = asset_hrefs[band]
                try:
                    plane, _ = _read_chip_array(
                        href=href,
                        lon=lon,
                        lat=lat,
                        chip_size=int(cfg.chip_size),
                        x_offset_px=int(ox),
                        y_offset_px=int(oy),
                    )
                    chip_planes.append(plane)
                except Exception as exc:
                    offset_error = f"chip_error_{band}:{exc}"
                    break

            if offset_error is not None:
                chip_error = offset_error
                continue

            stack = np.stack(chip_planes, axis=0).astype(np.float32)
            nodata_ratio = float(np.isnan(stack).mean())
            if nodata_ratio < best_nodata_ratio:
                best_nodata_ratio = nodata_ratio
                best_stack = stack
                best_offset = (int(ox), int(oy))

            if nodata_ratio <= float(cfg.max_nodata_ratio):
                break

        if best_stack is None:
            record["chip_status"] = "chip_error"
            record["chip_reason"] = chip_error or "chip_error_all_offsets"
            quality_counts["chip_error"] += 1
            rows.append(record)
            continue

        record["chip_overlap_candidates"] = int(len(offsets))
        record["chip_offset_x_px"] = int(best_offset[0])
        record["chip_offset_y_px"] = int(best_offset[1])
        record["chip_nodata_ratio"] = float(best_nodata_ratio)
        if float(best_nodata_ratio) > float(cfg.max_nodata_ratio):
            record["chip_status"] = "rejected"
            record["chip_reason"] = "nodata_reject"
            quality_counts["nodata_reject"] += 1
            rows.append(record)
            continue

        chip_file = chips_dir / f"sample_{sample_id}.npz"
        np.savez_compressed(chip_file, chip=best_stack, bands=np.array(bands), sample_id=np.array([sample_id]))

        record["chip_path"] = str(chip_file)

        if bool(cfg.emit_chip_images):
            p_low, p_high = cfg.chip_image_stretch_percentiles
            rgb_bands = [str(v).upper() for v in cfg.chip_image_rgb_bands]
            rgb_uint8 = _normalize_rgb_uint8(
                stack=best_stack,
                band_names=bands,
                rgb_bands=rgb_bands,
                p_low=float(p_low),
                p_high=float(p_high),
            )
            ext = "jpg" if str(cfg.chip_image_format).lower() in {"jpg", "jpeg"} else "png"
            image_file = chips_dir / f"sample_{sample_id}.{ext}"
            _write_chip_image(
                image_path=image_file,
                rgb_uint8=rgb_uint8,
                image_format=str(cfg.chip_image_format),
                quality=cfg.chip_image_quality,
            )
            record["chip_image_path"] = str(image_file)

        record["chip_status"] = "ok"
        record["chip_reason"] = ""
        quality_counts["ok"] += 1
        rows.append(record)

    index_df = pd.DataFrame(rows)
    index_df.to_csv(output_index_csv, index=False)

    return {
        "input_csv": str(input_csv),
        "output_index_csv": str(output_index_csv),
        "chips_dir": str(chips_dir),
        "n_rows_input": int(len(df)),
        "n_rows_processed": int(len(work)),
        "n_rows_index": int(len(index_df)),
        "quality_counts": {k: int(v) for k, v in quality_counts.items()},
        "bands": bands,
        "chip_size": int(cfg.chip_size),
        "max_nodata_ratio": float(cfg.max_nodata_ratio),
        "stac_query_cache_size": int(len(stac_cache)),
    }


def build_multimodal_index(
    split_csv: Path,
    temporal_csv: Path | None,
    chips_index_csv: Path,
    output_csv: Path,
) -> dict[str, Any]:
    split_df = pd.read_csv(split_csv, low_memory=False)
    if "sample_id" not in split_df.columns:
        raise ValueError("Split CSV must contain sample_id.")

    temporal_df = None
    if temporal_csv is not None and temporal_csv.exists():
        temporal_df = pd.read_csv(temporal_csv, low_memory=False)
        if "sample_id" not in temporal_df.columns:
            temporal_df = None

    chips_df = pd.read_csv(chips_index_csv, low_memory=False)
    if "sample_id" not in chips_df.columns:
        raise ValueError("Chip index CSV must contain sample_id.")

    out = split_df.copy()
    if temporal_df is not None:
        temporal_cols = [c for c in temporal_df.columns if c.startswith("img_t") or c in {"sample_id", "temporal_feature_status", "temporal_feature_error"}]
        out = out.merge(temporal_df[temporal_cols], on="sample_id", how="left", suffixes=("", "_tmp"))

    chip_cols = [
        c
        for c in chips_df.columns
        if c in {
            "sample_id",
            "chip_path",
            "chip_image_path",
            "chip_status",
            "chip_reason",
            "s2_item_id",
            "acquisition_date",
            "abs_day_diff",
            "s2_cloud_cover",
            "s2_solar_zenith",
            "chip_nodata_ratio",
            "chip_size",
            "chip_overlap_stride_pixels",
            "chip_offset_x_px",
            "chip_offset_y_px",
            "chip_overlap_candidates",
            "chip_bands",
        }
    ]
    out = out.merge(chips_df[chip_cols], on="sample_id", how="left", suffixes=("", "_chip"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    chip_ok = int((out.get("chip_status") == "ok").sum()) if "chip_status" in out.columns else 0
    temporal_ok = int((out.get("temporal_feature_status") == "ok").sum()) if "temporal_feature_status" in out.columns else 0

    return {
        "split_csv": str(split_csv),
        "temporal_csv": str(temporal_csv) if temporal_csv is not None else None,
        "chips_index_csv": str(chips_index_csv),
        "output_csv": str(output_csv),
        "n_rows_output": int(len(out)),
        "n_chip_ok": chip_ok,
        "n_temporal_ok": temporal_ok,
    }
