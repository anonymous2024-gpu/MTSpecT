from __future__ import annotations

from pathlib import Path
import json
import copy
import re
from typing import Any

import typer
import pandas as pd

from wq_pipeline.config import PipelineConfig, load_config
from wq_pipeline.data.io import ensure_dir
from wq_pipeline.integrations.fetch import build_eea_download_url, download_file
from wq_pipeline.integrations.ingest import (
    standardize_eo_table,
    standardize_in_situ_table,
    write_ingest_report,
)
from wq_pipeline.integrations.metadata_llm import (
    suggest_mapping_from_columns,
    suggest_mapping_with_llm,
)
from wq_pipeline.integrations.sentinel2_stac import S2MetadataConfig, build_s2_metadata_features
from wq_pipeline.integrations.sentinel2_temporal import S2TemporalConfig, build_temporal_point_features
from wq_pipeline.integrations.sentinel2_chips import S2ChipConfig, build_multimodal_index, build_s2_image_chips
from wq_pipeline.integrations.syke_finland import FinlandFetchConfig, fetch_finland_syke_dataset
from wq_pipeline.integrations.vesla_odata import VeslaODataConfig, fetch_vesla_odata_dataset
from wq_pipeline.training.pretrained import run_train_pretrained
from wq_pipeline.training.fusion import run_train_fusion
from wq_pipeline.training.region_transfer import run_region_transfer
from wq_pipeline.training.run import run_prepare, run_train
from wq_pipeline.visualize import create_visualizations

app = typer.Typer(help="Water quality parameter estimation pipeline.")


def _bbox_from_mgrs_tile(tile: str) -> tuple[float, float, float, float] | None:
    try:
        import mgrs
    except ImportError:
        return None

    code = str(tile).strip().upper()
    if len(code) != 5:
        return None

    try:
        converter = mgrs.MGRS()
        sw_lat, sw_lon = converter.toLatLon(f"{code}0000000000")
        ne_lat, ne_lon = converter.toLatLon(f"{code}9999999999")
    except Exception:
        return None

    min_lat = float(min(sw_lat, ne_lat))
    max_lat = float(max(sw_lat, ne_lat))
    min_lon = float(min(sw_lon, ne_lon))
    max_lon = float(max(sw_lon, ne_lon))
    return min_lon, min_lat, max_lon, max_lat


def _load_finland_boundary_gdf() -> "Any":
    try:
        import geopandas as gpd
    except ImportError:
        return None

    source_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    try:
        world = gpd.read_file(source_url)
    except Exception:
        return None

    for col in ["ADMIN", "NAME_EN", "NAME", "SOVEREIGNT"]:
        if col in world.columns:
            fin = world.loc[world[col].astype(str).str.lower() == "finland"].copy()
            if not fin.empty:
                if fin.crs is None:
                    fin = fin.set_crs("EPSG:4326")
                return fin.to_crs("EPSG:4326")
    return None


def _normalize_name_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_extended_parameter(name: str | None) -> str | None:
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None

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
    if "conductivity" in text:
        return "conductivity"
    if "dissolved oxygen" in text:
        return "dissolved_oxygen"
    if "oxygen saturation" in text:
        return "oxygen_saturation"
    if text == "ph":
        return "ph"
    if "temperature" in text:
        return "temperature"
    if "suspended solids" in text:
        return "suspended_solids"
    return None


@app.command("prepare")
def prepare(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    run_prepare(cfg, root=Path.cwd())


@app.command("prepare-matchup-sweep")
def prepare_matchup_sweep(
    config: str = typer.Option(..., help="Path to YAML config."),
    tolerances: str = typer.Option(
        "1,3,5,7",
        help="Comma-separated matchup tolerance windows in days (e.g., 1,3,5,7).",
    ),
    output_csv: str = typer.Option(
        "artifacts/prepared/matchup_tolerance_sweep.csv",
        help="Output CSV summary path.",
    ),
    output_json: str = typer.Option(
        "artifacts/prepared/matchup_tolerance_sweep.json",
        help="Output JSON summary path.",
    ),
) -> None:
    base_cfg = load_config(config)
    root = Path.cwd()

    try:
        tol_values = sorted({int(token.strip()) for token in tolerances.split(",") if token.strip() != ""})
    except ValueError as exc:
        raise typer.BadParameter("`tolerances` must be a comma-separated list of integers.") from exc

    if not tol_values:
        raise typer.BadParameter("At least one tolerance value is required.")

    records: list[dict[str, Any]] = []
    for tol in tol_values:
        raw = copy.deepcopy(base_cfg.raw)
        raw.setdefault("matchup", {})
        raw["matchup"]["date_tolerance_days"] = int(tol)
        raw["artifacts_dir"] = str(Path(base_cfg.artifacts_dir) / f"tolerance_{tol}d")

        cfg_tol = PipelineConfig(raw=raw)
        run_prepare(cfg_tol, root=root)

        stats_path = (root / str(cfg_tol.artifacts_dir) / "prepared" / "dataset_stats.json").resolve()
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        n_total = int(stats.get("n_total", 0))
        n_train = int(stats.get("n_train", 0))
        n_valid = int(stats.get("n_valid", 0))
        n_test = int(stats.get("n_test", 0))

        records.append(
            {
                "tolerance_days": int(tol),
                "artifacts_dir": str(cfg_tol.artifacts_dir),
                "n_total": n_total,
                "n_train": n_train,
                "n_valid": n_valid,
                "n_test": n_test,
                "train_ratio": (float(n_train) / float(n_total)) if n_total > 0 else 0.0,
                "valid_ratio": (float(n_valid) / float(n_total)) if n_total > 0 else 0.0,
                "test_ratio": (float(n_test) / float(n_total)) if n_total > 0 else 0.0,
            }
        )

    summary_df = pd.DataFrame(records).sort_values("tolerance_days").reset_index(drop=True)

    out_csv = (root / output_csv).resolve()
    out_json = (root / output_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    summary = {
        "config": str((root / config).resolve()),
        "tolerances": tol_values,
        "results": summary_df.to_dict(orient="records"),
        "output_csv": str(out_csv),
        "output_json": str(out_json),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


@app.command("train")
def train(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    run_train(cfg, root=Path.cwd())


@app.command("train-pretrained")
def train_pretrained(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    try:
        run_train_pretrained(cfg, root=Path.cwd())
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@app.command("train-fusion")
def train_fusion(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    try:
        run_train_fusion(cfg, root=Path.cwd())
    except (ImportError, ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@app.command("train-region-transfer")
def train_region_transfer(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    try:
        run_region_transfer(cfg, root=Path.cwd())
    except (ImportError, ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@app.command("fetch")
def fetch(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    fetch_cfg = cfg.fetch.get("eea", {})
    if not fetch_cfg:
        raise typer.BadParameter("Missing `fetch.eea` section in config.")

    dataset_id = str(fetch_cfg.get("dataset_id", "")).strip()
    resource_path = str(fetch_cfg.get("resource_path", "")).strip()
    base_url = str(fetch_cfg.get("base_url", "")).strip()
    output_path = Path(fetch_cfg.get("output_path", "data/external/eea_download.csv"))

    if not (dataset_id and resource_path and base_url):
        raise typer.BadParameter("fetch.eea requires base_url, dataset_id, and resource_path.")

    url = build_eea_download_url(base_url=base_url, dataset_id=dataset_id, resource_path=resource_path)
    destination = (Path.cwd() / output_path).resolve()
    download_file(url=url, output_path=destination)
    typer.echo(f"Downloaded: {destination}")


@app.command("ingest")
def ingest(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    ingest_cfg = cfg.data_ingest
    if not ingest_cfg:
        raise typer.BadParameter("Missing `data_ingest` section in config.")

    in_situ_cfg = ingest_cfg.get("in_situ", {})
    eo_cfg = ingest_cfg.get("eo", {})
    if not in_situ_cfg or not eo_cfg:
        raise typer.BadParameter("`data_ingest` requires both `in_situ` and `eo` blocks.")

    in_report = standardize_in_situ_table(
        input_path=(Path.cwd() / str(in_situ_cfg["input_path"])).resolve(),
        output_path=(Path.cwd() / str(in_situ_cfg["output_path"])).resolve(),
        column_map=dict(in_situ_cfg.get("column_map", {})),
        targets=cfg.targets,
        fmt=in_situ_cfg.get("input_format"),
    )

    eo_report = standardize_eo_table(
        input_path=(Path.cwd() / str(eo_cfg["input_path"])).resolve(),
        output_path=(Path.cwd() / str(eo_cfg["output_path"])).resolve(),
        column_map=dict(eo_cfg.get("column_map", {})),
        fmt=eo_cfg.get("input_format"),
    )

    report = {"in_situ": in_report, "eo": eo_report}
    report_path = (Path.cwd() / str(cfg.artifacts_dir) / "ingest" / "report.json").resolve()
    write_ingest_report(report_path, report)
    typer.echo(f"Ingest completed. Report: {report_path}")


@app.command("suggest-mapping")
def suggest_mapping(
    input_csv: str = typer.Option(..., help="Path to raw table to inspect."),
    use_llm: bool = typer.Option(False, help="Use LLM API in addition to heuristic mapping."),
    config: str | None = typer.Option(None, help="Optional YAML config for llm settings."),
) -> None:
    table = pd.read_csv((Path.cwd() / input_csv).resolve(), nrows=5)
    columns = table.columns.tolist()

    heuristic = suggest_mapping_from_columns(columns)
    payload = {
        "heuristic": [
            {"canonical": row.canonical, "source_column": row.source_column, "confidence": row.confidence}
            for row in heuristic
        ]
    }

    if use_llm:
        if not config:
            raise typer.BadParameter("--config is required when --use-llm is enabled.")
        cfg = load_config(config)
        payload["llm"] = suggest_mapping_with_llm(columns, cfg.llm)

    out_dir = ensure_dir((Path.cwd() / "artifacts" / "mapping").resolve())
    out_path = out_dir / "suggested_mapping.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Mapping suggestions written to: {out_path}")


@app.command("fetch-finland-syke")
def fetch_finland_syke(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    fi_cfg = cfg.finland_syke
    if not fi_cfg:
        raise typer.BadParameter("Missing `finland_syke` section in config.")

    outputs = fi_cfg.get("outputs", {})
    in_situ_csv = (Path.cwd() / str(outputs.get("in_situ_csv", "data/raw/in_situ_finland.csv"))).resolve()
    long_csv = (Path.cwd() / str(outputs.get("long_csv", "data/external/finland_syke_long.csv"))).resolve()
    station_csv = (Path.cwd() / str(outputs.get("station_csv", "data/external/finland_syke_stations.csv"))).resolve()

    fetch_conf = FinlandFetchConfig(
        start_date=str(fi_cfg.get("start_date", "2023-01-01")),
        end_date=str(fi_cfg.get("end_date", "2025-01-01")),
        station_limit=int(fi_cfg.get("station_limit", 250)) if fi_cfg.get("station_limit") is not None else None,
        parameters=list(fi_cfg.get("parameters", ["Chlorophyll a", "Turbidity", "Secchi depth", "Dissolved organic carbon"])),
        depth_layer=str(fi_cfg.get("depth_layer", "1.0")),
    )

    report = fetch_finland_syke_dataset(
        output_in_situ_csv=in_situ_csv,
        output_raw_long_csv=long_csv,
        output_station_csv=station_csv,
        cfg=fetch_conf,
    )

    out_dir = ensure_dir((Path.cwd() / str(cfg.artifacts_dir) / "finland_syke").resolve())
    report_path = out_dir / "fetch_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Fetched Finland SYKE dataset. Report: {report_path}")


@app.command("visualize")
def visualize(
    in_situ_csv: str = typer.Option(..., help="Path to canonical in-situ CSV."),
    long_csv: str | None = typer.Option(None, help="Optional path to long-format parameter CSV."),
    output_dir: str = typer.Option("artifacts/figures", help="Output directory for figure files."),
    timeseries_parameter: str = typer.Option(
        "Chlorophyll a",
        help="Parameter name for long-format timeseries plot (case-insensitive).",
    ),
) -> None:
    in_path = (Path.cwd() / in_situ_csv).resolve()
    long_path = (Path.cwd() / long_csv).resolve() if long_csv else None
    out_path = (Path.cwd() / output_dir).resolve()

    report = create_visualizations(
        in_situ_csv=in_path,
        long_csv=long_path,
        output_dir=out_path,
        timeseries_parameter=timeseries_parameter,
    )
    report_file = out_path / "visualization_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Visualizations written under: {out_path}")


@app.command("build-s2-metadata")
def build_s2_metadata(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    s2_cfg = cfg.s2_metadata
    if not s2_cfg:
        raise typer.BadParameter("Missing `s2_metadata` section in config.")

    in_situ_path = (Path.cwd() / str(s2_cfg.get("input_in_situ_csv", "data/raw/in_situ_finland.csv"))).resolve()
    eo_out_path = (Path.cwd() / str(s2_cfg.get("output_eo_csv", "data/raw/eo_features_finland_s2meta.csv"))).resolve()

    conf = S2MetadataConfig(
        stac_url=str(s2_cfg.get("stac_url", "https://planetarycomputer.microsoft.com/api/stac/v1")),
        collection=str(s2_cfg.get("collection", "sentinel-2-l2a")),
        tolerance_days=int(s2_cfg.get("tolerance_days", 5)),
        limit=int(s2_cfg.get("limit", 10)),
        max_rows=int(s2_cfg.get("max_rows")) if s2_cfg.get("max_rows") is not None else None,
    )

    report = build_s2_metadata_features(in_situ_csv=in_situ_path, output_eo_csv=eo_out_path, cfg=conf)
    out_dir = ensure_dir((Path.cwd() / str(cfg.artifacts_dir) / "s2_metadata").resolve())
    report_file = out_dir / "report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Built S2 metadata EO table: {eo_out_path}")


@app.command("build-s2-temporal-features")
def build_s2_temporal_features(
    config: str = typer.Option(..., help="Path to YAML config."),
    input_csv: str = typer.Option(..., help="Input CSV with sample_id/sample_date/lat/lon."),
    output_csv: str = typer.Option(..., help="Output CSV path to write temporal img_t* features."),
    max_rows: int | None = typer.Option(None, help="Optional override for temporal_features.max_rows."),
    temporal_offsets_days: str | None = typer.Option(
        None,
        help="Optional comma-separated offsets override, e.g. '-3,0,3'.",
    ),
) -> None:
    cfg = load_config(config)
    tf_cfg = cfg.temporal_features
    if not tf_cfg:
        raise typer.BadParameter("Missing `temporal_features` section in config.")

    in_path = (Path.cwd() / input_csv).resolve()
    out_path = (Path.cwd() / output_csv).resolve()

    offsets = [int(v) for v in tf_cfg.get("temporal_offsets_days", [-7, -3, -1, 0, 1, 3, 7])]
    if temporal_offsets_days:
        offsets = [int(v.strip()) for v in str(temporal_offsets_days).split(",") if str(v).strip()]

    conf = S2TemporalConfig(
        stac_url=str(tf_cfg.get("stac_url", "https://planetarycomputer.microsoft.com/api/stac/v1")),
        collection=str(tf_cfg.get("collection", "sentinel-2-l2a")),
        temporal_offsets_days=offsets,
        offset_tolerance_days=int(tf_cfg.get("offset_tolerance_days", 1)),
        max_cloud_cover=float(tf_cfg.get("max_cloud_cover")) if tf_cfg.get("max_cloud_cover") is not None else None,
        max_rows=int(max_rows) if max_rows is not None else (int(tf_cfg.get("max_rows")) if tf_cfg.get("max_rows") is not None else None),
        bands=[str(v) for v in tf_cfg.get("bands", ["B02", "B03", "B04", "B08"])],
        timeout_seconds=int(tf_cfg.get("timeout_seconds", 90)),
        progress_every_rows=int(tf_cfg.get("progress_every_rows", 25)),
    )

    report = build_temporal_point_features(input_csv=in_path, output_csv=out_path, cfg=conf)
    report_dir = ensure_dir((Path.cwd() / str(cfg.artifacts_dir) / "s2_temporal_features").resolve())
    report_file = report_dir / "report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Built temporal Sentinel-2 features CSV: {out_path}")


@app.command("build-s2-image-chips")
def build_s2_image_chips_cmd(
    config: str = typer.Option(..., help="Path to YAML config."),
    input_csv: str = typer.Option(..., help="Input CSV with sample_id/sample_date/lat/lon."),
    chips_dir: str = typer.Option(..., help="Output directory for chip .npz files."),
    output_index_csv: str = typer.Option(..., help="Output CSV path for chip index/QA metadata."),
    max_rows: int | None = typer.Option(None, help="Optional override for image_chips.max_rows."),
) -> None:
    cfg = load_config(config)
    ic_cfg = cfg.image_chips
    if not ic_cfg:
        raise typer.BadParameter("Missing `image_chips` section in config.")

    in_path = (Path.cwd() / input_csv).resolve()
    chips_path = (Path.cwd() / chips_dir).resolve()
    out_idx = (Path.cwd() / output_index_csv).resolve()

    conf = S2ChipConfig(
        stac_url=str(ic_cfg.get("stac_url", "https://planetarycomputer.microsoft.com/api/stac/v1")),
        collection=str(ic_cfg.get("collection", "sentinel-2-l2a")),
        tolerance_days=int(ic_cfg.get("tolerance_days", 7)),
        limit=int(ic_cfg.get("limit", 50)),
        max_cloud_cover=float(ic_cfg.get("max_cloud_cover")) if ic_cfg.get("max_cloud_cover") is not None else None,
        max_solar_zenith=float(ic_cfg.get("max_solar_zenith")) if ic_cfg.get("max_solar_zenith") is not None else None,
        chip_size=int(ic_cfg.get("chip_size", 64)),
        overlap_stride_pixels=int(ic_cfg.get("overlap_stride_pixels", max(int(ic_cfg.get("chip_size", 64)) // 2, 1))),
        emit_chip_images=bool(ic_cfg.get("emit_chip_images", False)),
        chip_image_format=str(ic_cfg.get("chip_image_format", "png")),
        chip_image_quality=int(ic_cfg.get("chip_image_quality")) if ic_cfg.get("chip_image_quality") is not None else None,
        chip_image_rgb_bands=[str(v) for v in ic_cfg.get("chip_image_rgb_bands", ["B04", "B03", "B02"])],
        chip_image_stretch_percentiles=(
            float(ic_cfg.get("chip_image_stretch_p_low", 2.0)),
            float(ic_cfg.get("chip_image_stretch_p_high", 98.0)),
        ),
        bands=[str(v) for v in ic_cfg.get("bands", ["B02", "B03", "B04", "B08"])],
        max_rows=int(max_rows) if max_rows is not None else (int(ic_cfg.get("max_rows")) if ic_cfg.get("max_rows") is not None else None),
        max_nodata_ratio=float(ic_cfg.get("max_nodata_ratio", 0.5)),
        timeout_seconds=int(ic_cfg.get("timeout_seconds", 90)),
    )

    report = build_s2_image_chips(input_csv=in_path, chips_dir=chips_path, output_index_csv=out_idx, cfg=conf)
    report_dir = ensure_dir((Path.cwd() / str(cfg.artifacts_dir) / "s2_image_chips").resolve())
    report_file = report_dir / "report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Built Sentinel-2 chip index: {out_idx}")


@app.command("build-multimodal-index")
def build_multimodal_index_cmd(
    split_csv: str = typer.Option(..., help="Prepared split CSV path (e.g., train.csv)."),
    temporal_csv: str | None = typer.Option(None, help="Optional temporal CSV path for the same split."),
    chips_index_csv: str = typer.Option(..., help="Chip index CSV path from build-s2-image-chips."),
    output_csv: str = typer.Option(..., help="Output multimodal index CSV path."),
) -> None:
    split_path = (Path.cwd() / split_csv).resolve()
    temporal_path = (Path.cwd() / temporal_csv).resolve() if temporal_csv else None
    chips_idx_path = (Path.cwd() / chips_index_csv).resolve()
    out_path = (Path.cwd() / output_csv).resolve()

    report = build_multimodal_index(
        split_csv=split_path,
        temporal_csv=temporal_path,
        chips_index_csv=chips_idx_path,
        output_csv=out_path,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("fetch-vesla-odata")
def fetch_vesla_odata(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    vc = cfg.vesla_odata
    if not vc:
        raise typer.BadParameter("Missing `vesla_odata` section in config.")

    outputs = vc.get("outputs", {})
    in_situ_csv = (Path.cwd() / str(outputs.get("in_situ_csv", "data/raw/in_situ_vesla_odata.csv"))).resolve()
    long_csv = (Path.cwd() / str(outputs.get("long_csv", "data/external/vesla_odata_long.csv"))).resolve()
    site_csv = (Path.cwd() / str(outputs.get("site_csv", "data/external/vesla_odata_sites.csv"))).resolve()
    catalog_csv = (Path.cwd() / str(outputs.get("catalog_csv", "data/external/vesla_parameter_catalog.csv"))).resolve()
    determination_csv = (Path.cwd() / str(outputs.get("determination_csv", "data/external/vesla_determination_catalog.csv"))).resolve()
    report_json = (Path.cwd() / str(outputs.get("report_json", "artifacts/vesla_odata/report.json"))).resolve()

    conf = VeslaODataConfig(
        service_root=str(vc.get("service_root", "https://rajapinnat.ymparisto.fi/api/vesla/2.0/odata")),
        start_date=str(vc.get("start_date", "2020-01-01")),
        end_date=str(vc.get("end_date", "2025-12-31")),
        top=int(vc.get("top", 1000)),
        max_pages=int(vc.get("max_pages", 20)),
        apply_date_filter=bool(vc.get("apply_date_filter", False)),
    )

    report = fetch_vesla_odata_dataset(
        output_in_situ_csv=in_situ_csv,
        output_long_csv=long_csv,
        output_site_csv=site_csv,
        output_catalog_csv=catalog_csv,
        output_determination_csv=determination_csv,
        output_report_json=report_json,
        cfg=conf,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("build-complete-finland-s2")
def build_complete_finland_s2(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    root = Path.cwd()

    fi_cfg = cfg.finland_syke
    if not fi_cfg:
        raise typer.BadParameter("Missing `finland_syke` section in config.")
    fi_outputs = fi_cfg.get("outputs", {})
    in_situ_csv = (root / str(fi_outputs.get("in_situ_csv", "data/raw/in_situ_finland.csv"))).resolve()
    long_csv = (root / str(fi_outputs.get("long_csv", "data/external/finland_syke_long.csv"))).resolve()
    station_csv = (root / str(fi_outputs.get("station_csv", "data/external/finland_syke_stations.csv"))).resolve()

    fi_fetch_conf = FinlandFetchConfig(
        start_date=str(fi_cfg.get("start_date", "2023-01-01")),
        end_date=str(fi_cfg.get("end_date", "2025-01-01")),
        station_limit=int(fi_cfg.get("station_limit")) if fi_cfg.get("station_limit") is not None else None,
        parameters=list(fi_cfg.get("parameters", ["Chlorophyll a", "Turbidity", "Secchi depth", "Dissolved organic carbon"])),
        depth_layer=str(fi_cfg.get("depth_layer", "any")),
    )
    finland_report = fetch_finland_syke_dataset(
        output_in_situ_csv=in_situ_csv,
        output_raw_long_csv=long_csv,
        output_station_csv=station_csv,
        cfg=fi_fetch_conf,
    )

    s2_cfg = cfg.s2_metadata
    if not s2_cfg:
        raise typer.BadParameter("Missing `s2_metadata` section in config.")
    eo_out_csv = (root / str(s2_cfg.get("output_eo_csv", "data/raw/eo_features_finland_s2meta.csv"))).resolve()
    s2_conf = S2MetadataConfig(
        stac_url=str(s2_cfg.get("stac_url", "https://planetarycomputer.microsoft.com/api/stac/v1")),
        collection=str(s2_cfg.get("collection", "sentinel-2-l2a")),
        tolerance_days=int(s2_cfg.get("tolerance_days", 5)),
        limit=int(s2_cfg.get("limit", 10)),
        max_rows=int(s2_cfg.get("max_rows")) if s2_cfg.get("max_rows") is not None else None,
    )
    s2_report = build_s2_metadata_features(in_situ_csv=in_situ_csv, output_eo_csv=eo_out_csv, cfg=s2_conf)

    prep_cfg = copy.deepcopy(cfg)
    prep_cfg.paths["in_situ_csv"] = str(in_situ_csv.relative_to(root)).replace("\\", "/")
    prep_cfg.paths["eo_features_csv"] = str(eo_out_csv.relative_to(root)).replace("\\", "/")
    run_prepare(prep_cfg, root=root)

    prepared_dir = (root / str(cfg.artifacts_dir) / "prepared").resolve()
    train_df = pd.read_csv(prepared_dir / "train.csv")
    valid_df = pd.read_csv(prepared_dir / "valid.csv")
    test_df = pd.read_csv(prepared_dir / "test.csv")
    paired_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)

    release_dir = ensure_dir((root / str(cfg.artifacts_dir) / "release").resolve())
    paired_csv = release_dir / "finland_complete_paired_s2.csv"
    paired_df.to_csv(paired_csv, index=False)

    dataset_report = {
        "in_situ_csv": str(in_situ_csv),
        "eo_csv": str(eo_out_csv),
        "paired_csv": str(paired_csv),
        "n_in_situ_rows": int(finland_report.get("n_samples", 0)),
        "n_eo_rows": int(s2_report.get("n_rows_out", 0)),
        "n_paired_rows": int(len(paired_df)),
        "targets": list(cfg.targets),
        "target_non_null": {t: int(paired_df[t].notna().sum()) for t in cfg.targets if t in paired_df.columns},
    }
    report_path = release_dir / "finland_complete_paired_s2_report.json"
    report_path.write_text(json.dumps(dataset_report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(dataset_report, indent=2))


@app.command("package-release")
def package_release(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    root = Path.cwd()

    s2_cfg = cfg.s2_metadata or {}
    in_situ_csv = (root / str(s2_cfg.get("input_in_situ_csv", "data/raw/in_situ_finland.csv"))).resolve()
    eo_csv = (root / str(s2_cfg.get("output_eo_csv", "data/raw/eo_features_finland_s2meta.csv"))).resolve()

    in_situ_df = pd.read_csv(in_situ_csv)
    eo_df = pd.read_csv(eo_csv)
    in_situ_df["sample_date"] = pd.to_datetime(in_situ_df["sample_date"], errors="coerce").dt.date.astype(str)
    if "eo_id" in eo_df.columns:
        eo_df["sample_id"] = eo_df["eo_id"].astype(str).str.replace(r"^S2_", "", regex=True)
    eo_cols = [
        column
        for column in [
            "sample_id",
            "eo_id",
            "acquisition_date",
            "abs_day_diff",
            "s2_cloud_cover",
            "s2_solar_zenith",
            "s2_solar_azimuth",
            "s2_view_zenith",
            "s2_view_azimuth",
            "s2_mgrs_tile",
            "s2_item_id",
        ]
        if column in eo_df.columns
    ]
    eo_min = eo_df[eo_cols].drop_duplicates(subset=["sample_id"], keep="first") if "sample_id" in eo_cols else eo_df.copy()
    paired = in_situ_df.merge(eo_min, on="sample_id", how="left")

    release_dir = ensure_dir((root / str(cfg.artifacts_dir) / "release").resolve())
    paired_csv = release_dir / "finland_complete_paired_s2.csv"
    report_json = release_dir / "finland_complete_paired_s2_report.json"
    paired.to_csv(paired_csv, index=False)

    report = {
        "in_situ_csv": str(in_situ_csv),
        "eo_csv": str(eo_csv),
        "paired_csv": str(paired_csv),
        "n_in_situ_rows": int(len(in_situ_df)),
        "n_eo_rows": int(len(eo_df)),
        "n_paired_rows": int(len(paired)),
        "n_rows_with_s2_metadata": int(paired["acquisition_date"].notna().sum()) if "acquisition_date" in paired.columns else 0,
        "targets": list(cfg.targets),
        "target_non_null": {target: int(paired[target].notna().sum()) for target in cfg.targets if target in paired.columns},
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


@app.command("package-release-extended")
def package_release_extended(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)
    root = Path.cwd()

    s2_cfg = cfg.s2_metadata or {}
    fi_cfg = cfg.finland_syke or {}
    vesla_cfg = cfg.vesla_odata or {}

    in_situ_csv = (root / str(s2_cfg.get("input_in_situ_csv", "data/raw/in_situ_finland.csv"))).resolve()
    eo_csv = (root / str(s2_cfg.get("output_eo_csv", "data/raw/eo_features_finland_s2meta.csv"))).resolve()

    fi_outputs = fi_cfg.get("outputs", {})
    long_csv = (root / str(fi_outputs.get("long_csv", "data/external/finland_syke_long.csv"))).resolve()
    station_csv = (root / str(fi_outputs.get("station_csv", "data/external/finland_syke_stations.csv"))).resolve()

    vesla_outputs = vesla_cfg.get("outputs", {})
    vesla_site_csv = (root / str(vesla_outputs.get("site_csv", "data/external/vesla_odata_sites.csv"))).resolve()

    in_situ_df = pd.read_csv(in_situ_csv)
    eo_df = pd.read_csv(eo_csv)
    long_df = pd.read_csv(long_csv) if long_csv.exists() else pd.DataFrame()
    station_df = pd.read_csv(station_csv) if station_csv.exists() else pd.DataFrame()
    vesla_site_df = pd.read_csv(vesla_site_csv) if vesla_site_csv.exists() else pd.DataFrame()

    in_situ_df["sample_date"] = pd.to_datetime(in_situ_df["sample_date"], errors="coerce").dt.date.astype(str)

    if "eo_id" in eo_df.columns:
        eo_df["sample_id"] = eo_df["eo_id"].astype(str).str.replace(r"^S2_", "", regex=True)
    eo_cols = [
        column
        for column in [
            "sample_id",
            "eo_id",
            "acquisition_date",
            "abs_day_diff",
            "s2_cloud_cover",
            "s2_solar_zenith",
            "s2_solar_azimuth",
            "s2_view_zenith",
            "s2_view_azimuth",
            "s2_mgrs_tile",
            "s2_item_id",
        ]
        if column in eo_df.columns
    ]
    eo_min = eo_df[eo_cols].drop_duplicates(subset=["sample_id"], keep="first") if "sample_id" in eo_cols else eo_df.copy()
    paired = in_situ_df.merge(eo_min, on="sample_id", how="left")

    if not long_df.empty:
        long_use = long_df.copy()
        long_use["sample_date"] = pd.to_datetime(long_use["sample_date"], errors="coerce").dt.date.astype(str)
        long_use["ext_param"] = long_use.get("parameter", pd.Series([None] * len(long_use))).map(_normalize_extended_parameter)
        long_use["value"] = pd.to_numeric(long_use.get("value", pd.Series([None] * len(long_use))), errors="coerce")
        long_use = long_use.dropna(subset=["station_id", "sample_date", "ext_param", "value"])

        long_wide = (
            long_use.pivot_table(
                index=["station_id", "sample_date"],
                columns="ext_param",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        depth_col = None
        if "layer" in long_use.columns:
            depth_col = (
                long_use.groupby(["station_id", "sample_date"], dropna=False)["layer"]
                .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else pd.NA)
                .reset_index()
                .rename(columns={"layer": "depth_layer"})
            )

        if depth_col is not None:
            long_wide = long_wide.merge(depth_col, on=["station_id", "sample_date"], how="left")

        long_wide["station_id"] = long_wide["station_id"].astype(str)
        paired["station_id"] = paired["station_id"].astype(str)
        paired = paired.merge(long_wide, on=["station_id", "sample_date"], how="left", suffixes=("", "_ext"))

    if not station_df.empty and "station_id" in station_df.columns:
        station_keep = [column for column in ["station_id", "name", "lakename", "municipality"] if column in station_df.columns]
        station_min = station_df[station_keep].drop_duplicates(subset=["station_id"]).copy()
        station_min["station_id"] = station_min["station_id"].astype(str)
        paired["station_id"] = paired["station_id"].astype(str)
        paired = paired.merge(station_min, on="station_id", how="left")

    if not vesla_site_df.empty and "station_id" in vesla_site_df.columns:
        requested_meta = [
            "Municipal",
            "EnvironmentType",
            "WaterManagementAreaCode",
            "WaterManagementArea",
            "WaterbasinCode",
            "Waterbasin",
            "WaterbodyType",
            "WaterbodyType_Id",
        ]
        available_meta = [column for column in requested_meta if column in vesla_site_df.columns]
        if available_meta:
            vesla_min = vesla_site_df[["station_id", *available_meta]].drop_duplicates(subset=["station_id"]).copy()
            vesla_min["station_id"] = vesla_min["station_id"].astype(str)
            paired = paired.merge(vesla_min, on="station_id", how="left")

    release_dir = ensure_dir((root / str(cfg.artifacts_dir) / "release").resolve())
    paired_csv = release_dir / "finland_complete_extended_paired_s2.csv"
    report_json = release_dir / "finland_complete_extended_paired_s2_report.json"
    paired.to_csv(paired_csv, index=False)

    core_params = [
        "chl_a",
        "doc",
        "secchi_depth",
        "turbidity",
        "tp",
        "tn",
        "toc",
        "conductivity",
        "dissolved_oxygen",
        "ph",
        "temperature",
        "suspended_solids",
        "oxygen_saturation",
    ]
    report = {
        "in_situ_csv": str(in_situ_csv),
        "eo_csv": str(eo_csv),
        "long_csv": str(long_csv),
        "station_csv": str(station_csv),
        "vesla_site_csv": str(vesla_site_csv),
        "paired_csv": str(paired_csv),
        "n_in_situ_rows": int(len(in_situ_df)),
        "n_eo_rows": int(len(eo_df)),
        "n_paired_rows": int(len(paired)),
        "n_rows_with_s2_metadata": int(paired["acquisition_date"].notna().sum()) if "acquisition_date" in paired.columns else 0,
        "parameter_non_null": {name: int(paired[name].notna().sum()) for name in core_params if name in paired.columns},
        "metadata_columns_present": [
            column
            for column in [
                "municipality",
                "Municipal",
                "EnvironmentType",
                "WaterManagementAreaCode",
                "WaterManagementArea",
                "WaterbasinCode",
                "Waterbasin",
                "WaterbodyType",
                "WaterbodyType_Id",
                "depth_layer",
            ]
            if column in paired.columns
        ],
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


@app.command("extract-bbox")
def extract_bbox(
    input_csv: str = typer.Option(..., help="Input CSV path to filter."),
    min_lon: float = typer.Option(..., help="Minimum longitude."),
    min_lat: float = typer.Option(..., help="Minimum latitude."),
    max_lon: float = typer.Option(..., help="Maximum longitude."),
    max_lat: float = typer.Option(..., help="Maximum latitude."),
    output_csv: str = typer.Option("artifacts/release/finland_bbox_subset.csv", help="Output CSV path."),
) -> None:
    root = Path.cwd()
    in_path = (root / input_csv).resolve()
    out_path = (root / output_csv).resolve()

    df = pd.read_csv(in_path)
    if "lon" not in df.columns or "lat" not in df.columns:
        raise typer.BadParameter("Input CSV must contain `lon` and `lat` columns.")

    lon = pd.to_numeric(df["lon"], errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    mask = lon.between(min_lon, max_lon, inclusive="both") & lat.between(min_lat, max_lat, inclusive="both")
    subset = df.loc[mask].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)

    summary = {
        "input_csv": str(in_path),
        "output_csv": str(out_path),
        "bbox": {
            "min_lon": float(min_lon),
            "min_lat": float(min_lat),
            "max_lon": float(max_lon),
            "max_lat": float(max_lat),
        },
        "n_input": int(len(df)),
        "n_output": int(len(subset)),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("make-release-splits")
def make_release_splits(
    source_csv: str = typer.Option(
        "artifacts/release/finland_complete_extended_paired_s2.csv",
        help="Source extended release CSV.",
    ),
    full_dataset_csv: str = typer.Option(
        "artifacts/release/finland_full_dataset.csv",
        help="Output CSV for full dataset.",
    ),
    eo_paired_subset_csv: str = typer.Option(
        "artifacts/release/finland_eo_paired_subset.csv",
        help="Output CSV for rows with EO pairing.",
    ),
    manifest_json: str = typer.Option(
        "artifacts/release/finland_release_manifest.json",
        help="Output JSON manifest path.",
    ),
) -> None:
    root = Path.cwd()
    src = (root / source_csv).resolve()
    full_out = (root / full_dataset_csv).resolve()
    eo_out = (root / eo_paired_subset_csv).resolve()
    manifest_out = (root / manifest_json).resolve()

    params = [
        "chl_a",
        "doc",
        "secchi_depth",
        "turbidity",
        "tp",
        "tn",
        "toc",
        "conductivity",
        "dissolved_oxygen",
        "ph",
        "temperature",
        "suspended_solids",
        "oxygen_saturation",
    ]

    df = pd.read_csv(src, low_memory=False)
    full_out.parent.mkdir(parents=True, exist_ok=True)
    eo_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_out, index=False)

    if "acquisition_date" in df.columns:
        acq = df["acquisition_date"]
        eo = df.loc[acq.notna() & (acq.astype(str).str.strip() != "")].copy()
    else:
        eo = df.iloc[0:0].copy()
    eo.to_csv(eo_out, index=False)

    station_col = "station_id" if "station_id" in df.columns else None
    present = [col for col in params if col in df.columns]

    manifest = {
        "source_csv": str(src),
        "full_dataset_csv": str(full_out),
        "eo_paired_subset_csv": str(eo_out),
        "n_full_rows": int(len(df)),
        "n_eo_paired_rows": int(len(eo)),
        "n_unique_stations_full": int(df[station_col].nunique(dropna=True)) if station_col else None,
        "n_unique_stations_eo": int(eo[station_col].nunique(dropna=True)) if station_col else None,
        "parameter_non_null_full": {col: int(df[col].notna().sum()) for col in present},
        "parameter_non_null_eo": {col: int(eo[col].notna().sum()) for col in present},
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2))


@app.command("map-s2-tiles")
def map_s2_tiles(
    input_csv: str = typer.Option(
        "artifacts/release/finland_full_dataset.csv",
        help="Release CSV containing lat/lon and optional s2_mgrs_tile.",
    ),
    output_png: str = typer.Option(
        "artifacts/figures/finland/stations_on_s2_tile_bboxes.png",
        help="Output static PNG map path.",
    ),
    output_html: str = typer.Option(
        "artifacts/figures/finland/stations_on_s2_tile_bboxes.html",
        help="Output interactive HTML map path.",
    ),
    output_tile_counts_csv: str = typer.Option(
        "artifacts/release/stations_per_s2_tile.csv",
        help="Output CSV with station counts and tile bounding boxes.",
    ),
    output_report_json: str = typer.Option(
        "artifacts/release/s2_tile_mapping_report.json",
        help="Output JSON report path.",
    ),
    point_sample_size: int = typer.Option(
        2000,
        help="Number of points to draw on interactive map (sampled for responsiveness).",
    ),
    color_by: str = typer.Option(
        "chl_a",
        help="Column to use for color coding points (e.g., chl_a, turbidity, doc).",
    ),
    start_date: str | None = typer.Option(
        None,
        help="Optional start date filter on sample_date (YYYY-MM-DD).",
    ),
    end_date: str | None = typer.Option(
        None,
        help="Optional end date filter on sample_date (YYYY-MM-DD).",
    ),
    max_cloud_cover: float | None = typer.Option(
        None,
        help="Optional quality filter: keep rows with s2_cloud_cover <= value.",
    ),
    max_abs_day_diff: float | None = typer.Option(
        None,
        help="Optional quality filter: keep rows with abs_day_diff <= value.",
    ),
    output_dashboard_png: str = typer.Option(
        "artifacts/figures/finland/s2_tile_dashboard.png",
        help="Output PNG for summary dashboard charts.",
    ),
    output_municipality_csv: str = typer.Option(
        "artifacts/release/municipality_summary.csv",
        help="Output CSV with municipality-level summaries.",
    ),
    output_waterbasin_csv: str = typer.Option(
        "artifacts/release/waterbasin_summary.csv",
        help="Output CSV with water-basin-level summaries.",
    ),
    output_municipality_choropleth_png: str = typer.Option(
        "artifacts/figures/finland/municipality_choropleth_rows.png",
        help="Output PNG path for municipality choropleth (if boundaries are available).",
    ),
    strict_s2_bbox: bool = typer.Option(
        False,
        help="Use strict Sentinel-2 MGRS tile bounds only (no station-extent expansion).",
    ),
) -> None:
    try:
        import matplotlib.pyplot as plt
        import folium
        import mgrs
        import geopandas as gpd
        from shapely.geometry import box
        import contextily as cx
    except ImportError as exc:
        raise typer.BadParameter(
            "Missing optional dependency for map-s2-tiles. Install: mgrs, folium, matplotlib, geopandas, contextily"
        ) from exc

    root = Path.cwd()
    in_path = (root / input_csv).resolve()
    png_path = (root / output_png).resolve()
    html_path = (root / output_html).resolve()
    counts_path = (root / output_tile_counts_csv).resolve()
    report_path = (root / output_report_json).resolve()
    dashboard_path = (root / output_dashboard_png).resolve()
    municipality_path = (root / output_municipality_csv).resolve()
    waterbasin_path = (root / output_waterbasin_csv).resolve()
    municipality_choropleth_path = (root / output_municipality_choropleth_png).resolve()

    df = pd.read_csv(in_path, low_memory=False)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise typer.BadParameter("Input CSV must contain `lat` and `lon` columns.")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        raise typer.BadParameter("No valid coordinates after parsing `lat`/`lon`.")

    n_rows_pre_filter = int(len(df))
    if "sample_date" in df.columns and (start_date or end_date):
        sample_dates = pd.to_datetime(df["sample_date"], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if start_date:
            start_ts = pd.to_datetime(start_date, errors="coerce")
            if pd.isna(start_ts):
                raise typer.BadParameter("Invalid --start-date format. Use YYYY-MM-DD.")
            mask = mask & (sample_dates >= start_ts)
        if end_date:
            end_ts = pd.to_datetime(end_date, errors="coerce")
            if pd.isna(end_ts):
                raise typer.BadParameter("Invalid --end-date format. Use YYYY-MM-DD.")
            mask = mask & (sample_dates <= end_ts)
        df = df.loc[mask].copy()

    if max_cloud_cover is not None and "s2_cloud_cover" in df.columns:
        cloud = pd.to_numeric(df["s2_cloud_cover"], errors="coerce")
        df = df.loc[cloud.notna() & (cloud <= float(max_cloud_cover))].copy()

    if max_abs_day_diff is not None and "abs_day_diff" in df.columns:
        day_diff = pd.to_numeric(df["abs_day_diff"], errors="coerce")
        df = df.loc[day_diff.notna() & (day_diff <= float(max_abs_day_diff))].copy()

    if df.empty:
        raise typer.BadParameter("No rows remain after applying filters.")

    tile_col = "s2_mgrs_tile"
    converter = mgrs.MGRS()
    df["point_mgrs_tile"] = df.apply(
        lambda row: converter.toMGRS(float(row["lat"]), float(row["lon"]))[:5],
        axis=1,
    )
    if tile_col not in df.columns:
        df[tile_col] = df["point_mgrs_tile"]
    else:
        df[tile_col] = (
            df[tile_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
        )
        missing_mask = df[tile_col].isna()
        if missing_mask.any():
            df.loc[missing_mask, tile_col] = df.loc[missing_mask, "point_mgrs_tile"]

    tile_match_mask = (
        df["point_mgrs_tile"].notna()
        & df[tile_col].notna()
        & (df["point_mgrs_tile"] == df[tile_col])
    )
    tile_mismatch_mask = (
        df["point_mgrs_tile"].notna()
        & df[tile_col].notna()
        & (df["point_mgrs_tile"] != df[tile_col])
    )
    tile_pair_counts = (
        df.loc[tile_mismatch_mask, [tile_col, "point_mgrs_tile"]]
        .value_counts()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )

    bbox_tile_col = "point_mgrs_tile" if strict_s2_bbox else tile_col
    tile_series = df[bbox_tile_col].dropna().astype(str)
    unique_tiles = sorted(tile_series.unique().tolist())

    tile_rows: list[dict[str, Any]] = []
    for tile in unique_tiles:
        bbox = _bbox_from_mgrs_tile(tile)
        subset = df.loc[df[bbox_tile_col] == tile]
        min_station_lon = float(subset["lon"].min())
        min_station_lat = float(subset["lat"].min())
        max_station_lon = float(subset["lon"].max())
        max_station_lat = float(subset["lat"].max())
        row: dict[str, Any] = {
            "s2_mgrs_tile": tile,
            "station_count": int(len(subset)),
            "min_station_lon": min_station_lon,
            "min_station_lat": min_station_lat,
            "max_station_lon": max_station_lon,
            "max_station_lat": max_station_lat,
        }
        if bbox is not None:
            pad_lon = 0.0005
            pad_lat = 0.0005
            if strict_s2_bbox:
                row["tile_min_lon"] = float(bbox[0])
                row["tile_min_lat"] = float(bbox[1])
                row["tile_max_lon"] = float(bbox[2])
                row["tile_max_lat"] = float(bbox[3])
            else:
                row["tile_min_lon"] = min(float(bbox[0]), min_station_lon) - pad_lon
                row["tile_min_lat"] = min(float(bbox[1]), min_station_lat) - pad_lat
                row["tile_max_lon"] = max(float(bbox[2]), max_station_lon) + pad_lon
                row["tile_max_lat"] = max(float(bbox[3]), max_station_lat) + pad_lat
            row["bbox_source"] = "mgrs"
            row["bbox_adjusted_with_station_extent"] = not strict_s2_bbox
        else:
            pad_lon = 0.0005
            pad_lat = 0.0005
            row["tile_min_lon"] = min_station_lon - pad_lon
            row["tile_min_lat"] = min_station_lat - pad_lat
            row["tile_max_lon"] = max_station_lon + pad_lon
            row["tile_max_lat"] = max_station_lat + pad_lat
            row["bbox_source"] = "station_extent_fallback"
            row["bbox_adjusted_with_station_extent"] = True
        tile_rows.append(row)

    tile_df = pd.DataFrame(tile_rows).sort_values(["station_count", "s2_mgrs_tile"], ascending=[False, True])
    tile_df["station_share_pct"] = (tile_df["station_count"] / float(len(df)) * 100.0).round(3)

    merged_bbox = df.merge(
        tile_df[["s2_mgrs_tile", "tile_min_lon", "tile_min_lat", "tile_max_lon", "tile_max_lat"]],
        left_on=bbox_tile_col,
        right_on="s2_mgrs_tile",
        how="left",
    )
    within_bbox = (
        merged_bbox["lon"].between(merged_bbox["tile_min_lon"], merged_bbox["tile_max_lon"], inclusive="both")
        & merged_bbox["lat"].between(merged_bbox["tile_min_lat"], merged_bbox["tile_max_lat"], inclusive="both")
    )
    n_points_outside_bboxes = int((~within_bbox.fillna(False)).sum())

    if color_by not in df.columns:
        color_by = "chl_a" if "chl_a" in df.columns else tile_col

    color_series = pd.to_numeric(df[color_by], errors="coerce") if color_by in df.columns else pd.Series([pd.NA] * len(df))
    df["color_metric_numeric"] = color_series

    if color_by in df.columns:
        tile_color_stats = (
            df.groupby(bbox_tile_col, dropna=False)["color_metric_numeric"]
            .agg(["mean", "median", "min", "max", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{color_by}_mean",
                    "median": f"{color_by}_median",
                    "min": f"{color_by}_min",
                    "max": f"{color_by}_max",
                    "count": f"{color_by}_non_null_count",
                }
            )
        )
        tile_df = tile_df.merge(tile_color_stats, left_on="s2_mgrs_tile", right_on=bbox_tile_col, how="left")
        if bbox_tile_col != "s2_mgrs_tile" and bbox_tile_col in tile_df.columns:
            tile_df = tile_df.drop(columns=[bbox_tile_col], errors="ignore")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    municipality_path.parent.mkdir(parents=True, exist_ok=True)
    waterbasin_path.parent.mkdir(parents=True, exist_ok=True)
    municipality_choropleth_path.parent.mkdir(parents=True, exist_ok=True)

    stations_gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    tile_gdf = gpd.GeoDataFrame(
        tile_df.copy(),
        geometry=[
            box(
                float(row["tile_min_lon"]),
                float(row["tile_min_lat"]),
                float(row["tile_max_lon"]),
                float(row["tile_max_lat"]),
            )
            for row in tile_df.to_dict(orient="records")
        ],
        crs="EPSG:4326",
    )
    finland_boundary = _load_finland_boundary_gdf()

    fig, ax = plt.subplots(figsize=(10, 12))
    static_basemap_mode = "web_tiles"
    static_basemap_provider = "CartoDB.Positron"

    stations_3857 = stations_gdf.to_crs(epsg=3857)
    tiles_3857 = tile_gdf.to_crs(epsg=3857)

    if finland_boundary is not None:
        finland_3857 = finland_boundary.to_crs(epsg=3857)
        min_x, min_y, max_x, max_y = finland_3857.total_bounds
    else:
        min_x = float(min(tiles_3857.total_bounds[0], stations_3857.total_bounds[0]))
        min_y = float(min(tiles_3857.total_bounds[1], stations_3857.total_bounds[1]))
        max_x = float(max(tiles_3857.total_bounds[2], stations_3857.total_bounds[2]))
        max_y = float(max(tiles_3857.total_bounds[3], stations_3857.total_bounds[3]))

    x_pad = (max_x - min_x) * 0.08 if max_x > min_x else 20_000
    y_pad = (max_y - min_y) * 0.08 if max_y > min_y else 20_000
    ax.set_xlim(min_x - x_pad, max_x + x_pad)
    ax.set_ylim(min_y - y_pad, max_y + y_pad)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False, zoom=6)
    except Exception:
        static_basemap_mode = "finland_boundary_fallback"
        static_basemap_provider = "none"
        if finland_boundary is not None:
            finland_3857 = finland_boundary.to_crs(epsg=3857)
            finland_3857.plot(ax=ax, color="#f3f4f6", edgecolor="#4b5563", linewidth=1.0, zorder=1)

    tiles_3857.plot(ax=ax, facecolor="none", edgecolor="#2563eb", linewidth=1.0, alpha=0.9, zorder=2)
    tile_centroids = tiles_3857.copy()
    tile_centroids["geometry"] = tile_centroids.geometry.centroid
    for row in tile_centroids[["s2_mgrs_tile", "geometry"]].itertuples(index=False):
        ax.text(
            float(row.geometry.x),
            float(row.geometry.y),
            str(row.s2_mgrs_tile),
            fontsize=7,
            ha="center",
            va="center",
            alpha=0.7,
            zorder=3,
        )

    if df["color_metric_numeric"].notna().any():
        stations_plot = stations_3857.assign(color_metric_numeric=df["color_metric_numeric"].values)
        stations_plot.plot(
            ax=ax,
            column="color_metric_numeric",
            cmap="viridis",
            markersize=7,
            alpha=0.55,
            zorder=4,
            legend=True,
            legend_kwds={"label": f"{color_by} (numeric)", "shrink": 0.7},
        )
    else:
        stations_3857.plot(ax=ax, markersize=4, color="#dc2626", alpha=0.35, zorder=4, label="Water-quality samples")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Finland water-quality over Sentinel-2 tiles (color: {color_by})")
    ax.grid(alpha=0.2)
    if not df["color_metric_numeric"].notna().any():
        ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    choropleth_data = tile_df[["s2_mgrs_tile", "station_count"]].copy()
    tile_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "s2_mgrs_tile": str(row["s2_mgrs_tile"]),
                    "station_count": int(row["station_count"]),
                    f"{color_by}_mean": (
                        None
                        if pd.isna(row.get(f"{color_by}_mean", pd.NA))
                        else float(row.get(f"{color_by}_mean", 0.0))
                    ),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [float(row["tile_min_lon"]), float(row["tile_min_lat"])],
                            [float(row["tile_max_lon"]), float(row["tile_min_lat"])],
                            [float(row["tile_max_lon"]), float(row["tile_max_lat"])],
                            [float(row["tile_min_lon"]), float(row["tile_max_lat"])],
                            [float(row["tile_min_lon"]), float(row["tile_min_lat"])],
                        ]
                    ],
                },
            }
            for row in tile_df.to_dict(orient="records")
        ],
    }

    folium.Choropleth(
        geo_data=tile_geojson,
        data=choropleth_data,
        columns=["s2_mgrs_tile", "station_count"],
        key_on="feature.properties.s2_mgrs_tile",
        fill_color="YlGnBu",
        fill_opacity=0.4,
        line_opacity=0.25,
        line_color="#1e3a8a",
        legend_name="Rows per Sentinel-2 tile",
        name="Tile density choropleth",
    ).add_to(fmap)

    for row in tile_df.to_dict(orient="records"):
        folium.Rectangle(
            bounds=[
                [float(row["tile_min_lat"]), float(row["tile_min_lon"])],
                [float(row["tile_max_lat"]), float(row["tile_max_lon"])],
            ],
            color="#2563eb",
            fill=True,
            fill_opacity=0.05,
            weight=1,
            tooltip=f"{row['s2_mgrs_tile']} ({row['station_count']} rows)",
        ).add_to(fmap)

    if point_sample_size > 0 and len(df) > point_sample_size:
        points_df = df.sample(n=point_sample_size, random_state=42)
    else:
        points_df = df

    point_cols = ["lat", "lon", bbox_tile_col]
    station_col = "station_id" if "station_id" in points_df.columns else None
    if station_col is not None:
        point_cols.append(station_col)

    for row in points_df[point_cols + ["color_metric_numeric"]].itertuples(index=False):
        station_value = getattr(row, station_col) if station_col is not None else "n/a"
        color_metric = getattr(row, "color_metric_numeric")
        color_metric_text = "n/a" if pd.isna(color_metric) else f"{float(color_metric):.3f}"
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=2,
            color="#dc2626",
            fill=True,
            fill_opacity=0.7,
            weight=0,
            tooltip=f"station={station_value}, tile={getattr(row, bbox_tile_col)}, {color_by}={color_metric_text}",
        ).add_to(fmap)

    municipality_col = "municipality" if "municipality" in df.columns else None
    if municipality_col is not None:
        municipality_summary = (
            df.groupby(municipality_col, dropna=False)
            .agg(
                n_rows=(municipality_col, "size"),
                n_stations=("station_id", lambda s: s.nunique(dropna=True)) if "station_id" in df.columns else (municipality_col, "size"),
                n_tiles=(bbox_tile_col, lambda s: s.nunique(dropna=True)),
                color_by_mean=("color_metric_numeric", "mean"),
            )
            .reset_index()
            .rename(columns={municipality_col: "municipality", "color_by_mean": f"{color_by}_mean"})
            .sort_values("n_rows", ascending=False)
        )
        municipality_summary.to_csv(municipality_path, index=False)
    else:
        municipality_summary = pd.DataFrame(columns=["municipality", "n_rows", "n_stations", "n_tiles", f"{color_by}_mean"])
        municipality_summary.to_csv(municipality_path, index=False)

    waterbasin_col = "Waterbasin" if "Waterbasin" in df.columns else ("WaterManagementArea" if "WaterManagementArea" in df.columns else None)
    if waterbasin_col is not None:
        waterbasin_summary = (
            df.groupby(waterbasin_col, dropna=False)
            .agg(
                n_rows=(waterbasin_col, "size"),
                n_stations=("station_id", lambda s: s.nunique(dropna=True)) if "station_id" in df.columns else (waterbasin_col, "size"),
                n_tiles=(bbox_tile_col, lambda s: s.nunique(dropna=True)),
                color_by_mean=("color_metric_numeric", "mean"),
            )
            .reset_index()
            .rename(columns={waterbasin_col: "waterbasin", "color_by_mean": f"{color_by}_mean"})
            .sort_values("n_rows", ascending=False)
        )
        waterbasin_summary.to_csv(waterbasin_path, index=False)
    else:
        waterbasin_summary = pd.DataFrame(columns=["waterbasin", "n_rows", "n_stations", "n_tiles", f"{color_by}_mean"])
        waterbasin_summary.to_csv(waterbasin_path, index=False)

    municipality_choropleth_generated = False
    if not municipality_summary.empty:
        try:
            admin2_url = "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/FIN/ADM2/geoBoundaries-FIN-ADM2.geojson"
            muni_boundaries = gpd.read_file(admin2_url)
            name_col_candidates = ["shapeName", "name", "NAME_2", "ADM2_EN", "admin2Name"]
            name_col = next((c for c in name_col_candidates if c in muni_boundaries.columns), None)

            if name_col is not None:
                muni_boundaries = muni_boundaries.to_crs(epsg=3857)
                muni_boundaries["municipality_key"] = muni_boundaries[name_col].map(_normalize_name_key)
                muni_summary = municipality_summary.copy()
                muni_summary["municipality_key"] = muni_summary["municipality"].map(_normalize_name_key)
                merged = muni_boundaries.merge(muni_summary, on="municipality_key", how="left")

                fig_muni, ax_muni = plt.subplots(figsize=(10, 12))
                merged.plot(
                    ax=ax_muni,
                    column="n_rows",
                    cmap="YlOrRd",
                    linewidth=0.2,
                    edgecolor="#6b7280",
                    legend=True,
                    legend_kwds={"label": "Rows per municipality", "shrink": 0.7},
                )
                ax_muni.set_axis_off()
                ax_muni.set_title("Finland municipality choropleth (water-quality row count)")
                plt.tight_layout()
                fig_muni.savefig(municipality_choropleth_path, dpi=220)
                plt.close(fig_muni)
                municipality_choropleth_generated = True
        except Exception:
            municipality_choropleth_generated = False

    fig_dash, axes = plt.subplots(2, 2, figsize=(14, 10))
    top_tiles = tile_df.sort_values("station_count", ascending=False).head(12)
    axes[0, 0].barh(top_tiles["s2_mgrs_tile"].astype(str), top_tiles["station_count"], color="#2563eb")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Top 12 tiles by row count")
    axes[0, 0].set_xlabel("Rows")

    key_cols = [c for c in ["chl_a", "doc", "turbidity", "secchi_depth", "tp", "tn", "toc"] if c in df.columns]
    if key_cols:
        missing_pct = (df[key_cols].isna().mean() * 100.0).sort_values(ascending=False)
        axes[0, 1].bar(missing_pct.index, missing_pct.values, color="#ef4444")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].set_ylabel("Missing %")
        axes[0, 1].set_title("Parameter missingness")
    else:
        axes[0, 1].text(0.5, 0.5, "No target columns available", ha="center", va="center")
        axes[0, 1].set_axis_off()

    if "sample_date" in df.columns:
        sample_dates = pd.to_datetime(df["sample_date"], errors="coerce")
        monthly = sample_dates.dropna().dt.to_period("M").astype(str).value_counts().sort_index()
        if len(monthly) > 0:
            axes[1, 0].plot(monthly.index, monthly.values, color="#059669")
            axes[1, 0].tick_params(axis="x", rotation=60)
            axes[1, 0].set_title("Monthly sample volume")
            axes[1, 0].set_ylabel("Rows")
        else:
            axes[1, 0].text(0.5, 0.5, "No valid sample_date values", ha="center", va="center")
            axes[1, 0].set_axis_off()
    else:
        axes[1, 0].text(0.5, 0.5, "sample_date column not available", ha="center", va="center")
        axes[1, 0].set_axis_off()

    top_munis = municipality_summary.head(12)
    if not top_munis.empty:
        axes[1, 1].barh(top_munis["municipality"].astype(str), top_munis["n_rows"], color="#7c3aed")
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title("Top municipalities by row count")
        axes[1, 1].set_xlabel("Rows")
    else:
        axes[1, 1].text(0.5, 0.5, "Municipality column not available", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig_dash.tight_layout()
    fig_dash.savefig(dashboard_path, dpi=220)
    plt.close(fig_dash)

    fmap.save(str(html_path))
    tile_df.to_csv(counts_path, index=False)

    report = {
        "input_csv": str(in_path),
        "n_rows_before_filters": int(n_rows_pre_filter),
        "n_rows_input": int(len(df)),
        "n_unique_tiles": int(len(tile_df)),
        "tile_column": tile_col,
        "bbox_tile_column": bbox_tile_col,
        "strict_s2_bbox": bool(strict_s2_bbox),
        "color_by": color_by,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "max_cloud_cover": max_cloud_cover,
            "max_abs_day_diff": max_abs_day_diff,
        },
        "n_drawn_points_interactive": int(len(points_df)),
        "n_point_vs_s2_tile_matches": int(tile_match_mask.sum()),
        "n_point_vs_s2_tile_mismatches": int(tile_mismatch_mask.sum()),
        "point_vs_s2_tile_mismatch_top_pairs": tile_pair_counts.head(10).to_dict(orient="records"),
        "n_stations_with_point_vs_s2_tile_mismatch": (
            int(df.loc[tile_mismatch_mask, "station_id"].nunique(dropna=True)) if "station_id" in df.columns else None
        ),
        "outputs": {
            "png": str(png_path),
            "html": str(html_path),
            "tile_counts_csv": str(counts_path),
            "dashboard_png": str(dashboard_path),
            "municipality_summary_csv": str(municipality_path),
            "waterbasin_summary_csv": str(waterbasin_path),
            "municipality_choropleth_png": str(municipality_choropleth_path),
        },
        "bbox_sources": tile_df["bbox_source"].value_counts(dropna=False).to_dict(),
        "n_tiles_bbox_adjusted": int(tile_df["bbox_adjusted_with_station_extent"].sum()) if "bbox_adjusted_with_station_extent" in tile_df.columns else 0,
        "n_points_outside_bboxes": n_points_outside_bboxes,
        "has_finland_boundary_basemap": bool(finland_boundary is not None),
        "static_basemap_mode": static_basemap_mode,
        "static_basemap_provider": static_basemap_provider,
        "municipality_choropleth_generated": municipality_choropleth_generated,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
