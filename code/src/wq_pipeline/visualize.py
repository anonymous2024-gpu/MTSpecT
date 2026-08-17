from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _plot_map(df: pd.DataFrame, out_path: Path, title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["lon"], df["lat"], s=9, alpha=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_parameter_histograms(df: pd.DataFrame, out_path: Path, params: list[str]) -> None:
    if not params:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(0.5, 0.5, "No parameter columns available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, parameter in zip(axes, params):
        series = pd.to_numeric(df.get(parameter), errors="coerce").dropna()
        if series.empty:
            ax.set_title(f"{parameter} (no data)")
            continue
        ax.hist(series, bins=35, alpha=0.75)
        ax.set_title(parameter)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Count")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _slugify_filename(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "parameter"


def _plot_finland_timeseries(long_df: pd.DataFrame, out_path: Path, parameter: str = "Chlorophyll a") -> bool:
    param_series = long_df["parameter"].astype(str)
    requested = str(parameter).strip().lower()
    use = long_df[param_series.str.strip().str.lower() == requested].copy()
    use["sample_date"] = pd.to_datetime(use["sample_date"], errors="coerce")
    use = use.dropna(subset=["sample_date", "value"])
    if use.empty:
        return False

    station_counts = use.groupby("station_id").size().sort_values(ascending=False)
    top_stations = station_counts.head(6).index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    for station in top_stations:
        station_df = use[use["station_id"] == station].sort_values("sample_date")
        ax.plot(station_df["sample_date"], station_df["value"], marker="o", linewidth=1.0, markersize=2.5, label=str(station))

    ax.set_title(f"Finland stations: {parameter} time series (top stations by samples)")
    ax.set_xlabel("Date")
    ax.set_ylabel(parameter)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def create_visualizations(
    in_situ_csv: Path,
    long_csv: Path | None,
    output_dir: Path,
    timeseries_parameter: str = "Chlorophyll a",
) -> dict:
    df = pd.read_csv(in_situ_csv, low_memory=False)
    for col in ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    eu_map = output_dir / "map_eu_stations.png"
    fi_map = output_dir / "map_finland_stations.png"
    hist = output_dir / "parameter_histograms.png"

    _plot_map(df, eu_map, "EU/Regional station locations", xlim=(-15, 45), ylim=(34, 72))
    _plot_map(df, fi_map, "Finland station locations", xlim=(19, 33), ylim=(59, 71))

    params = [p for p in ["chl_a", "turbidity", "secchi_depth", "doc", "toc", "tp", "tn"] if p in df.columns]
    _plot_parameter_histograms(df, hist, params=params)

    ts_slug = _slugify_filename(timeseries_parameter)
    if ts_slug == "chlorophyll_a":
        ts_output = output_dir / "finland_chla_timeseries.png"
    else:
        ts_output = output_dir / f"finland_{ts_slug}_timeseries.png"
    ts_path = None
    if long_csv and long_csv.exists():
        long_df = pd.read_csv(long_csv, low_memory=False)
        if {"parameter", "sample_date", "value", "station_id"}.issubset(set(long_df.columns)):
            generated = _plot_finland_timeseries(long_df, ts_output, parameter=timeseries_parameter)
            if generated and ts_output.exists() and ts_output.stat().st_size > 0:
                ts_path = ts_output
            elif ts_output.exists():
                ts_output.unlink()
    elif ts_output.exists():
        ts_output.unlink()

    return {
        "n_points": int(len(df)),
        "eu_map": str(eu_map),
        "finland_map": str(fi_map),
        "histogram": str(hist),
        "timeseries_parameter": str(timeseries_parameter),
        "timeseries": str(ts_path) if ts_path else None,
    }
