"""
Step 2 — Baseline Model Training

Trains all baseline model families across all four tolerance windows:
  - Tabular Random Forest (RF)
  - Tabular family (RF, Ridge, Lasso, MLP, HistGBT, TabularTransformer)
  - Temporal sequence models
  - Image-chip CNN/ViT models
  - HuggingFace pretrained vision encoders
  - Fusion (tabular + temporal) models

Outputs a combined baseline metric summary at:
  artifacts/prepared/baseline_metric_summary.json
  artifacts/prepared/baseline_metric_summary.csv

Usage
-----
    # Run all baseline families
    python scripts/02_train_baselines.py --tolerances 1,3,5,7 --config configs/default.yaml

    # Tabular only
    python scripts/02_train_baselines.py --tolerances 1,3,5,7 --families tabular

    # Multiple selected families
    python scripts/02_train_baselines.py --tolerances 1,3,5,7 --families tabular,temporal,fusion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FAMILIES = ["tabular", "tabular_family", "temporal", "image", "hf", "fusion"]


def _parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return None
    return float(numeric.mean())


def build_summary(root: Path) -> None:
    """Aggregate all baseline validation CSVs into a single summary."""
    outputs: list[dict] = []

    source_files = {
        "tabular_rf":     root / "artifacts/prepared/tabular_rf_tolerance_validation.csv",
        "tabular_family": root / "artifacts/prepared/tabular_family_tolerance_validation.csv",
        "temporal":       root / "artifacts/prepared/temporal_family_tolerance_validation.csv",
        "image":          root / "artifacts/prepared/image_family_tolerance_validation.csv",
        "hf":             root / "artifacts/prepared/hf_pretrained_tolerance_validation.csv",
        "fusion":         root / "artifacts/prepared/fusion_tolerance_validation.csv",
    }

    for family, path in source_files.items():
        if not path.exists():
            print(f"[baseline-summary] SKIP {family}: {path} not found")
            continue
        df = pd.read_csv(path, low_memory=False)
        for _, row in df.iterrows():
            outputs.append({
                "family":            family,
                "model":             row.get("model"),
                "tolerance_days":    int(row.get("tolerance_days")),
                "macro_rmse":        row.get("mean_rmse"),
                "macro_mae":         row.get("mean_mae"),
                "macro_r2":          row.get("mean_r2"),
                "inference_ms_per_sample": row.get("inference_ms_per_sample"),
            })

    out_dir = root / "artifacts" / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "baseline_metric_summary.json"
    csv_path  = out_dir / "baseline_metric_summary.csv"

    json_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    pd.DataFrame(outputs).to_csv(csv_path, index=False)
    print(f"[baseline-summary] Saved {len(outputs)} rows → {json_path}")
    print(f"[baseline-summary] Saved {len(outputs)} rows → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Train all baseline model families."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tolerances", default="1,3,5,7")
    parser.add_argument("--families", default=",".join(FAMILIES),
                        help=f"Comma-separated families to run. Options: {FAMILIES}")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip training; just rebuild the summary from existing CSVs.")
    args = parser.parse_args()

    import subprocess, sys
    root = Path(__file__).resolve().parent.parent
    py   = sys.executable
    tols = args.tolerances
    families = _parse_list(args.families)

    def run(script: str, extra: list[str] | None = None) -> None:
        cmd = [py, str(root / "scripts" / script),
               "--tolerances", tols, "--config", args.config]
        if extra:
            cmd.extend(extra)
        print(f"\n[baselines] Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(root), check=True)

    if not args.summary_only:
        if "tabular" in families:
            run("run_tabular_rf_tolerance_sweep.py")
        if "tabular_family" in families:
            run("run_tabular_family_tolerance_sweep.py")
        if "temporal" in families:
            run("run_temporal_tolerance_sweep.py")
            run("run_temporal_family_tolerance_sweep.py")
        if "image" in families:
            run("run_image_chip_tolerance_sweep.py")
            run("run_image_family_tolerance_sweep.py")
        if "hf" in families:
            run("run_hf_pretrained_tolerance_sweep.py")
        if "fusion" in families:
            run("run_fusion_tolerance_sweep.py")

    build_summary(root)
    print("\n[baselines] All baseline training complete.")


if __name__ == "__main__":
    main()
