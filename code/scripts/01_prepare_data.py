"""
Step 1 — Data Preparation

Downloads SYKE/VESLA in-situ measurements, pairs them with Sentinel-2 imagery
within a date-tolerance window, applies QC, extracts image chips and temporal
features, and writes per-tolerance train/valid/test splits.

Usage
-----
    # Run all tolerances end-to-end
    python scripts/01_prepare_data.py --tolerances 1,3,5,7 --config configs/default.yaml

    # Single tolerance
    python scripts/01_prepare_data.py --tolerances 7 --config configs/default.yaml

    # Skip data fetch if raw CSVs already exist
    python scripts/01_prepare_data.py --tolerances 1,3,5,7 --skip-fetch

Outputs
-------
    artifacts/tolerance_{N}d/prepared/train.csv
    artifacts/tolerance_{N}d/prepared/valid.csv
    artifacts/tolerance_{N}d/prepared/test.csv
    artifacts/tolerance_{N}d/prepared/multimodal_train.csv
    artifacts/tolerance_{N}d/prepared/multimodal_valid.csv
    artifacts/tolerance_{N}d/prepared/multimodal_test.csv
    artifacts/tolerance_{N}d/prepared/temporal/
    artifacts/tolerance_{N}d/prepared/chips/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n[prepare] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), check=True)
    if result.returncode != 0:
        print(f"[prepare] ERROR: command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Prepare all data splits for MT-SpecT++ experiments."
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to pipeline config YAML.")
    parser.add_argument("--tolerances", default="1,3,5,7",
                        help="Comma-separated tolerance days.")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip SYKE/VESLA download (use existing raw CSVs).")
    parser.add_argument("--skip-s2", action="store_true",
                        help="Skip Sentinel-2 metadata build (use existing eo_features.csv).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    py = sys.executable

    # ---- Step 1a: Fetch in-situ data from SYKE/VESLA -------------------------
    if not args.skip_fetch:
        _run([py, "-m", "wq_pipeline.cli", "fetch-finland-syke",
              "--config", args.config], cwd=root)

    # ---- Step 1b: Build Sentinel-2 metadata ----------------------------------
    if not args.skip_s2:
        _run([py, "-m", "wq_pipeline.cli", "build-s2-metadata",
              "--config", args.config], cwd=root)
        _run([py, "-m", "wq_pipeline.cli", "build-complete-finland-s2",
              "--config", args.config], cwd=root)

    # ---- Step 1c: Prepare splits for each tolerance --------------------------
    for tol in [t.strip() for t in args.tolerances.split(",") if t.strip()]:
        print(f"\n[prepare] === Tolerance {tol}d ===")
        _run([py, "-m", "wq_pipeline.cli", "prepare",
              "--config", args.config,
              "--tolerance", tol], cwd=root)

    print("\n[prepare] Data preparation complete.")
    print("  Outputs: artifacts/tolerance_Nd/prepared/")


if __name__ == "__main__":
    main()
