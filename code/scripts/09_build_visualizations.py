"""
Step 9 — Build Visualizations

Generates all figures used in the paper:

  Section A — Coverage maps (pre/post QC, all stations)
  Section B — Temporal availability and quality-rejection maps (per tolerance)
  Section C — Performance maps (combined 2×2 + per-tolerance overlays)
  Section D — Uncertainty maps (combined 2×2 + per-tolerance overlays)
  Section E — Model-delta maps (MT-SpecT++ vs best baseline, per tolerance)
  Section F — Metric-curve panels (RMSE / MAE / R² / runtime vs tolerance)
  Section G — Calibration / reliability diagrams

Requires Step 8 to have been run (test_predictions_with_coords.csv needed for
performance, uncertainty, and model-delta maps).

Usage
-----
    python scripts/09_build_visualizations.py \
        --tolerances 1 3 5 7

    # Calibration diagrams only
    python scripts/09_build_visualizations.py --calibration-only

Outputs
-------
    artifacts/figures/coverage/       — coverage maps
    artifacts/figures/maps/           — performance / uncertainty / delta maps
    artifacts/figures/metric_curves/  — RMSE / MAE / R² / runtime curves
    artifacts/figures/calibration/    — reliability diagrams
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n[viz] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 9: Generate all paper figures."
    )
    parser.add_argument("--root",              default=".")
    parser.add_argument("--tolerances",        nargs="+", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--calibration-only",  action="store_true")
    parser.add_argument("--maps-only",         action="store_true")
    parser.add_argument("--predictions-csv",   default=None,
                        help="Override path to predictions CSV.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    py   = sys.executable
    tols = [str(t) for t in args.tolerances]

    if not args.calibration_only:
        # Coverage + performance + uncertainty + delta + metric-curve panels
        cmd = [py, str(root / "scripts" / "build_phase_i_visualizations.py"),
               "--root", str(root),
               "--tolerances"] + tols
        if args.predictions_csv:
            cmd += ["--predictions-csv", args.predictions_csv]
        _run(cmd, cwd=root)

    if not args.maps_only:
        # Reliability diagrams
        out_dir = str(root / "artifacts" / "figures" / "calibration")
        _run([py, str(root / "scripts" / "plot_reliability_diagrams.py"),
              "--out-dir", out_dir], cwd=root)

    print("\n[viz] All figures generated.")
    print("  Coverage maps   → artifacts/figures/phase_i/")
    print("  Calibration     → artifacts/figures/calibration/")
    print("  Metric curves   → artifacts/visualization/phase_j/")


if __name__ == "__main__":
    main()
