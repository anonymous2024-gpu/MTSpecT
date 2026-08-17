"""
Step 4 — Select Best MT-SpecT++ Model per Tolerance

Parses the batch-sweep summary JSONs produced by Step 3 and selects the single
best checkpoint per tolerance using a weighted composite score over 9 metrics
(RMSE, MAE, R², MAPE, NLL, CRPS, |PICP−0.95|, MPIW, ECE).

Usage
-----
    python scripts/04_select_best_model.py \
        --tolerances 1,3,5,7 \
        --output-json artifacts/prepared/mtspect_best_per_tolerance.json \
        --output-csv  artifacts/prepared/mtspect_best_per_tolerance.csv

Outputs
-------
    artifacts/prepared/mtspect_best_per_tolerance.json
    artifacts/prepared/mtspect_best_per_tolerance.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_tolerances(raw: str) -> List[int]:
    vals = [int(p.strip()) for p in str(raw).split(",") if p.strip()]
    if not vals:
        raise ValueError("No tolerances provided.")
    return vals


def _parse_metric_string(value) -> float:
    """Parse '14.19 ± 0.76' → 14.19, or pass through a numeric value."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.match(r"^([+-]?\d*\.?\d+)", str(value).strip())
    if match:
        return float(match.group(1))
    raise ValueError(f"Cannot parse metric: {value!r}")


def _load_sweep_summary(root: Path, tolerance: int) -> List[Dict[str, Any]]:
    path = (root / "artifacts" / f"tolerance_{tolerance}d"
            / "results" / "mtspect" / "summary_batch_sweep.json")
    if not path.exists():
        print(f"[select] WARNING: No sweep summary for τ={tolerance}d → {path}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    except Exception as exc:
        print(f"[select] ERROR reading {path}: {exc}")
        return []


def _extract_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in ["rmse", "mae", "r2", "mape", "nll", "crps",
                "picp", "mpiw", "ece_regression"]:
        if key in record:
            try:
                metrics[key] = _parse_metric_string(record[key])
            except ValueError:
                pass
    return metrics


def _weighted_score(metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Lower score = better.
    Each metric is mapped to [0,1] (higher-is-better), then averaged and inverted.
    """
    required = ["rmse", "mae", "r2", "mape", "nll", "crps", "picp", "mpiw", "ece_regression"]
    missing = [k for k in required if k not in metrics]
    if missing:
        raise ValueError(f"Missing metrics: {missing}")

    norms = {
        "rmse": 1 / (1 + metrics["rmse"]),
        "mae":  1 / (1 + metrics["mae"]),
        "r2":   abs(metrics["r2"]),
        "mape": 1 / (1 + metrics["mape"]),
        "nll":  1 / (1 + metrics["nll"]),
        "crps": 1 / (1 + metrics["crps"]),
        "picp": 1 / (1 + abs(metrics["picp"] - 0.95)),
        "mpiw": 1 / (1 + metrics["mpiw"]),
        "ece":  1 / (1 + metrics["ece_regression"]),
    }
    composite = sum(norms.values()) / len(norms)
    final_score = 1 - composite          # lower = better
    return final_score, {**norms, "composite_score": final_score}


def _select_best(records: List[Dict[str, Any]], tolerance: int) -> Optional[Dict[str, Any]]:
    scored = []
    for rec in records:
        try:
            metrics = _extract_metrics(rec)
            score, norms = _weighted_score(metrics)
            scored.append({**rec, "weighted_score": score,
                           "normalized_metrics": norms, "raw_metrics": metrics})
        except Exception as exc:
            print(f"[select] tol={tolerance} skip record: {exc}")

    if not scored:
        return None

    scored.sort(key=lambda x: x["weighted_score"])
    best = scored[0]
    print(f"[select] τ={tolerance}d best bs={best.get('batch_size')} "
          f"score={best['weighted_score']:.4f} "
          f"rmse={best['raw_metrics'].get('rmse', '?')} "
          f"r2={best['raw_metrics'].get('r2', '?')}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Select best MT-SpecT++ checkpoint per tolerance."
    )
    parser.add_argument("--root",         default=".")
    parser.add_argument("--tolerances",   default="1,3,5,7")
    parser.add_argument("--output-json",  required=True)
    parser.add_argument("--output-csv",   required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tolerances = _parse_tolerances(args.tolerances)

    selections = []
    for tol in tolerances:
        records = _load_sweep_summary(root, tol)
        best = _select_best(records, tol)
        if best:
            selections.append({
                "tolerance_days":  tol,
                "selected_model": {
                    "batch_size":      best.get("batch_size"),
                    "weighted_score":  best["weighted_score"],
                    "checkpoint_path": best.get("checkpoint_path"),
                    "metrics_csv":     best.get("metrics_csv"),
                },
                "raw_metrics":        best["raw_metrics"],
                "normalized_metrics": best["normalized_metrics"],
                "full_record":        best,
            })
        else:
            print(f"[select] WARNING: no valid models found for τ={tol}d")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(selections, indent=2, default=str), encoding="utf-8")
    print(f"[select] Saved → {out_json}")

    if selections:
        rows = [{
            "tolerance_days":  s["tolerance_days"],
            "batch_size":      s["selected_model"]["batch_size"],
            "weighted_score":  s["selected_model"]["weighted_score"],
            "checkpoint_path": s["selected_model"]["checkpoint_path"],
            **s["raw_metrics"],
        } for s in selections]
        out_csv = Path(args.output_csv)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[select] Saved → {out_csv}")


if __name__ == "__main__":
    main()
