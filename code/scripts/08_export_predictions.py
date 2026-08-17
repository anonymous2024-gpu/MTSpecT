"""
Step 8 — Export Test Predictions with Coordinates

For each tolerance, loads the best MT-SpecT++ checkpoint (from Step 4),
runs inference on the held-out test split, and saves a long-form CSV with
per-sample geographic coordinates attached.

Usage
-----
    python scripts/08_export_predictions.py \
        --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
        --allow-missing-modalities

Outputs
-------
    artifacts/tolerance_{N}d/results/mtspect/test_predictions_with_coords.csv

    Columns: lat, lon, station_id, lake_id, municipality,
             target, y_true, y_pred, y_std
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wq_pipeline.mtspect.trainer import TrainConfig, predict_mtspect


def _load_selections(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(sel["tolerance_days"]): sel for sel in data}


def _cfg_from_selection(sel: dict, allow_missing: bool) -> TrainConfig:
    rec = sel["full_record"]
    tol = int(sel["tolerance_days"])
    bs  = int(sel["selected_model"]["batch_size"])
    return TrainConfig(
        tolerance_days=tol,
        batch_size=bs,
        learning_rate=float(rec.get("learning_rate", 1e-4)),
        weight_decay=float(rec.get("weight_decay", 1e-5)),
        early_stopping_patience=int(rec.get("early_stopping_patience", 15)),
        early_stopping_min_delta=float(rec.get("early_stopping_min_delta", 1e-4)),
        hidden_dim=int(rec.get("hidden_dim", 256)),
        text_dim=int(rec.get("text_dim", 128)),
        text_model_id=str(rec.get("text_model_id", "sentence-transformers/all-MiniLM-L6-v2")),
        text_model_trainable=bool(rec.get("text_model_trainable", False)),
        text_use_lora=bool(rec.get("text_use_lora", True)),
        text_use_qlora=bool(rec.get("text_use_qlora", False)),
        lora_r=int(rec.get("lora_r", 16)),
        lora_alpha=int(rec.get("lora_alpha", 32)),
        lora_dropout=float(rec.get("lora_dropout", 0.05)),
        use_vlm_bridge=bool(rec.get("use_vlm_bridge", True)),
        vision_rgb_model=str(rec.get("vision_rgb_model", "qwen_vl")),
        vision_rgb_model_id=str(rec.get("vision_rgb_model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
        vision_ms_model=str(rec.get("vision_ms_model", "spectral_cnn")),
        temporal_heads=int(rec.get("temporal_heads", 4)),
        temporal_layers=int(rec.get("temporal_layers", 2)),
        dropout=float(rec.get("dropout", 0.1)),
        covariance_rank=int(rec.get("covariance_rank", 0)),
        max_text_len=int(rec.get("max_text_len", 64)),
        lambda_h=float(rec.get("lambda_h", 1.0)),
        lambda_a=float(rec.get("lambda_a", 0.0)),
        lambda_r=float(rec.get("lambda_r", 0.0)),
        lambda_p=float(rec.get("lambda_p", 0.0)),
        use_wandb=False,
        optimizer=str(rec.get("optimizer", "adamw")),
        scheduler=str(rec.get("scheduler", "cosine")),
        gradient_clip_norm=float(rec.get("gradient_clip_norm", 1.0)),
        require_all_modalities=False if allow_missing else bool(rec.get("require_all_modalities", False)),
        min_multimodal_rows=int(rec.get("min_multimodal_rows", 32)),
        model_log_name=f"mtspect_tol{tol}_bs{bs}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8: Export MT-SpecT++ test predictions with coordinates."
    )
    parser.add_argument("--root",              default=".")
    parser.add_argument("--selection-json",    required=True)
    parser.add_argument("--tolerances",        default=None,
                        help="Comma-separated tolerances (default: all in selection-json).")
    parser.add_argument("--allow-missing-modalities", action="store_true")
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    selections = _load_selections(root / args.selection_json)
    tolerances = (
        [int(t.strip()) for t in args.tolerances.split(",")]
        if args.tolerances else sorted(selections.keys())
    )

    for tol in tolerances:
        if tol not in selections:
            print(f"[export] WARNING: no selection for τ={tol}d, skipping")
            continue

        cfg   = _cfg_from_selection(selections[tol], args.allow_missing_modalities)
        out   = (root / "artifacts" / f"tolerance_{tol}d"
                 / "results" / "mtspect" / "test_predictions_with_coords.csv")
        out.parent.mkdir(parents=True, exist_ok=True)

        print(f"[export] τ={tol}d — running inference …")
        try:
            preds = predict_mtspect(root=root, cfg=cfg)
            if preds is None or preds.empty:
                print(f"[export] τ={tol}d WARNING: empty predictions")
                continue
            preds.to_csv(out, index=False)
            print(f"[export] τ={tol}d → {out}  ({len(preds)} rows)")
        except Exception as exc:
            print(f"[export] τ={tol}d ERROR: {exc}")

    print("\n[export] Done.")


if __name__ == "__main__":
    main()
