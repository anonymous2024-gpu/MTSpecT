"""
Step 7 — Grouped K-Fold Cross-Validation

Runs 5-fold lake-grouped cross-validation for MT-SpecT++ at all tolerances.
Folds are split on lake_id to prevent data leakage.  The held-out test split
is never used; only train+valid are pooled for CV.

Usage
-----
    # Recommended: use best-model configs from Step 4
    python scripts/07_run_kfold.py \
        --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
        --n-folds 5 \
        --epochs 100 \
        --allow-missing-modalities \
        --no-wandb

    # Single tolerance
    python scripts/07_run_kfold.py \
        --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
        --tolerances 7 \
        --n-folds 5 --epochs 100 --no-wandb

Outputs
-------
    artifacts/tolerance_{N}d/results/mtspect/kfold/fold_{k}/metrics.json
    artifacts/tolerance_{N}d/results/mtspect/kfold_summary.json
    artifacts/tolerance_{N}d/results/mtspect/kfold_summary.csv
    artifacts/prepared/mtspect_kfold_all_tolerances.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from wq_pipeline.mtspect.trainer import TrainConfig, train_mtspect


def _load_selections(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(sel["tolerance_days"]): sel for sel in data}


def _config_from_selection(sel: dict, fold_idx: int, epochs: int,
                            use_wandb: bool, wandb_project: str,
                            wandb_prefix: str) -> TrainConfig:
    rec = sel["full_record"]
    tol = int(sel["tolerance_days"])
    return TrainConfig(
        tolerance_days=tol,
        batch_size=int(sel["selected_model"]["batch_size"]),
        epochs=epochs,
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
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=f"{wandb_prefix}_tol{tol}_fold{fold_idx}",
        optimizer=str(rec.get("optimizer", "adamw")),
        scheduler=str(rec.get("scheduler", "cosine")),
        gradient_clip_norm=float(rec.get("gradient_clip_norm", 1.0)),
        require_all_modalities=False,
        min_multimodal_rows=int(rec.get("min_multimodal_rows", 32)),
        model_log_name=f"mtspect_kfold_tol{tol}_fold{fold_idx}",
    )


def _is_finite(val) -> bool:
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Grouped k-fold cross-validation for MT-SpecT++."
    )
    parser.add_argument("--root",              default=".")
    parser.add_argument("--selection-json",    required=True)
    parser.add_argument("--tolerances",        default=None,
                        help="Comma-separated tolerances (default: all in selection-json).")
    parser.add_argument("--n-folds",           type=int, default=5)
    parser.add_argument("--epochs",            type=int, default=100)
    parser.add_argument("--use-wandb",         action="store_true", default=False)
    parser.add_argument("--no-wandb",          action="store_true")
    parser.add_argument("--wandb-project",     default="wq-mtspect")
    parser.add_argument("--wandb-run-prefix",  default="mtspect_kfold")
    parser.add_argument("--allow-missing-modalities", action="store_true")
    args = parser.parse_args()

    root      = Path(args.root).resolve()
    use_wandb = args.use_wandb and not args.no_wandb
    selections = _load_selections(root / args.selection_json)
    tolerances = (
        [int(t.strip()) for t in args.tolerances.split(",")]
        if args.tolerances else sorted(selections.keys())
    )

    all_summaries: list[dict] = []

    for tol in tolerances:
        if tol not in selections:
            print(f"[kfold] WARNING: no selection for τ={tol}d, skipping")
            continue

        prep = root / "artifacts" / f"tolerance_{tol}d" / "prepared"
        train_df = pd.read_csv(prep / "multimodal_train.csv")
        valid_df = pd.read_csv(prep / "multimodal_valid.csv")
        pool_df  = pd.concat([train_df, valid_df], ignore_index=True)

        if "lake_id" not in pool_df.columns:
            print(f"[kfold] τ={tol}d SKIP — missing lake_id column")
            continue

        pool_df["lake_id"] = pool_df["lake_id"].astype(str)
        gkf    = GroupKFold(n_splits=args.n_folds)
        groups = pool_df["lake_id"].values

        fold_results: list[dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
                gkf.split(pool_df, groups=groups)):
            fold_train = pool_df.iloc[train_idx].reset_index(drop=True)
            fold_val   = pool_df.iloc[val_idx].reset_index(drop=True)

            cfg = _config_from_selection(
                selections[tol], fold_idx, args.epochs,
                use_wandb, args.wandb_project, args.wandb_run_prefix
            )

            print(f"[kfold] τ={tol}d fold={fold_idx}  "
                  f"train={len(fold_train)} val={len(fold_val)}")

            try:
                result = train_mtspect(
                    root=root, cfg=cfg,
                    train_df_override=fold_train,
                    valid_df_override=fold_val,
                )
            except Exception as exc:
                result = {
                    "tolerance_days": tol,
                    "status": "error",
                    "error": repr(exc),
                }
            result["fold"] = fold_idx
            fold_results.append(result)

            fold_dir = (root / "artifacts" / f"tolerance_{tol}d"
                        / "results" / "mtspect" / "kfold" / f"fold_{fold_idx}")
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "metrics.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(f"[kfold] τ={tol}d fold={fold_idx} "
                  f"status={result.get('status')} "
                  f"mean_rmse={result.get('mean_rmse')}")

        # ---- Aggregate -------------------------------------------------------
        ok = [r for r in fold_results if r.get("status") == "ok"]
        agg: dict = {
            "tolerance_days": tol,
            "n_folds":        args.n_folds,
            "n_ok_folds":     len(ok),
        }
        for key in ["best_val_loss", "best_val_r2", "mean_rmse", "mean_mae", "mean_r2"]:
            vals = [float(r[key]) for r in ok
                    if r.get(key) is not None and _is_finite(r[key])]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"]  = float(np.std(vals, ddof=1))

        fold_rows = [{
            "fold": r.get("fold"), "tolerance_days": tol,
            "status": r.get("status"),
            "best_val_loss": r.get("best_val_loss"),
            "best_val_r2":   r.get("best_val_r2"),
            "mean_rmse":     r.get("mean_rmse"),
            "mean_mae":      r.get("mean_mae"),
            "mean_r2":       r.get("mean_r2"),
        } for r in fold_results]

        out_dir = root / "artifacts" / f"tolerance_{tol}d" / "results" / "mtspect"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fold_rows).to_csv(out_dir / "kfold_summary.csv", index=False)
        (out_dir / "kfold_summary.json").write_text(
            json.dumps({"aggregate": agg, "folds": fold_rows}, indent=2),
            encoding="utf-8"
        )
        print(f"[kfold] τ={tol}d aggregate: {agg}")
        all_summaries.append({"tolerance_days": tol, "aggregate": agg, "folds": fold_rows})

    combined = root / "artifacts" / "prepared" / "mtspect_kfold_all_tolerances.json"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\n[kfold] All-tolerance summary → {combined}")


if __name__ == "__main__":
    main()
