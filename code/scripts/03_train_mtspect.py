"""
Step 3 — MT-SpecT++ Training Sweep

Trains MT-SpecT++ across all tolerance windows and batch sizes.
Produces one checkpoint per (tolerance, batch_size) combination and a
consolidated summary JSON for downstream model selection.

Usage
-----
    # Standard grid (paper configuration)
    python scripts/03_train_mtspect.py \
        --tolerances 1,3,5,7 \
        --batch-sizes 32,64,128,256 \
        --epochs 100

    # Single run for a specific tolerance
    python scripts/03_train_mtspect.py --tolerances 7 --batch-sizes 32 --epochs 100

    # Full auto-tune sweep (longer)
    python scripts/03_train_mtspect.py \
        --tolerances 1,3,5,7 \
        --batch-sizes 32,64,128,256 \
        --auto-tune \
        --epochs 100

    # Reproduce paper results exactly (best configs per tolerance)
    python scripts/03_train_mtspect.py \
        --tolerances 1 --batch-sizes 128 \
        --tolerances 3 --batch-sizes 64 \
        --tolerances 5 --batch-sizes 32 \
        --tolerances 7 --batch-sizes 32

Outputs
-------
    artifacts/tolerance_{N}d/results/mtspect/mtspect_tol{N}_bs{K}/
        checkpoint.pt
        metrics.json
    artifacts/prepared/mtspect_run_summary.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from wq_pipeline.mtspect.trainer import TrainConfig, train_mtspect


def _parse_list_int(raw: str) -> list[int]:
    vals = [int(p.strip()) for p in str(raw).split(",") if p.strip()]
    if not vals:
        raise ValueError(f"Could not parse integer list from: {raw!r}")
    return vals


def _parse_list_float(raw: str) -> list[float]:
    return [float(p.strip()) for p in str(raw).split(",") if p.strip()]


def _parse_list_str(raw: str) -> list[str]:
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _best_baseline_rmse(root: Path, tolerance_days: int) -> float | None:
    """Load best baseline macro-RMSE for a given tolerance (used for logging only)."""
    path = root / "artifacts" / "prepared" / "baseline_metric_summary.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "tolerance_days" not in df.columns or "macro_rmse" not in df.columns:
        return None
    sub = df[df["tolerance_days"] == tolerance_days]
    vals = pd.to_numeric(sub["macro_rmse"], errors="coerce").dropna()
    return float(vals.min()) if not vals.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Train MT-SpecT++ across tolerance/batch-size grid."
    )
    parser.add_argument("--root", default=".",
                        help="Project root directory.")
    parser.add_argument("--tolerances", default="1,3,5,7",
                        help="Comma-separated tolerance windows in days.")
    parser.add_argument("--batch-sizes", default="32,64,128,256",
                        help="Comma-separated batch sizes to sweep.")
    parser.add_argument("--epochs",                    type=int,   default=100)
    parser.add_argument("--learning-rate",             type=float, default=1e-4)
    parser.add_argument("--weight-decay",              type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience",   type=int,   default=15)
    parser.add_argument("--early-stopping-min-delta",  type=float, default=1e-4)
    parser.add_argument("--hidden-dim",                type=int,   default=256)
    parser.add_argument("--text-dim",                  type=int,   default=128)
    parser.add_argument("--text-model-id",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text-model-trainable",      action="store_true")
    parser.add_argument("--text-use-lora",             action="store_true", default=True)
    parser.add_argument("--no-text-lora",              action="store_true")
    parser.add_argument("--text-use-qlora",            action="store_true")
    parser.add_argument("--lora-r",                    type=int,   default=16)
    parser.add_argument("--lora-alpha",                type=int,   default=32)
    parser.add_argument("--lora-dropout",              type=float, default=0.05)
    parser.add_argument("--use-vlm-bridge",            action="store_true", default=True)
    parser.add_argument("--no-vlm-bridge",             action="store_true")
    parser.add_argument("--vision-rgb-model",          default="qwen_vl",
                        choices=["qwen_vl", "timm", "cnn", "qwen_vl_only"])
    parser.add_argument("--vision-rgb-model-id",
                        default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--vision-ms-model",           default="spectral_cnn",
                        choices=["spectral_cnn"])
    parser.add_argument("--temporal-heads",            type=int,   default=4)
    parser.add_argument("--temporal-layers",           type=int,   default=2)
    parser.add_argument("--dropout",                   type=float, default=0.1)
    parser.add_argument("--covariance-rank",           type=int,   default=0)
    parser.add_argument("--max-text-len",              type=int,   default=64)
    parser.add_argument("--lambda-h",                  type=float, default=1.0)
    parser.add_argument("--lambda-a",                  type=float, default=0.0)
    parser.add_argument("--lambda-r",                  type=float, default=0.0)
    parser.add_argument("--lambda-p",                  type=float, default=0.0)
    parser.add_argument("--aux-warmup-epochs",         type=int,   default=20)
    parser.add_argument("--prefit-l2",                 type=float, default=1e-3)
    parser.add_argument("--optimizer",                 default="adamw",
                        choices=["adamw", "radam"])
    parser.add_argument("--scheduler",                 default="cosine",
                        choices=["cosine", "onecycle", "none"])
    parser.add_argument("--gradient-clip-norm",        type=float, default=1.0)
    parser.add_argument("--require-all-modalities",    action="store_true", default=True)
    parser.add_argument("--allow-missing-modalities",  action="store_true")
    parser.add_argument("--min-multimodal-rows",       type=int,   default=32)
    parser.add_argument("--targets",                   default=None,
                        help="Comma-separated targets (e.g. chl_a,tn,tp,turbidity). "
                             "Omit to use all 5 defaults.")
    parser.add_argument("--enable-profiling",          action="store_true",
                        help="Profile GFLOPs and inference latency.")
    parser.add_argument("--use-amp",                   action="store_true",
                        help="Enable automatic mixed precision (FP16).")
    parser.add_argument("--text-lr-scale",             type=float, default=1.0)
    parser.add_argument("--vision-lr-scale",           type=float, default=1.0)
    parser.add_argument("--curriculum",                action="store_true",
                        help="Run tolerances in ascending order (curriculum learning).")
    parser.add_argument("--auto-tune",                 action="store_true",
                        help="Grid-search over LR, hidden-dim, dropout, optimizer.")
    parser.add_argument("--tune-learning-rates",       default="3e-4,1e-4,5e-5")
    parser.add_argument("--tune-hidden-dims",          default="128,256")
    parser.add_argument("--tune-dropouts",             default="0.1,0.2")
    parser.add_argument("--tune-optimizers",           default="adamw,radam")
    parser.add_argument("--use-wandb",                 action="store_true", default=True)
    parser.add_argument("--no-wandb",                  action="store_true")
    parser.add_argument("--wandb-project",             default="wq-mtspect")
    parser.add_argument("--wandb-run-prefix",          default="mtspect")
    parser.add_argument("--summary-json",
                        default="artifacts/prepared/mtspect_run_summary.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tolerances = (sorted(_parse_list_int(args.tolerances))
                  if args.curriculum else _parse_list_int(args.tolerances))
    batches = _parse_list_int(args.batch_sizes)

    if args.no_wandb:
        args.use_wandb = False
    if args.no_text_lora:
        args.text_use_lora = False
    if args.no_vlm_bridge:
        args.use_vlm_bridge = False
    if args.allow_missing_modalities:
        args.require_all_modalities = False

    targets = ([t.strip() for t in args.targets.split(",")]
               if args.targets else None)

    tune_lrs   = _parse_list_float(args.tune_learning_rates)
    tune_hids  = _parse_list_int(args.tune_hidden_dims)
    tune_drops = _parse_list_float(args.tune_dropouts)
    tune_opts  = _parse_list_str(args.tune_optimizers)

    rows: list[dict] = []
    prev_best_ckpt: str | None = None

    for tol in tolerances:
        baseline_rmse = _best_baseline_rmse(root, tol)

        if args.auto_tune:
            candidates: list[TrainConfig] = [
                TrainConfig(
                    tolerance_days=int(tol),
                    batch_size=int(bs),
                    epochs=int(args.epochs),
                    learning_rate=float(lr),
                    weight_decay=float(args.weight_decay),
                    early_stopping_patience=int(args.early_stopping_patience),
                    early_stopping_min_delta=float(args.early_stopping_min_delta),
                    hidden_dim=int(hid),
                    text_dim=int(args.text_dim),
                    text_model_id=str(args.text_model_id),
                    text_model_trainable=bool(args.text_model_trainable),
                    text_use_lora=bool(args.text_use_lora),
                    text_use_qlora=bool(args.text_use_qlora),
                    lora_r=int(args.lora_r),
                    lora_alpha=int(args.lora_alpha),
                    lora_dropout=float(args.lora_dropout),
                    use_vlm_bridge=bool(args.use_vlm_bridge),
                    vision_rgb_model=str(args.vision_rgb_model),
                    vision_rgb_model_id=str(args.vision_rgb_model_id),
                    vision_ms_model=str(args.vision_ms_model),
                    temporal_heads=int(args.temporal_heads),
                    temporal_layers=int(args.temporal_layers),
                    dropout=float(dr),
                    covariance_rank=int(args.covariance_rank),
                    lambda_h=float(args.lambda_h),
                    lambda_a=float(args.lambda_a),
                    lambda_r=float(args.lambda_r),
                    lambda_p=float(args.lambda_p),
                    max_text_len=int(args.max_text_len),
                    use_wandb=bool(args.use_wandb),
                    wandb_project=str(args.wandb_project),
                    wandb_run_name=f"{args.wandb_run_prefix}_tol{tol}_bs{bs}_lr{lr}_h{hid}_d{dr}_{op}",
                    optimizer=str(op),
                    scheduler=str(args.scheduler),
                    gradient_clip_norm=float(args.gradient_clip_norm),
                    aux_warmup_epochs=int(args.aux_warmup_epochs),
                    prefit_l2=float(args.prefit_l2),
                    require_all_modalities=bool(args.require_all_modalities),
                    min_multimodal_rows=int(args.min_multimodal_rows),
                    targets=targets,
                    enable_profiling=bool(args.enable_profiling),
                    use_amp=bool(args.use_amp),
                    text_lr_scale=float(args.text_lr_scale),
                    vision_lr_scale=float(args.vision_lr_scale),
                )
                for bs in batches
                for lr in tune_lrs
                for hid in tune_hids
                for dr in tune_drops
                for op in tune_opts
            ]
        else:
            candidates = [
                TrainConfig(
                    tolerance_days=int(tol),
                    batch_size=int(bs),
                    epochs=int(args.epochs),
                    learning_rate=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                    early_stopping_patience=int(args.early_stopping_patience),
                    early_stopping_min_delta=float(args.early_stopping_min_delta),
                    hidden_dim=int(args.hidden_dim),
                    text_dim=int(args.text_dim),
                    text_model_id=str(args.text_model_id),
                    text_model_trainable=bool(args.text_model_trainable),
                    text_use_lora=bool(args.text_use_lora),
                    text_use_qlora=bool(args.text_use_qlora),
                    lora_r=int(args.lora_r),
                    lora_alpha=int(args.lora_alpha),
                    lora_dropout=float(args.lora_dropout),
                    use_vlm_bridge=bool(args.use_vlm_bridge),
                    vision_rgb_model=str(args.vision_rgb_model),
                    vision_rgb_model_id=str(args.vision_rgb_model_id),
                    vision_ms_model=str(args.vision_ms_model),
                    temporal_heads=int(args.temporal_heads),
                    temporal_layers=int(args.temporal_layers),
                    dropout=float(args.dropout),
                    covariance_rank=int(args.covariance_rank),
                    lambda_h=float(args.lambda_h),
                    lambda_a=float(args.lambda_a),
                    lambda_r=float(args.lambda_r),
                    lambda_p=float(args.lambda_p),
                    max_text_len=int(args.max_text_len),
                    use_wandb=bool(args.use_wandb),
                    wandb_project=str(args.wandb_project),
                    wandb_run_name=f"{args.wandb_run_prefix}_tol{tol}_bs{bs}",
                    optimizer=str(args.optimizer),
                    scheduler=str(args.scheduler),
                    gradient_clip_norm=float(args.gradient_clip_norm),
                    aux_warmup_epochs=int(args.aux_warmup_epochs),
                    prefit_l2=float(args.prefit_l2),
                    require_all_modalities=bool(args.require_all_modalities),
                    min_multimodal_rows=int(args.min_multimodal_rows),
                    targets=targets,
                    enable_profiling=bool(args.enable_profiling),
                    use_amp=bool(args.use_amp),
                    text_lr_scale=float(args.text_lr_scale),
                    vision_lr_scale=float(args.vision_lr_scale),
                )
                for bs in batches
            ]

        tol_runs: list[dict] = []
        for cfg in candidates:
            if args.curriculum and prev_best_ckpt:
                cfg = replace(cfg, warm_start_checkpoint=prev_best_ckpt)
            try:
                summary = train_mtspect(root=root, cfg=cfg)
            except Exception as exc:
                summary = {
                    "family": "mtspect",
                    "tolerance_days": int(tol),
                    "batch_size": int(cfg.batch_size),
                    "status": "error",
                    "error": repr(exc),
                }
            if baseline_rmse is not None and summary.get("mean_rmse") is not None:
                summary["baseline_best_rmse"] = float(baseline_rmse)
                summary["beats_baseline"] = bool(float(summary["mean_rmse"]) < baseline_rmse)
            tol_runs.append(summary)
            rows.append(summary)

        ok_runs = [r for r in tol_runs if r.get("status") == "ok"]
        if ok_runs:
            ok_runs.sort(key=lambda r: (
                float(r.get("mean_rmse") or 1e18),
                float(r.get("best_val_loss") or 1e18),
                -(float(r.get("best_val_r2") or -1e9)
                  if pd.notna(r.get("best_val_r2")) else -1e9),
            ))
            best = dict(ok_runs[0])
            best["selected_best_for_tolerance"] = True
            best["status"] = "selected"
            rows.append(best)
            print(f"[train] tol={tol} best bs={best.get('batch_size')} "
                  f"val_r2={best.get('best_val_r2')} mean_rmse={best.get('mean_rmse')}")
            if args.curriculum:
                prev_best_ckpt = ok_runs[0].get("checkpoint_path")

    out_path = (root / args.summary_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    run_rows = [r for r in rows if not r.get("selected_best_for_tolerance")]
    ok_count = sum(1 for r in run_rows if str(r.get("status")) == "ok")
    print(f"\n[train] Completed {ok_count}/{len(run_rows)} runs.")
    print(f"[train] Summary → {out_path}")


if __name__ == "__main__":
    main()
