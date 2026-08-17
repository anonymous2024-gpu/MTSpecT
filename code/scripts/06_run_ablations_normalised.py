"""
Step 6 — Normalised Ablation Metrics

Re-runs each ablation variant, exports per-target predictions, and computes
normalised metrics (nRMSE, nMAE, R²) for direct comparison in the paper's
ablation table.  Normalisation uses the training-set target standard deviation:
    nRMSE = RMSE / σ_train,  nMAE = MAE / σ_train

Usage
-----
    # All four tolerances
    python scripts/06_run_ablations_normalised.py \
        --tolerances 1 3 5 7 \
        --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
        --no-wandb --allow-missing-modalities

    # Specific ablation variants only
    python scripts/06_run_ablations_normalised.py \
        --tolerances 7 \
        --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
        --no-wandb \
        --ablations no_tabular no_temporal tabular_only temporal_only \
                    vision_only text_only no_route_reg no_aux_losses

Outputs
-------
    artifacts/tolerance_{N}d/results/mtspect/ablations/ablation_{name}_predictions.csv
    artifacts/tolerance_{N}d/results/mtspect/ablations/ablation_normalised_metrics.json
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wq_pipeline.mtspect.trainer import TrainConfig, train_mtspect, predict_mtspect

TARGETS = ["conductivity", "tn", "chl_a", "turbidity", "tp"]

ALL_ABLATION_NAMES = [
    "no_tabular", "no_temporal", "no_rgb", "no_npz", "no_text",
    "no_quality_gates",
    "fusion_sum", "fusion_mean",
    "deterministic",
    "tabular_only", "temporal_only", "vision_only", "text_only",
    "no_alignment_loss", "no_route_reg", "no_physics_loss", "no_aux_losses",
    "no_vlm_bridge", "text_frozen", "no_cross_attn_fusion",
]


def _load_selections(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(sel["tolerance_days"]): sel for sel in data}


def _sigma_train(root: Path, tol: int) -> dict[str, float]:
    """Per-target training standard deviation for normalisation."""
    train_csv = root / "artifacts" / f"tolerance_{tol}d" / "prepared" / "train.csv"
    if not train_csv.exists():
        return {}
    df = pd.read_csv(train_csv, low_memory=False)
    sigmas = {}
    for t in TARGETS:
        if t in df.columns:
            vals = pd.to_numeric(df[t], errors="coerce").dropna()
            if len(vals) > 1:
                sigmas[t] = float(vals.std(ddof=1))
    return sigmas


def _base_config(sel: dict, epochs: int, use_wandb: bool,
                 wandb_project: str, allow_missing: bool) -> TrainConfig:
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
        lambda_h=float(rec.get("lambda_h", 1.0)),
        lambda_a=float(rec.get("lambda_a", 0.0)),
        lambda_r=float(rec.get("lambda_r", 0.0)),
        lambda_p=float(rec.get("lambda_p", 0.0)),
        max_text_len=int(rec.get("max_text_len", 64)),
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=f"abl_norm_tol{tol}",
        optimizer=str(rec.get("optimizer", "adamw")),
        scheduler=str(rec.get("scheduler", "cosine")),
        gradient_clip_norm=float(rec.get("gradient_clip_norm", 1.0)),
        require_all_modalities=False if allow_missing else bool(rec.get("require_all_modalities", False)),
        min_multimodal_rows=int(rec.get("min_multimodal_rows", 32)),
        model_log_name=f"mtspect_ablation_tol{tol}",
        use_amp=bool(rec.get("use_amp", False)),
    )


def _apply_ablation(cfg: TrainConfig, name: str) -> TrainConfig:
    overrides = {
        "no_tabular":          {"enable_modality_tabular":  False},
        "no_temporal":         {"enable_modality_temporal": False},
        "no_rgb":              {"enable_modality_rgb":      False},
        "no_npz":              {"enable_modality_npz":      False},
        "no_text":             {"enable_modality_text":     False},
        "no_quality_gates":    {"enable_quality_gates":     False},
        "fusion_sum":          {"enable_fusion_mode":       "sum"},
        "fusion_mean":         {"enable_fusion_mode":       "mean"},
        "deterministic":       {"enable_uncertainty_head":  False},
        "tabular_only":        {"enable_modality_temporal": False,
                                "enable_modality_rgb": False,
                                "enable_modality_npz": False,
                                "enable_modality_text": False},
        "temporal_only":       {"enable_modality_tabular":  False,
                                "enable_modality_rgb": False,
                                "enable_modality_npz": False,
                                "enable_modality_text": False},
        "vision_only":         {"enable_modality_tabular":  False,
                                "enable_modality_temporal": False,
                                "enable_modality_text":     False},
        "text_only":           {"enable_modality_tabular":  False,
                                "enable_modality_temporal": False,
                                "enable_modality_rgb": False,
                                "enable_modality_npz": False},
        "no_alignment_loss":   {"lambda_a": 0.0},
        "no_route_reg":        {"lambda_r": 0.0},
        "no_physics_loss":     {"lambda_p": 0.0},
        "no_aux_losses":       {"lambda_a": 0.0, "lambda_r": 0.0, "lambda_p": 0.0},
        "no_vlm_bridge":       {"use_vlm_bridge":           False},
        "text_frozen":         {"text_use_lora":            False},
        "no_cross_attn_fusion": {"enable_cross_attn_expert": False},
    }.get(name, {})

    d = cfg.__dict__.copy()
    d.update(overrides)
    d["model_log_name"]  = f"{cfg.model_log_name}_{name}"
    d["wandb_run_name"]  = f"{cfg.wandb_run_name}_{name}"
    return TrainConfig(**d)


def _compute_normalised(preds_df: pd.DataFrame, sigmas: dict[str, float]) -> dict:
    results = {}
    for t in TARGETS:
        sub = preds_df[preds_df["target"] == t].copy()
        if sub.empty or t not in sigmas or sigmas[t] < 1e-9:
            continue
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        sigma  = sigmas[t]
        rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae    = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        results[t] = {
            "nrmse": rmse / sigma,
            "nmae":  mae  / sigma,
            "r2":    r2,
            "n":     int(len(sub)),
        }
    if results:
        macro_nrmse = float(np.mean([v["nrmse"] for v in results.values()]))
        macro_nmae  = float(np.mean([v["nmae"]  for v in results.values()]))
        macro_r2    = float(np.mean([v["r2"]    for v in results.values()]))
        results["macro"] = {"nrmse": macro_nrmse, "nmae": macro_nmae, "r2": macro_r2}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6: Compute normalised ablation metrics."
    )
    parser.add_argument("--root",              default=".")
    parser.add_argument("--selection-json",    required=True)
    parser.add_argument("--tolerances",        nargs="+", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--epochs",            type=int,  default=100)
    parser.add_argument("--ablations",         nargs="*", default=None,
                        help=f"Ablation names to run. Defaults: all {len(ALL_ABLATION_NAMES)}.")
    parser.add_argument("--use-wandb",         action="store_true", default=False)
    parser.add_argument("--no-wandb",          action="store_true")
    parser.add_argument("--wandb-project",     default="wq-mtspect-ablations")
    parser.add_argument("--allow-missing-modalities", action="store_true")
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    use_wandb  = args.use_wandb and not args.no_wandb
    abl_names  = args.ablations or ALL_ABLATION_NAMES
    selections = _load_selections(root / args.selection_json)

    for tol in args.tolerances:
        if tol not in selections:
            print(f"[abl-norm] WARNING: no selection for τ={tol}d, skipping")
            continue

        sigmas  = _sigma_train(root, tol)
        sel     = selections[tol]
        base    = _base_config(sel, args.epochs, use_wandb,
                               args.wandb_project, args.allow_missing_modalities)
        abl_dir = (root / "artifacts" / f"tolerance_{tol}d"
                   / "results" / "mtspect" / "ablations")
        abl_dir.mkdir(parents=True, exist_ok=True)

        tol_metrics: list[dict] = []

        for name in abl_names:
            print(f"[abl-norm] τ={tol}d  {name}")
            cfg = _apply_ablation(base, name)
            try:
                result = train_mtspect(root=root, cfg=cfg)
                preds  = predict_mtspect(root=root, cfg=cfg)
                if preds is not None and not preds.empty:
                    preds.to_csv(
                        abl_dir / f"ablation_{name}_predictions.csv", index=False
                    )
                    norm_metrics = _compute_normalised(preds, sigmas)
                else:
                    norm_metrics = {}
            except Exception as exc:
                print(f"[abl-norm] ERROR {name}: {exc}")
                norm_metrics = {"error": repr(exc)}

            tol_metrics.append({"ablation_name": name, "metrics": norm_metrics})

        out = abl_dir / "ablation_normalised_metrics.json"
        out.write_text(json.dumps(tol_metrics, indent=2, default=str), encoding="utf-8")
        print(f"[abl-norm] τ={tol}d → {out}")

    print("\n[abl-norm] Done.")


if __name__ == "__main__":
    main()
