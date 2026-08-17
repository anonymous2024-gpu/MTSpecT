"""
Step 5 — Ablation Studies

Trains 20 ablation variants of MT-SpecT++ per tolerance, using the best
model configuration selected in Step 4. Ablations cover:
  - Single-modality removal  (no_tabular, no_temporal, no_rgb, no_npz, no_text)
  - Quality gates disabled   (no_quality_gates)
  - Fusion mode variants     (fusion_sum, fusion_mean)
  - Deterministic head       (no uncertainty)
  - Multi-modality subsets   (tabular_only, temporal_only, vision_only, text_only)
  - Loss term ablations      (no_alignment_loss, no_route_reg, no_physics_loss, no_aux_losses)
  - VLM bridge removed       (no_vlm_bridge)
  - Text encoder frozen      (text_frozen)
  - Cross-attention removed  (no_cross_attn_fusion)

Usage
-----
    python scripts/05_run_ablations.py \
        --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
        --tolerances 1,3,5,7 \
        --epochs 100 \
        --allow-missing-modalities \
        --no-wandb

Outputs
-------
    artifacts/tolerance_{N}d/results/mtspect/ablations/ablation_{name}.json
    artifacts/prepared/mtspect_ablations.json   (combined, all tolerances)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from wq_pipeline.mtspect.trainer import TrainConfig, train_mtspect


def _load_selections(path: Path) -> Dict[int, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Selection file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(sel["tolerance_days"]): sel for sel in data}


def _base_config(sel: Dict, epochs: int, use_wandb: bool,
                 wandb_project: str, allow_missing: bool) -> TrainConfig:
    rec = sel["full_record"]
    tol = int(sel["tolerance_days"])
    bs  = int(sel["selected_model"]["batch_size"])
    return TrainConfig(
        tolerance_days=tol,
        batch_size=bs,
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
        wandb_run_name=f"mtspect_ablation_tol{tol}",
        optimizer=str(rec.get("optimizer", "adamw")),
        scheduler=str(rec.get("scheduler", "cosine")),
        gradient_clip_norm=float(rec.get("gradient_clip_norm", 1.0)),
        require_all_modalities=False if allow_missing else bool(rec.get("require_all_modalities", False)),
        min_multimodal_rows=int(rec.get("min_multimodal_rows", 32)),
        model_log_name=f"mtspect_ablation_tol{tol}",
        use_amp=bool(rec.get("use_amp", False)),
    )


def _ablation_configs(base: TrainConfig, tol: int,
                      wandb_prefix: str) -> List[tuple[str, TrainConfig]]:
    """Return list of (ablation_name, TrainConfig)."""

    def _variant(name: str, overrides: dict) -> tuple[str, TrainConfig]:
        cfg = base.__dict__.copy()
        cfg.update(overrides)
        cfg["wandb_run_name"]  = f"{wandb_prefix}_tol{tol}_{name}"
        cfg["model_log_name"]  = f"mtspect_ablation_tol{tol}_{name}"
        return name, TrainConfig(**cfg)

    return [
        # --- single modality removal ---
        _variant("no_tabular",  {"enable_modality_tabular":  False}),
        _variant("no_temporal", {"enable_modality_temporal": False}),
        _variant("no_rgb",      {"enable_modality_rgb":      False}),
        _variant("no_npz",      {"enable_modality_npz":      False}),
        _variant("no_text",     {"enable_modality_text":     False}),
        # --- quality gates ---
        _variant("no_quality_gates", {"enable_quality_gates": False}),
        # --- fusion variants ---
        _variant("fusion_sum",  {"enable_fusion_mode": "sum"}),
        _variant("fusion_mean", {"enable_fusion_mode": "mean"}),
        # --- uncertainty head ---
        _variant("deterministic", {"enable_uncertainty_head": False}),
        # --- multi-modality subsets ---
        _variant("tabular_only", {
            "enable_modality_temporal": False,
            "enable_modality_rgb":      False,
            "enable_modality_npz":      False,
            "enable_modality_text":     False,
        }),
        _variant("temporal_only", {
            "enable_modality_tabular": False,
            "enable_modality_rgb":     False,
            "enable_modality_npz":     False,
            "enable_modality_text":    False,
        }),
        _variant("vision_only", {
            "enable_modality_tabular":  False,
            "enable_modality_temporal": False,
            "enable_modality_text":     False,
        }),
        _variant("text_only", {
            "enable_modality_tabular":  False,
            "enable_modality_temporal": False,
            "enable_modality_rgb":      False,
            "enable_modality_npz":      False,
        }),
        # --- loss term ablations ---
        _variant("no_alignment_loss", {"lambda_a": 0.0}),
        _variant("no_route_reg",      {"lambda_r": 0.0}),
        _variant("no_physics_loss",   {"lambda_p": 0.0}),
        _variant("no_aux_losses",     {"lambda_a": 0.0, "lambda_r": 0.0, "lambda_p": 0.0}),
        # --- architecture ablations ---
        _variant("no_vlm_bridge",        {"use_vlm_bridge": False}),
        _variant("text_frozen",          {"text_use_lora": False}),
        _variant("no_cross_attn_fusion", {"enable_cross_attn_expert": False}),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: Run MT-SpecT++ ablation studies."
    )
    parser.add_argument("--root",              default=".")
    parser.add_argument("--selection-json",    required=True,
                        help="JSON from step 4: artifacts/prepared/mtspect_best_per_tolerance.json")
    parser.add_argument("--tolerances",        default=None,
                        help="Subset of tolerances to run (default: all in selection-json).")
    parser.add_argument("--epochs",            type=int, default=100)
    parser.add_argument("--use-wandb",         action="store_true", default=True)
    parser.add_argument("--no-wandb",          action="store_true")
    parser.add_argument("--wandb-project",     default="wq-mtspect-ablations")
    parser.add_argument("--wandb-run-prefix",  default="mtspect_ablation")
    parser.add_argument("--allow-missing-modalities", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.no_wandb:
        args.use_wandb = False

    selections = _load_selections(root / args.selection_json)
    tolerances = (
        [int(t.strip()) for t in args.tolerances.split(",")]
        if args.tolerances else sorted(selections.keys())
    )
    print(f"[ablations] Tolerances: {tolerances}")

    all_results: list[dict] = []

    for tol in tolerances:
        if tol not in selections:
            print(f"[ablations] WARNING: no selection for τ={tol}d, skipping")
            continue
        sel  = selections[tol]
        base = _base_config(sel, args.epochs, args.use_wandb,
                            args.wandb_project, args.allow_missing_modalities)
        variants = _ablation_configs(base, tol, args.wandb_run_prefix)
        print(f"[ablations] τ={tol}d — running {len(variants)} variants")

        abl_dir = (root / "artifacts" / f"tolerance_{tol}d"
                   / "results" / "mtspect" / "ablations")
        abl_dir.mkdir(parents=True, exist_ok=True)

        for i, (name, cfg) in enumerate(variants, 1):
            print(f"[ablations] τ={tol}d  [{i}/{len(variants)}]  {name}")
            try:
                result = train_mtspect(root=root, cfg=cfg)
                result["ablation_name"] = name
            except Exception as exc:
                result = {
                    "tolerance_days": tol,
                    "ablation_name":  name,
                    "status":         "error",
                    "error":          repr(exc),
                }
            result["tolerance_days"] = tol
            all_results.append(result)
            (abl_dir / f"ablation_{name}.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )

    out = root / "artifacts" / "prepared" / "mtspect_ablations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n[ablations] Combined results → {out}")


if __name__ == "__main__":
    main()
