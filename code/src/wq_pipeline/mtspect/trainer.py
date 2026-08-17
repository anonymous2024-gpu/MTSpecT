from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wq_pipeline.data.io import ensure_dir

from .data import (
    TARGETS,
    BatchTensors,
    DEFAULT_NPZ_BAND_ORDER,
    HFPromptTokenizer,
    MTSpectDataset,
    collate_batch,
    fit_normalization_stats,
    infer_feature_spec,
    invert_target_scaling,
    load_multimodal_split_frames,
)
from .losses import alignment_infonce, masked_huber, masked_nll, physics_penalty, route_regularizer
from .metrics import bootstrap_metric_stats, gaussian_crps, gaussian_nll, prediction_interval_metrics, regression_ece, regression_metrics
from .model import MTSpectConfig, MTSpectModel
from .visualize import plot_training_curves
from .wandb_utils import init_wandb


@dataclass
class TrainConfig:
    tolerance_days: int
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    hidden_dim: int = 256
    text_dim: int = 128
    temporal_heads: int = 4
    temporal_layers: int = 2
    dropout: float = 0.1
    covariance_rank: int = 0
    lambda_h: float = 1.0
    lambda_a: float = 0.0
    lambda_r: float = 0.0
    lambda_p: float = 0.0
    aux_warmup_epochs: int = 20
    max_text_len: int = 64
    num_workers: int = 0
    use_wandb: bool = True
    wandb_project: str = "wq-mtspect"
    wandb_run_name: str = "mtspect"
    text_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_model_trainable: bool = False
    text_use_lora: bool = True
    text_use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_vlm_bridge: bool = True
    vision_rgb_model: str = "qwen_vl"
    vision_rgb_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vision_ms_model: str = "spectral_cnn"
    model_log_name: str = "mtspect"
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip_norm: float = 1.0
    targets: list[str] | None = None  # None → use all 5 defaults (TARGETS)
    use_amp: bool = False
    text_lr_scale: float = 1.0
    vision_lr_scale: float = 1.0
    stability_mode: bool = False
    normalize_inputs: bool = True
    normalize_targets: bool = True
    require_all_modalities: bool = True
    min_multimodal_rows: int = 32
    enable_profiling: bool = False
    restore_best_state: bool = True
    prefit_tabular_skip: bool = True
    prefit_l2: float = 1e-3
    warm_start_checkpoint: str | None = None  # Path to .pt file to load weights from before training
    # Ablation flags
    enable_modality_tabular: bool = True
    enable_modality_temporal: bool = True
    enable_modality_rgb: bool = True
    enable_modality_npz: bool = True
    enable_modality_text: bool = True
    enable_quality_gates: bool = True
    enable_fusion_mode: str = "concat"  # "concat", "sum", "mean"
    enable_uncertainty_head: bool = True  # If False, use deterministic head
    enable_cross_attn_expert: bool = True  # MoE additive-only when False
    enable_spectral_adapter: bool = True   # If False, replace SSA with linear projection
    enable_band_bridge: bool = True        # If False, disable SpectralBandBridge (NPZ→RGB path for VLM)
    enable_tabular_skip: bool = True       # If False, remove direct tabular→output skip connection


def _to_device(batch: BatchTensors, device: str) -> BatchTensors:
    return BatchTensors(
        tabular_x=batch.tabular_x.to(device),
        temporal_x=batch.temporal_x.to(device),
        quality_x=batch.quality_x.to(device),
        rgb_x=batch.rgb_x.to(device),
        npz_x=batch.npz_x.to(device),
        text_tokens=batch.text_tokens.to(device),
        text_mask=batch.text_mask.to(device),
        y=batch.y.to(device),
        y_mask=batch.y_mask.to(device),
    )


def _set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "cuda_available": True,
        "gpu_name": props.name,
        "gpu_total_memory_gb": float(props.total_memory / (1024**3)),
        "gpu_sm_count": int(props.multi_processor_count),
        "gpu_compute_capability": f"{props.major}.{props.minor}",
    }


def _profile_flops_and_throughput(model: MTSpectModel, batch: BatchTensors, device: str) -> dict:
    b = _to_device(batch, device)
    model.eval()
    with torch.no_grad():
        _ = model(b.tabular_x, b.temporal_x, b.quality_x, b.rgb_x, b.npz_x, b.text_tokens, b.text_mask)

    sample_count = int(b.tabular_x.shape[0])
    inference_ms_per_sample = None
    samples_per_sec = None
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(b.tabular_x, b.temporal_x, b.quality_x, b.rgb_x, b.npz_x, b.text_tokens, b.text_mask)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = max(time.time() - t0, 1e-9)
    inference_ms_per_sample = 1000.0 * dt / max(sample_count, 1)
    samples_per_sec = sample_count / dt

    flops_total = None
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.no_grad():
                _ = model(b.tabular_x, b.temporal_x, b.quality_x, b.rgb_x, b.npz_x, b.text_tokens, b.text_mask)
        flops_total = float(sum(float(getattr(e, "flops", 0) or 0) for e in prof.key_averages()))
    except Exception:
        flops_total = None

    gflops_per_sample = (flops_total / 1e9 / max(sample_count, 1)) if flops_total is not None else None
    return {
        "profile_batch_size": sample_count,
        "profile_inference_ms_per_sample": inference_ms_per_sample,
        "profile_samples_per_sec": samples_per_sec,
        "profile_flops_total": flops_total,
        "profile_gflops_per_sample": gflops_per_sample,
    }


def _eval(model: MTSpectModel, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_w = 0.0
    r2_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            out = model(b.tabular_x, b.temporal_x, b.quality_x, b.rgb_x, b.npz_x, b.text_tokens, b.text_mask)
            nll = masked_nll(out["mu"], out["logvar"], b.y, b.y_mask)
            w = float(b.y_mask.sum().item())
            total_loss += float(nll.item()) * w
            total_w += w

            y_true = b.y.detach().cpu().numpy()
            y_pred = out["mu"].detach().cpu().numpy()
            y_mask = b.y_mask.detach().cpu().numpy() > 0.5
            for ti in range(y_true.shape[1]):
                m = y_mask[:, ti]
                if m.sum() < 2:
                    continue
                truth = y_true[m, ti]
                pred = y_pred[m, ti]
                denom = float(np.sum((truth - truth.mean()) ** 2))
                if denom <= 1e-4:
                    continue
                r2 = float(1.0 - np.sum((pred - truth) ** 2) / denom)
                r2_values.append(r2)

    val_loss = total_loss / max(total_w, 1e-6)
    val_r2 = float(np.mean(r2_values)) if r2_values else float("nan")
    return float(val_loss), val_r2


def _predict(model: MTSpectModel, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    lvs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            out = model(b.tabular_x, b.temporal_x, b.quality_x, b.rgb_x, b.npz_x, b.text_tokens, b.text_mask)
            ys.append(b.y.detach().cpu().numpy())
            ms.append(b.y_mask.detach().cpu().numpy())
            mus.append(out["mu"].detach().cpu().numpy())
            lvs.append(out["logvar"].detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ms, axis=0), np.concatenate(mus, axis=0), np.concatenate(lvs, axis=0)


def train_mtspect(
    root: Path,
    cfg: TrainConfig,
    train_df_override: pd.DataFrame | None = None,
    valid_df_override: pd.DataFrame | None = None,
) -> dict:
    root = root.resolve()
    prep_dir = root / "artifacts" / f"tolerance_{cfg.tolerance_days}d" / "prepared"

    if str(cfg.vision_ms_model).strip().lower() != "spectral_cnn":
        raise ValueError("vision_ms_model must be 'spectral_cnn' for strict multimodal NPZ support.")
    if str(cfg.vision_rgb_model).strip().lower() == "qwen_vl_only" and bool(cfg.require_all_modalities):
        raise ValueError(
            "Invalid config: qwen_vl_only cannot be used when NPZ strict multimodal is enabled. "
            "Use vision_rgb_model='qwen_vl' and vision_ms_model='spectral_cnn'."
        )

    if train_df_override is not None and valid_df_override is not None:
        train_df, valid_df = train_df_override, valid_df_override
        # Test set is always the held-out split from disk; never participates in CV folds
        _, _, test_df, modality_coverage = load_multimodal_split_frames(
            prep_dir,
            require_all_modalities=bool(cfg.require_all_modalities),
            min_rows=1,
        )
        modality_coverage["train_rows_kept"] = len(train_df)
        modality_coverage["valid_rows_kept"] = len(valid_df)
    else:
        train_df, valid_df, test_df, modality_coverage = load_multimodal_split_frames(
            prep_dir,
            require_all_modalities=bool(cfg.require_all_modalities),
            min_rows=int(cfg.min_multimodal_rows),
        )

    _set_reproducibility(seed=42 + int(cfg.tolerance_days) + int(cfg.batch_size))
    active_targets = cfg.targets or TARGETS
    feature_spec = infer_feature_spec(train_df, targets=active_targets)
    tok = HFPromptTokenizer(model_id=cfg.text_model_id, max_len=cfg.max_text_len)

    train_ds = MTSpectDataset(train_df, feature_spec=feature_spec, tokenizer=tok, max_text_len=cfg.max_text_len, targets=active_targets)
    valid_ds = MTSpectDataset(valid_df, feature_spec=feature_spec, tokenizer=tok, max_text_len=cfg.max_text_len, targets=active_targets)
    test_ds = MTSpectDataset(test_df, feature_spec=feature_spec, tokenizer=tok, max_text_len=cfg.max_text_len, targets=active_targets)

    norm_stats = None
    if cfg.normalize_inputs or cfg.normalize_targets:
        norm_stats = fit_normalization_stats(train_ds)
        train_ds.apply_normalization(norm_stats, normalize_targets=cfg.normalize_targets)
        valid_ds.apply_normalization(norm_stats, normalize_targets=cfg.normalize_targets)
        test_ds.apply_normalization(norm_stats, normalize_targets=cfg.normalize_targets)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = MTSpectConfig(
        tabular_dim=len(feature_spec.tabular_columns),
        temporal_band_dim=len(feature_spec.temporal_band_columns),
        quality_dim=max(1, len(feature_spec.quality_columns)),
        n_targets=len(train_ds.targets),
        vocab_size=0,
        text_model_id=cfg.text_model_id,
        text_model_trainable=cfg.text_model_trainable,
        text_use_lora=cfg.text_use_lora,
        text_use_qlora=cfg.text_use_qlora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        text_max_len=cfg.max_text_len,
        hidden_dim=cfg.hidden_dim,
        text_dim=cfg.text_dim,
        temporal_heads=cfg.temporal_heads,
        temporal_layers=cfg.temporal_layers,
        dropout=cfg.dropout,
        covariance_rank=cfg.covariance_rank,
        vision_rgb_model=cfg.vision_rgb_model,
        vision_rgb_model_id=cfg.vision_rgb_model_id,
        vision_ms_model=cfg.vision_ms_model,
        npz_channels=len(DEFAULT_NPZ_BAND_ORDER),
        use_vlm_bridge=cfg.use_vlm_bridge,
        # Ablation flags
        enable_modality_tabular=cfg.enable_modality_tabular,
        enable_modality_temporal=cfg.enable_modality_temporal,
        enable_modality_rgb=cfg.enable_modality_rgb,
        enable_modality_npz=cfg.enable_modality_npz,
        enable_modality_text=cfg.enable_modality_text,
        enable_quality_gates=cfg.enable_quality_gates,
        enable_fusion_mode=cfg.enable_fusion_mode,
        enable_uncertainty_head=cfg.enable_uncertainty_head,
        enable_cross_attn_expert=cfg.enable_cross_attn_expert,
        enable_spectral_adapter=cfg.enable_spectral_adapter,
        enable_band_bridge=cfg.enable_band_bridge,
        enable_tabular_skip=cfg.enable_tabular_skip,
    )
    model = MTSpectModel(model_cfg).to(device)

    if cfg.warm_start_checkpoint:
        _ws = torch.load(cfg.warm_start_checkpoint, map_location=device)
        _sd = _ws.get("model_state_dict", _ws)
        _missing, _unexpected = model.load_state_dict(_sd, strict=False)
        print(
            f"[warm_start] Loaded weights from {cfg.warm_start_checkpoint} "
            f"(missing={len(_missing)}, unexpected={len(_unexpected)})"
        )

    if bool(cfg.prefit_tabular_skip) and hasattr(model, "tabular_skip"):
        try:
            x = np.asarray(train_ds.tabular, dtype=np.float64)
            y = np.asarray(train_ds.y, dtype=np.float64)
            y_mask = np.asarray(train_ds.y_mask, dtype=np.float64) > 0.5
            n_features = int(x.shape[1])
            n_targets = int(y.shape[1])
            w = np.zeros((n_targets, n_features), dtype=np.float32)
            b = np.zeros((n_targets,), dtype=np.float32)
            l2 = float(max(0.0, cfg.prefit_l2))

            for ti in range(n_targets):
                m = y_mask[:, ti]
                if int(m.sum()) < max(8, n_features // 2):
                    continue
                xt = x[m]
                yt = y[m, ti]
                xb = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float64)], axis=1)
                eye = np.eye(xb.shape[1], dtype=np.float64)
                eye[-1, -1] = 0.0
                a = xb.T @ xb + l2 * eye
                c = xb.T @ yt
                try:
                    sol = np.linalg.solve(a, c)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(a, c, rcond=None)[0]
                w[ti, :] = sol[:-1].astype(np.float32)
                b[ti] = np.float32(sol[-1])

            with torch.no_grad():
                model.tabular_skip.weight.copy_(torch.tensor(w, device=model.tabular_skip.weight.device))
                model.tabular_skip.bias.copy_(torch.tensor(b, device=model.tabular_skip.bias.device))
        except Exception:
            pass

    # Build per-module param groups so text/vision encoders can use scaled LRs
    text_ids = {id(p) for p in model.text.parameters()}
    vision_ids = {id(p) for p in list(model.rgb_backbone.parameters()) + list(model.npz_backbone.parameters())}
    param_groups = [
        {"params": [p for p in model.parameters() if id(p) not in text_ids and id(p) not in vision_ids], "lr": cfg.learning_rate},
        {"params": [p for p in model.text.parameters()], "lr": cfg.learning_rate * cfg.text_lr_scale},
        {"params": [p for p in model.rgb_backbone.parameters()] + [p for p in model.npz_backbone.parameters()], "lr": cfg.learning_rate * cfg.vision_lr_scale},
    ]
    opt_name = str(cfg.optimizer).strip().lower()
    if opt_name == "radam":
        optimizer = torch.optim.RAdam(param_groups, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp))

    scheduler = None
    sched_name = str(cfg.scheduler).strip().lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    elif sched_name == "onecycle":
        total_steps = max(1, cfg.epochs * max(1, len(train_loader)))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    repro = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "seed": int(42 + int(cfg.tolerance_days) + int(cfg.batch_size)),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        **_gpu_info(),
        "n_params_total": total_params,
        "n_params_trainable": trainable_params,
    }

    profile_info = {}
    if bool(cfg.enable_profiling):
        try:
            first_batch = next(iter(valid_loader))
            profile_info = _profile_flops_and_throughput(model, first_batch, device=device)
        except Exception:
            profile_info = {}

    wandb_handle = init_wandb(
        enabled=cfg.use_wandb,
        project=cfg.wandb_project,
        run_name=f"{cfg.wandb_run_name}-tol{cfg.tolerance_days}",
        config={**asdict(cfg), **asdict(model_cfg), **repro, **profile_info},
    )
    wandb_handle.log({"system_info": repro, "profile_info": profile_info})

    tmp_dir = ensure_dir(root / "artifacts" / "prepared" / "tmp")
    best_ckpt_tmp = tmp_dir / f"mtspect_best_tol{cfg.tolerance_days}_bs{cfg.batch_size}_{os.getpid()}.pt"

    history_rows: list[dict] = []
    best_val = float("inf")
    best_val_r2 = float("nan")
    best_epoch = 0
    no_improve = 0
    skipped_nonfinite_steps = 0

    run_started = time.time()
    amp_ctx = torch.cuda.amp.autocast if bool(cfg.use_amp) else contextlib.nullcontext
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        e0 = time.time()
        train_loss_acc = 0.0
        train_w_acc = 0.0
        aux_scale = min(1.0, float(epoch) / max(1.0, float(cfg.aux_warmup_epochs)))
        for batch in train_loader:
            b = _to_device(batch, device)
            with amp_ctx():
                out = model(b.tabular_x, b.temporal_x, b.quality_x, b.rgb_x, b.npz_x, b.text_tokens, b.text_mask)

                nll = masked_nll(out["mu"], out["logvar"], b.y, b.y_mask)
                huber = masked_huber(out["mu"], b.y, b.y_mask)
                align = alignment_infonce(out["z_fuse"], out["z_text"]) if b.y.shape[0] > 1 else torch.tensor(0.0, device=b.y.device)
                route = route_regularizer(out["mix_add"], out["mix_xattn"], out["alpha_v"], out["alpha_t"], out["alpha_l"])
                phys = physics_penalty(out["mu"])
                loss = nll + cfg.lambda_h * huber + aux_scale * (cfg.lambda_a * align + cfg.lambda_r * route + cfg.lambda_p * phys)

            if not torch.isfinite(loss):
                skipped_nonfinite_steps += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.gradient_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and sched_name == "onecycle":
                scheduler.step()

            w = float(b.y_mask.sum().item())
            train_loss_acc += float(loss.item()) * w
            train_w_acc += w

        epoch_sec = time.time() - e0
        train_loss = train_loss_acc / max(train_w_acc, 1e-6)
        val_loss, val_r2 = _eval(model, valid_loader, device)

        history = {
            "epoch": epoch,
            "batch_size": int(cfg.batch_size),
            "tolerance_days": int(cfg.tolerance_days),
            "model": str(cfg.model_log_name),
            "epoch_sec": float(epoch_sec),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_r2": float(val_r2),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(history)
        print(
            f"[epoch-timer][mtspect] tol={cfg.tolerance_days} model={cfg.model_log_name} bs={cfg.batch_size} epoch={epoch}/{cfg.epochs} sec={epoch_sec:.3f} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_r2={val_r2:.4f}"
        )
        wandb_handle.log(history)

        if val_loss < (best_val - cfg.early_stopping_min_delta):
            best_val = val_loss
            best_val_r2 = val_r2
            best_epoch = epoch
            if bool(cfg.restore_best_state):
                torch.save(model.state_dict(), best_ckpt_tmp)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.early_stopping_patience:
            break
        if scheduler is not None and sched_name != "onecycle":
            scheduler.step()

    if bool(cfg.restore_best_state) and best_ckpt_tmp.exists():
        model.load_state_dict(torch.load(best_ckpt_tmp, map_location=device))
        best_ckpt_tmp.unlink(missing_ok=True)

    infer_t0 = time.time()
    y_true, y_mask, y_pred, y_logvar = _predict(model, test_loader, device)
    infer_sec = time.time() - infer_t0
    infer_ms_per_sample = 1000.0 * infer_sec / max(len(y_true), 1)

    if norm_stats is not None and cfg.normalize_targets:
        y_pred, y_logvar = invert_target_scaling(y_pred, y_logvar, norm_stats)
        y_true, _ = invert_target_scaling(y_true, y_logvar * 0.0, norm_stats)

    metric_rows, metric_summary = regression_metrics(y_true, y_pred, y_mask, train_ds.targets)
    bootstrap_stats = bootstrap_metric_stats(y_true, y_pred, y_logvar, y_mask, train_ds.targets, n_bootstrap=1000, seed=42)
    unc_report = {
        "nll": gaussian_nll(y_true, y_pred, y_logvar, y_mask),
        "crps": gaussian_crps(y_true, y_pred, y_logvar, y_mask),
        **prediction_interval_metrics(y_true, y_pred, y_logvar, y_mask),
        "ece_regression": regression_ece(y_true, y_pred, y_logvar, y_mask),
    }

    run_ended = time.time()
    artifacts_root = ensure_dir(root / "artifacts")
    run_name = f"mtspect_tol{cfg.tolerance_days}_bs{cfg.batch_size}"
    result_dir = ensure_dir(artifacts_root / f"tolerance_{cfg.tolerance_days}d" / "results" / "mtspect" / run_name)

    ckpt_path = result_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_csv = result_dir / "metrics_mtspect.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    history_df = pd.DataFrame(history_rows)
    history_csv = result_dir / "history_mtspect.csv"
    history_df.to_csv(history_csv, index=False)

    curve_dir = ensure_dir(result_dir / "curves")
    curve_paths = plot_training_curves(history_csv, curve_dir)

    uncertainty_json = result_dir / "uncertainty_mtspect.json"
    uncertainty_json.write_text(json.dumps(unc_report, indent=2), encoding="utf-8")

    summary = {
        # "family": "mtspect",
        "model": "mtspect",
        "tolerance_days": int(cfg.tolerance_days),
        "batch_size": int(cfg.batch_size),
        "run_started_at": float(run_started),
        "run_ended_at": float(run_ended),
        "run_duration_sec": float(run_ended - run_started),
        "inference_ms_per_sample": float(infer_ms_per_sample),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_val_r2": float(best_val_r2),
        "learning_rate": float(cfg.learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "hidden_dim": int(cfg.hidden_dim),
        "dropout": float(cfg.dropout),
        "optimizer": str(cfg.optimizer),
        "scheduler": str(cfg.scheduler),
        "early_stopping_patience": int(cfg.early_stopping_patience),
        "early_stopping_min_delta": float(cfg.early_stopping_min_delta),
        "lambda_h": float(cfg.lambda_h),
        "lambda_a": float(cfg.lambda_a),
        "lambda_r": float(cfg.lambda_r),
        "lambda_p": float(cfg.lambda_p),
        "aux_warmup_epochs": int(cfg.aux_warmup_epochs),
        "max_text_len": int(cfg.max_text_len),
        "gradient_clip_norm": float(cfg.gradient_clip_norm),
        "prefit_l2": float(cfg.prefit_l2),
        "stability_mode": bool(cfg.stability_mode),
        "use_amp": bool(cfg.use_amp),
        "skipped_nonfinite_steps": int(skipped_nonfinite_steps),
        "text_lr_scale": float(cfg.text_lr_scale),
        "vision_lr_scale": float(cfg.vision_lr_scale),
        "target_log1p_mask": [False] * len(active_targets),
        "text_model_id": str(cfg.text_model_id),
        "text_model_trainable": bool(cfg.text_model_trainable),
        "text_use_lora": bool(cfg.text_use_lora),
        "text_use_qlora": bool(cfg.text_use_qlora),
        "lora_r": int(cfg.lora_r),
        "lora_alpha": int(cfg.lora_alpha),
        "lora_dropout": float(cfg.lora_dropout),
        "use_vlm_bridge": bool(cfg.use_vlm_bridge),
        "vision_rgb_model": str(cfg.vision_rgb_model),
        "vision_rgb_model_id": str(cfg.vision_rgb_model_id),
        "vision_ms_model": str(cfg.vision_ms_model),
        "require_all_modalities": bool(cfg.require_all_modalities),
        "min_multimodal_rows": int(cfg.min_multimodal_rows),
        **repro,
        **modality_coverage,
        **profile_info,
        "checkpoint_path": str(ckpt_path),
        "metrics_csv": str(metrics_csv),
        "history_csv": str(history_csv),
        "curve_paths": [str(p) for p in curve_paths],
        **metric_summary,
        **unc_report,
        **bootstrap_stats,
        "status": "ok",
    }
    summary_json = result_dir / "summary_mtspect.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    prepared_dir = ensure_dir(artifacts_root / "prepared")
    agg_csv = prepared_dir / "mtspect_tolerance_validation.csv"
    agg_json = prepared_dir / "mtspect_tolerance_validation.json"

    row_df = pd.DataFrame([summary])
    if agg_csv.exists():
        old = pd.read_csv(agg_csv)
        old = old[~((old.get("tolerance_days") == cfg.tolerance_days) & (old.get("batch_size") == cfg.batch_size))]
        out = pd.concat([old, row_df], axis=0, ignore_index=True)
    else:
        out = row_df
    out.to_csv(agg_csv, index=False)
    agg_json.write_text(json.dumps(out.to_dict(orient="records"), indent=2), encoding="utf-8")

    compat_row = {
        "phase": "G",
        "family": "mtspect",
        "model": "mtspect++",
        "tolerance_days": int(cfg.tolerance_days),
        "batch_size": int(cfg.batch_size),
        "macro_rmse": metric_summary.get("mean_rmse"),
        "macro_mae": metric_summary.get("mean_mae"),
        "macro_r2": metric_summary.get("mean_r2"),
        "inference_ms_per_sample": float(infer_ms_per_sample),
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
    }
    compat_csv = prepared_dir / "mtspect_metric_suite_summary.csv"
    compat_json = prepared_dir / "mtspect_metric_suite_summary.json"
    compat_df = pd.DataFrame([compat_row])
    if compat_csv.exists():
        oldc = pd.read_csv(compat_csv)
        oldc = oldc[~((oldc.get("tolerance_days") == cfg.tolerance_days) & (oldc.get("batch_size") == cfg.batch_size))]
        compat_out = pd.concat([oldc, compat_df], axis=0, ignore_index=True)
    else:
        compat_out = compat_df
    compat_out.to_csv(compat_csv, index=False)
    compat_json.write_text(json.dumps(compat_out.to_dict(orient="records"), indent=2), encoding="utf-8")

    wandb_handle.log(
        {
            "final_mean_rmse": metric_summary.get("mean_rmse"),
            "final_mean_r2": metric_summary.get("mean_r2"),
            "final_nll": unc_report.get("nll"),
            "final_ece": unc_report.get("ece_regression"),
            "final_status": "ok",
        }
    )
    wandb_handle.finish()

    return summary


def predict_mtspect(root: Path, cfg: TrainConfig) -> pd.DataFrame:
    """Run inference on the test split and return a coordinates-linked long-form DataFrame.

    Returns one row per (sample × target) pair with columns:
      lat, lon, station_id, lake_id, municipality, target, y_true, y_pred, y_std

    Rows where y_mask == 0 (target not measured for that sample) are dropped.
    """
    root = root.resolve()
    prep_dir = root / "artifacts" / f"tolerance_{cfg.tolerance_days}d" / "prepared"

    _, _, test_df, _ = load_multimodal_split_frames(
        prep_dir,
        require_all_modalities=bool(cfg.require_all_modalities),
        min_rows=1,
    )

    # We need norm_stats fitted on training data — reload train split for that purpose
    train_df, _, _, _ = load_multimodal_split_frames(
        prep_dir,
        require_all_modalities=bool(cfg.require_all_modalities),
        min_rows=1,
    )

    active_targets = cfg.targets or TARGETS
    feature_spec = infer_feature_spec(train_df, targets=active_targets)
    tok = HFPromptTokenizer(model_id=cfg.text_model_id, max_len=cfg.max_text_len)

    train_ds = MTSpectDataset(train_df, feature_spec=feature_spec, tokenizer=tok, max_text_len=cfg.max_text_len, targets=active_targets)
    test_ds = MTSpectDataset(test_df, feature_spec=feature_spec, tokenizer=tok, max_text_len=cfg.max_text_len, targets=active_targets)

    norm_stats = None
    if cfg.normalize_inputs or cfg.normalize_targets:
        norm_stats = fit_normalization_stats(train_ds)
        train_ds.apply_normalization(norm_stats, normalize_targets=cfg.normalize_targets)
        test_ds.apply_normalization(norm_stats, normalize_targets=cfg.normalize_targets)

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = MTSpectConfig(
        tabular_dim=len(feature_spec.tabular_columns),
        temporal_band_dim=len(feature_spec.temporal_band_columns),
        quality_dim=max(1, len(feature_spec.quality_columns)),
        n_targets=len(test_ds.targets),
        vocab_size=0,
        text_model_id=cfg.text_model_id,
        text_model_trainable=cfg.text_model_trainable,
        text_use_lora=cfg.text_use_lora,
        text_use_qlora=cfg.text_use_qlora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        text_max_len=cfg.max_text_len,
        hidden_dim=cfg.hidden_dim,
        text_dim=cfg.text_dim,
        temporal_heads=cfg.temporal_heads,
        temporal_layers=cfg.temporal_layers,
        dropout=cfg.dropout,
        covariance_rank=cfg.covariance_rank,
        vision_rgb_model=cfg.vision_rgb_model,
        vision_rgb_model_id=cfg.vision_rgb_model_id,
        vision_ms_model=cfg.vision_ms_model,
        npz_channels=len(DEFAULT_NPZ_BAND_ORDER),
        use_vlm_bridge=cfg.use_vlm_bridge,
        enable_modality_tabular=cfg.enable_modality_tabular,
        enable_modality_temporal=cfg.enable_modality_temporal,
        enable_modality_rgb=cfg.enable_modality_rgb,
        enable_modality_npz=cfg.enable_modality_npz,
        enable_modality_text=cfg.enable_modality_text,
        enable_quality_gates=cfg.enable_quality_gates,
        enable_fusion_mode=cfg.enable_fusion_mode,
        enable_uncertainty_head=cfg.enable_uncertainty_head,
        enable_cross_attn_expert=cfg.enable_cross_attn_expert,
        enable_spectral_adapter=cfg.enable_spectral_adapter,
        enable_band_bridge=cfg.enable_band_bridge,
        enable_tabular_skip=cfg.enable_tabular_skip,
    )
    model = MTSpectModel(model_cfg).to(device)

    run_name = f"mtspect_tol{cfg.tolerance_days}_bs{cfg.batch_size}"
    ckpt_path = root / "artifacts" / f"tolerance_{cfg.tolerance_days}d" / "results" / "mtspect" / run_name / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_mask, y_pred, y_logvar = _predict(model, test_loader, device)

    if norm_stats is not None and cfg.normalize_targets:
        y_pred, y_logvar = invert_target_scaling(y_pred, y_logvar, norm_stats)
        y_true, _ = invert_target_scaling(y_true, y_logvar * 0.0, norm_stats)

    # Read coordinate and metadata columns from test.csv (row-aligned with test_df)
    test_csv_path = prep_dir / "test.csv"
    test_meta = pd.read_csv(test_csv_path)
    # Align to the same rows test_df uses (test_df may be a multimodal subset)
    meta_cols = ["lat", "lon", "station_id", "lake_id", "municipality"]
    available_meta = [c for c in meta_cols if c in test_meta.columns]
    # test_df index references rows from the original test.csv
    if hasattr(test_df, "index"):
        meta_aligned = test_meta.loc[test_df.index, available_meta].reset_index(drop=True)
    else:
        meta_aligned = test_meta[available_meta].iloc[: len(y_true)].reset_index(drop=True)

    # Build long-form DataFrame: one row per (sample × target)
    y_std = np.exp(0.5 * y_logvar)  # convert log-variance to std
    rows = []
    for i in range(len(y_true)):
        meta_row = meta_aligned.iloc[i] if i < len(meta_aligned) else {}
        for ti, target_name in enumerate(active_targets):
            if y_mask[i, ti] < 0.5:
                continue
            row: dict = {}
            for col in available_meta:
                row[col] = meta_row.get(col) if isinstance(meta_row, dict) else meta_row[col]
            row["target"] = target_name
            row["y_true"] = float(y_true[i, ti])
            row["y_pred"] = float(y_pred[i, ti])
            row["y_std"] = float(y_std[i, ti])
            rows.append(row)

    result_df = pd.DataFrame(rows)
    # Ensure column order matches spec
    ordered_cols = [c for c in ["lat", "lon", "station_id", "lake_id", "municipality", "target", "y_true", "y_pred", "y_std"] if c in result_df.columns]
    result_df = result_df[ordered_cols]
    return result_df
