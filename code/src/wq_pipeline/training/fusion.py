from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wq_pipeline.config import PipelineConfig
from wq_pipeline.data.io import ensure_dir
from wq_pipeline.features import select_feature_columns
from wq_pipeline.models.fusion import build_fusion_model, require_torch, resolve_profile


def _set_seed(seed: int, torch) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_temporal_tensor(
    df: pd.DataFrame,
    temporal_steps: int,
    base_features: list[str],
    required_dim: int | None = None,
) -> np.ndarray:
    step_to_cols: dict[int, list[str]] = {}
    for col in df.columns:
        if not col.startswith("img_t"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        suffix = col[5:]
        step_text, _, _ = suffix.partition("_")
        if step_text.isdigit():
            step = int(step_text)
            if 0 <= step < temporal_steps:
                step_to_cols.setdefault(step, []).append(col)

    if step_to_cols:
        common_dim = int(required_dim) if required_dim is not None else min(len(cols) for cols in step_to_cols.values())
        if common_dim <= 0:
            step_to_cols = {}
        else:
            ordered_steps = list(range(temporal_steps))
            fallback_df = df[base_features].apply(pd.to_numeric, errors="coerce")
            fallback = np.nan_to_num(fallback_df.to_numpy(dtype=np.float32), nan=0.0)
            fallback = fallback[:, :common_dim] if fallback.shape[1] >= common_dim else np.pad(
                fallback,
                ((0, 0), (0, max(0, common_dim - fallback.shape[1]))),
                mode="constant",
            )
            tokens: list[np.ndarray] = []
            for step in ordered_steps:
                cols = sorted(step_to_cols.get(step, []))
                if len(cols) > 0:
                    token_raw = np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0)
                    if token_raw.shape[1] >= common_dim:
                        token = token_raw[:, :common_dim]
                    else:
                        token = np.pad(token_raw, ((0, 0), (0, common_dim - token_raw.shape[1])), mode="constant")
                else:
                    token = fallback.copy()
                tokens.append(token)
            return np.stack(tokens, axis=1)

    fallback_df = df[base_features].apply(pd.to_numeric, errors="coerce")
    fallback = np.nan_to_num(fallback_df.to_numpy(dtype=np.float32), nan=0.0)
    if required_dim is not None:
        req = int(required_dim)
        if fallback.shape[1] >= req:
            fallback = fallback[:, :req]
        else:
            fallback = np.pad(fallback, ((0, 0), (0, req - fallback.shape[1])), mode="constant")
    repeat = np.repeat(fallback[:, None, :], temporal_steps, axis=1)
    return repeat


def _normalize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0) + 1e-6
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def _normalize_with_train(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0) + 1e-6
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def _normalize_temporal_with_train(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=(0, 1), keepdims=True)
    std = np.nanstd(train_x, axis=(0, 1), keepdims=True) + 1e-6
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return (train_x - mean) / std, (other_x - mean) / std, mean.squeeze(), std.squeeze()


def _normalize_temporal_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=(0, 1), keepdims=True)
    std = np.nanstd(train_x, axis=(0, 1), keepdims=True) + 1e-6
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return (train_x - mean) / std, (test_x - mean) / std, mean.squeeze(), std.squeeze()


def _evaluate_metrics(pred: np.ndarray, y_true: np.ndarray, y_mask: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for idx, target_name in enumerate(targets):
        mask = y_mask[:, idx]
        if not np.any(mask):
            continue
        truth = y_true[mask, idx]
        estimate = pred[mask, idx]
        finite_mask = np.isfinite(truth) & np.isfinite(estimate)
        if not np.any(finite_mask):
            continue
        truth = truth[finite_mask]
        estimate = estimate[finite_mask]
        rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
        mae = float(np.mean(np.abs(estimate - truth)))
        denom = float(np.sum((truth - truth.mean()) ** 2)) + 1e-12
        r2 = float(1.0 - np.sum((estimate - truth) ** 2) / denom)
        rows.append({"target": target_name, "rmse": rmse, "mae": mae, "r2": r2, "n_test": int(finite_mask.sum())})
    if not rows:
        return pd.DataFrame(columns=["target", "rmse", "mae", "r2", "n_test"])
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def run_train_fusion(cfg: PipelineConfig, root: Path) -> None:
    torch, _, optim = require_torch()

    artifacts = ensure_dir((root / str(cfg.artifacts_dir)).resolve())
    prep_dir = artifacts / "prepared"
    if not (prep_dir / "train.csv").exists() or not (prep_dir / "test.csv").exists():
        raise FileNotFoundError("Prepared splits not found. Run `prepare` first.")

    temporal_dir = prep_dir / "temporal"

    def _resolve_split_path(split_name: str) -> Path:
        preferred = temporal_dir / f"{split_name}_temporal.csv"
        legacy = prep_dir / f"{split_name}_temporal.csv"
        base = prep_dir / f"{split_name}.csv"
        if preferred.exists():
            return preferred
        if legacy.exists():
            return legacy
        return base

    train_path = _resolve_split_path("train")
    valid_path = _resolve_split_path("valid")
    test_path = _resolve_split_path("test")

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)
    train_full = pd.concat([train, valid], axis=0, ignore_index=True)

    fusion_cfg = dict(cfg.raw.get("fusion", {}))
    profile_name = str(fusion_cfg.get("profile", "small"))
    fusion_mode = str(fusion_cfg.get("fusion_mode", "cross_attention")).strip().lower()
    epoch_log_file_raw = str(fusion_cfg.get("epoch_log_file", "")).strip()
    epoch_log_context = str(fusion_cfg.get("epoch_log_context", "")).strip()
    epoch_log_file = Path(epoch_log_file_raw).resolve() if epoch_log_file_raw else None
    if epoch_log_file is not None:
        epoch_log_file.parent.mkdir(parents=True, exist_ok=True)
    profile = resolve_profile(profile_name)

    temporal_steps = int(fusion_cfg.get("temporal_steps", 5))
    heteroscedastic = bool(fusion_cfg.get("heteroscedastic", True))
    use_temporal_attention = bool(fusion_cfg.get("temporal_attention", True))

    checkpoint_rel = str(fusion_cfg.get("pretrained_checkpoint_path", "")).strip()
    checkpoint_path = (root / checkpoint_rel).resolve() if checkpoint_rel else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        checkpoint_path = None

    hidden_overrides = {
        "hidden_dim": int(fusion_cfg.get("hidden_dim", profile["hidden_dim"])),
        "tabular_hidden_dim": int(fusion_cfg.get("tabular_hidden_dim", profile["tabular_hidden_dim"])),
        "temporal_hidden_dim": int(fusion_cfg.get("temporal_hidden_dim", profile["temporal_hidden_dim"])),
        "num_heads": int(fusion_cfg.get("num_heads", profile["num_heads"])),
        "num_layers": int(fusion_cfg.get("num_layers", profile["num_layers"])),
        "dropout": float(fusion_cfg.get("dropout", profile["dropout"])),
        "batch_size": int(fusion_cfg.get("batch_size", profile["batch_size"])),
        "epochs": int(fusion_cfg.get("epochs", profile["epochs"])),
        "learning_rate": float(fusion_cfg.get("learning_rate", profile["learning_rate"])),
        "weight_decay": float(fusion_cfg.get("weight_decay", profile["weight_decay"])),
    }
    early_stopping_patience = int(fusion_cfg.get("early_stopping_patience", 15))
    early_stopping_min_delta = float(fusion_cfg.get("early_stopping_min_delta", 1e-4))

    _set_seed(cfg.seed, torch)

    target_cols = [target for target in cfg.targets if target in train_full.columns]
    if not target_cols:
        raise ValueError("No target columns found in prepared split for fusion training.")

    exclude_columns = list(cfg.features.get("exclude_columns", []))
    exclude_columns = sorted(set(exclude_columns + [c for c in train_full.columns if c.startswith("img_t")]))
    tabular_features = select_feature_columns(train_full, targets=cfg.targets, exclude_columns=exclude_columns)
    train_tab_df = train[tabular_features].apply(pd.to_numeric, errors="coerce")
    valid_tab_df = valid[tabular_features].apply(pd.to_numeric, errors="coerce")
    test_tab_df = test[tabular_features].apply(pd.to_numeric, errors="coerce")

    usable_features: list[str] = []
    for col in tabular_features:
        if train_tab_df[col].notna().any():
            usable_features.append(col)
    tabular_features = usable_features
    if not tabular_features:
        raise ValueError("No usable tabular features for fusion training after removing all-NaN columns.")

    train_tab_df = train_tab_df[tabular_features]
    valid_tab_df = valid_tab_df[tabular_features]
    test_tab_df = test_tab_df[tabular_features]

    x_train_tab = np.nan_to_num(train_tab_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_valid_tab = np.nan_to_num(valid_tab_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_test_tab = np.nan_to_num(test_tab_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    x_train_temp = np.nan_to_num(
        _prepare_temporal_tensor(train, temporal_steps=temporal_steps, base_features=tabular_features),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    temporal_dim = int(x_train_temp.shape[2])
    x_valid_temp = np.nan_to_num(
        _prepare_temporal_tensor(valid, temporal_steps=temporal_steps, base_features=tabular_features, required_dim=temporal_dim),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    x_test_temp = np.nan_to_num(
        _prepare_temporal_tensor(test, temporal_steps=temporal_steps, base_features=tabular_features, required_dim=temporal_dim),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if not use_temporal_attention:
        x_train_temp = x_train_temp[:, :1, :]
        x_valid_temp = x_valid_temp[:, :1, :]
        x_test_temp = x_test_temp[:, :1, :]

    y_train = train[target_cols].to_numpy(dtype=np.float32)
    y_valid = valid[target_cols].to_numpy(dtype=np.float32)
    y_test = test[target_cols].to_numpy(dtype=np.float32)
    y_train_mask = np.isfinite(y_train).astype(np.float32)
    y_valid_mask = np.isfinite(y_valid).astype(np.float32)
    y_test_mask = np.isfinite(y_test)
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_valid = np.nan_to_num(y_valid, nan=0.0)

    if float(y_valid_mask.sum()) <= 0.0 and len(y_train) >= 20:
        rng = np.random.default_rng(int(cfg.seed) + 1234)
        idx = rng.permutation(len(y_train))
        n_val = max(1, int(0.1 * len(idx)))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        x_valid_tab = x_train_tab[val_idx]
        x_valid_temp = x_train_temp[val_idx]
        y_valid = y_train[val_idx]
        y_valid_mask = y_train_mask[val_idx]

        x_train_tab = x_train_tab[tr_idx]
        x_train_temp = x_train_temp[tr_idx]
        y_train = y_train[tr_idx]
        y_train_mask = y_train_mask[tr_idx]

    x_train_tab, x_valid_tab, tab_mean, tab_std = _normalize_with_train(x_train_tab, x_valid_tab)
    x_test_tab = (x_test_tab - tab_mean) / tab_std
    x_train_temp, x_valid_temp, temp_mean, temp_std = _normalize_temporal_with_train(x_train_temp, x_valid_temp)
    x_test_temp = (x_test_temp - temp_mean.reshape(1, 1, -1)) / temp_std.reshape(1, 1, -1)
    x_train_tab = np.nan_to_num(x_train_tab, nan=0.0, posinf=0.0, neginf=0.0)
    x_valid_tab = np.nan_to_num(x_valid_tab, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_tab = np.nan_to_num(x_test_tab, nan=0.0, posinf=0.0, neginf=0.0)
    x_train_temp = np.nan_to_num(x_train_temp, nan=0.0, posinf=0.0, neginf=0.0)
    x_valid_temp = np.nan_to_num(x_valid_temp, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_temp = np.nan_to_num(x_test_temp, nan=0.0, posinf=0.0, neginf=0.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            _ = torch.zeros(1, device="cuda")
        except Exception:
            device = "cpu"

    model = build_fusion_model(
        tabular_dim=x_train_tab.shape[1],
        temporal_input_dim=x_train_temp.shape[2],
        target_dim=len(target_cols),
        temporal_steps=x_train_temp.shape[1],
        profile=hidden_overrides,
        fusion_mode=fusion_mode,
        pretrained_checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hidden_overrides["learning_rate"]),
        weight_decay=float(hidden_overrides["weight_decay"]),
    )

    train_tab_t = torch.tensor(x_train_tab, device=device)
    train_temp_t = torch.tensor(x_train_temp, device=device)
    train_y_t = torch.tensor(y_train, device=device)
    train_mask_t = torch.tensor(y_train_mask, device=device)
    valid_tab_t = torch.tensor(x_valid_tab, device=device)
    valid_temp_t = torch.tensor(x_valid_temp, device=device)
    valid_y_t = torch.tensor(y_valid, device=device)
    valid_mask_t = torch.tensor(y_valid_mask, device=device)

    n_train = train_tab_t.shape[0]
    batch_size = int(hidden_overrides["batch_size"])
    epochs = int(hidden_overrides["epochs"])
    idx_all = np.arange(n_train)
    best_state = None
    best_valid_loss = float("inf")
    no_improve = 0
    epochs_trained = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        np.random.shuffle(idx_all)
        model.train()
        train_losses: list[float] = []
        for start in range(0, n_train, batch_size):
            idx = idx_all[start : start + batch_size]
            tab_batch = train_tab_t[idx]
            temp_batch = train_temp_t[idx]
            y_batch = train_y_t[idx]
            mask_batch = train_mask_t[idx]

            mean_pred, logvar_pred = model(tab_batch, temp_batch)
            if heteroscedastic:
                logvar_pred = torch.clamp(logvar_pred, min=-6.0, max=6.0)
                precision = torch.exp(-logvar_pred)
                per_target = 0.5 * (precision * ((mean_pred - y_batch) ** 2) + logvar_pred)
            else:
                per_target = (mean_pred - y_batch) ** 2

            loss = (per_target * mask_batch).sum() / (mask_batch.sum() + 1e-6)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_mean, val_logvar = model(valid_tab_t, valid_temp_t)
            if heteroscedastic:
                val_logvar = torch.clamp(val_logvar, min=-6.0, max=6.0)
                precision = torch.exp(-val_logvar)
                val_per_target = 0.5 * (precision * ((val_mean - valid_y_t) ** 2) + val_logvar)
            else:
                val_per_target = (val_mean - valid_y_t) ** 2
            val_loss = (val_per_target * valid_mask_t).sum() / (valid_mask_t.sum() + 1e-6)
            val_loss_value = float(val_loss.detach().cpu().item())

        train_loss_value = float(np.mean(train_losses)) if train_losses else float("inf")
        epoch_sec = float(time.perf_counter() - epoch_start)
        prefix = f"{epoch_log_context} " if epoch_log_context else ""
        line = (
            f"[epoch-timer][fusion_family] {prefix}"
            f"epoch={epoch + 1}/{int(epochs)} sec={epoch_sec:.3f} "
            f"train_loss={train_loss_value:.6f} valid_loss={val_loss_value:.6f}"
        )
        print(line, flush=True)
        if epoch_log_file is not None:
            with epoch_log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        epochs_trained = epoch + 1
        if val_loss_value < (best_valid_loss - early_stopping_min_delta):
            best_valid_loss = val_loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_tab_t = torch.tensor(x_test_tab, device=device)
        test_temp_t = torch.tensor(x_test_temp, device=device)
        mean_test, logvar_test = model(test_tab_t, test_temp_t)
        pred_test = np.nan_to_num(mean_test.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        pred_logvar = np.nan_to_num(np.clip(logvar_test.cpu().numpy(), -6.0, 6.0), nan=0.0, posinf=0.0, neginf=0.0)

    metrics = _evaluate_metrics(pred_test, y_test, y_test_mask, target_cols)

    out_dir = ensure_dir(artifacts / "results")
    metrics.to_csv(out_dir / "metrics_fusion.csv", index=False)

    rmse_vals = pd.to_numeric(metrics["rmse"], errors="coerce") if not metrics.empty else pd.Series(dtype=float)
    mae_vals = pd.to_numeric(metrics["mae"], errors="coerce") if not metrics.empty else pd.Series(dtype=float)
    r2_vals = pd.to_numeric(metrics["r2"], errors="coerce") if not metrics.empty else pd.Series(dtype=float)

    summary = {
        "device": device,
        "targets": target_cols,
        "n_features_tabular": int(x_train_tab.shape[1]),
        "n_features_temporal": int(x_train_temp.shape[2]),
        "temporal_steps": int(x_train_temp.shape[1]),
        "profile": profile_name,
        "fusion_mode": fusion_mode,
        "heteroscedastic": heteroscedastic,
        "temporal_attention": use_temporal_attention,
        "epochs_requested": epochs,
        "epochs_trained": int(epochs_trained),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "batch_size": batch_size,
        "mean_rmse": float(rmse_vals.mean()) if not rmse_vals.empty else None,
        "mean_mae": float(mae_vals.mean()) if not mae_vals.empty else None,
        "mean_r2": float(r2_vals.mean()) if not r2_vals.empty else None,
        "mean_predictive_variance": float(np.exp(pred_logvar).mean()),
        "tabular_normalization": {
            "mean_shape": list(np.asarray(tab_mean).shape),
            "std_shape": list(np.asarray(tab_std).shape),
        },
        "temporal_normalization": {
            "mean_shape": list(np.asarray(temp_mean).shape),
            "std_shape": list(np.asarray(temp_std).shape),
        },
    }
    (out_dir / "summary_fusion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_dir = ensure_dir(artifacts / "models")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "targets": target_cols,
            "tabular_features": tabular_features,
            "temporal_steps": int(x_train_temp.shape[1]),
            "profile": profile_name,
            "fusion_mode": fusion_mode,
        },
        model_dir / "fusion_multimodal.pt",
    )
