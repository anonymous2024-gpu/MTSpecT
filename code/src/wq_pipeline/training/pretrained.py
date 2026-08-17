from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wq_pipeline.config import PipelineConfig
from wq_pipeline.data.io import ensure_dir
from wq_pipeline.features import select_feature_columns


def _require_torch() -> tuple:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for `train-pretrained`. Install it first, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu126"
        ) from exc
    return torch, nn, optim


def run_train_pretrained(cfg: PipelineConfig, root: Path) -> None:
    torch, nn, optim = _require_torch()

    artifacts = ensure_dir((root / str(cfg.artifacts_dir)).resolve())
    prep_dir = artifacts / "prepared"
    if not (prep_dir / "train.csv").exists() or not (prep_dir / "test.csv").exists():
        raise FileNotFoundError("Prepared splits not found. Run `prepare` first.")

    train = pd.read_csv(prep_dir / "train.csv")
    valid = pd.read_csv(prep_dir / "valid.csv")
    test = pd.read_csv(prep_dir / "test.csv")
    train_full = pd.concat([train, valid], axis=0, ignore_index=True)

    feature_columns = select_feature_columns(
        train_full,
        targets=cfg.targets,
        exclude_columns=list(cfg.features.get("exclude_columns", [])),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = cfg.pretrained
    hidden_dim = int(conf.get("hidden_dim", 256))
    dropout = float(conf.get("dropout", 0.1))
    epochs = int(conf.get("epochs", 50))
    batch_size = int(conf.get("batch_size", 256))
    early_stopping_patience = int(conf.get("early_stopping_patience", 15))
    early_stopping_min_delta = float(conf.get("early_stopping_min_delta", 1e-4))
    learning_rate = float(conf.get("learning_rate", 1e-3))
    weight_decay = float(conf.get("weight_decay", 1e-5))

    target_cols = [target for target in cfg.targets if target in train_full.columns]
    if not target_cols:
        raise ValueError("No target columns available for pretrained training.")

    x_train = train[feature_columns].to_numpy(dtype=np.float32)
    y_train = train[target_cols].to_numpy(dtype=np.float32)
    y_train_mask = np.isfinite(y_train).astype(np.float32)
    y_train = np.nan_to_num(y_train, nan=0.0)

    x_valid = valid[feature_columns].to_numpy(dtype=np.float32)
    y_valid = valid[target_cols].to_numpy(dtype=np.float32)
    y_valid_mask = np.isfinite(y_valid).astype(np.float32)
    y_valid = np.nan_to_num(y_valid, nan=0.0)

    if float(y_valid_mask.sum()) <= 0.0 and len(x_train) >= 20:
        rng = np.random.default_rng(int(cfg.seed) + 4321)
        idx = rng.permutation(len(x_train))
        n_val = max(1, int(0.1 * len(idx)))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        x_valid = x_train[val_idx]
        y_valid = y_train[val_idx]
        y_valid_mask = y_train_mask[val_idx]

        x_train = x_train[tr_idx]
        y_train = y_train[tr_idx]
        y_train_mask = y_train_mask[tr_idx]

    x_test = test[feature_columns].to_numpy(dtype=np.float32)
    y_test = test[target_cols].to_numpy(dtype=np.float32)
    y_test_mask = np.isfinite(y_test)

    finite_mask = np.isfinite(x_train)
    safe_train = np.where(finite_mask, x_train, 0.0)
    finite_count = np.maximum(finite_mask.sum(axis=0), 1)
    feat_mean = safe_train.sum(axis=0) / finite_count

    centered = np.where(finite_mask, x_train - feat_mean, 0.0)
    feat_var = (centered**2).sum(axis=0) / finite_count
    feat_std = np.sqrt(feat_var)

    feat_mean = np.nan_to_num(feat_mean, nan=0.0, posinf=0.0, neginf=0.0)
    feat_std = np.nan_to_num(feat_std, nan=1.0, posinf=1.0, neginf=1.0)
    feat_std[feat_std < 1e-6] = 1.0

    x_train = (x_train - feat_mean) / feat_std
    x_valid = (x_valid - feat_mean) / feat_std
    x_test = (x_test - feat_mean) / feat_std

    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_valid = np.nan_to_num(x_valid, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    x_train = x_train.astype(np.float32, copy=False)
    x_valid = x_valid.astype(np.float32, copy=False)
    x_test = x_test.astype(np.float32, copy=False)

    x_train_t = torch.tensor(x_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    y_mask_t = torch.tensor(y_train_mask, device=device)
    x_valid_t = torch.tensor(x_valid, device=device)
    y_valid_t = torch.tensor(y_valid, device=device)
    y_valid_mask_t = torch.tensor(y_valid_mask, device=device)

    class MultiTaskNet(nn.Module):
        def __init__(self, in_dim: int, hid_dim: int, out_dim: int, drop: float) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hid_dim, hid_dim),
                nn.GELU(),
                nn.Dropout(drop),
            )
            self.head = nn.Linear(hid_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            return self.head(z)

    model = MultiTaskNet(len(feature_columns), hidden_dim, len(target_cols), dropout).to(device)

    ckpt_path = conf.get("checkpoint_path")
    if ckpt_path:
        checkpoint = torch.load((root / ckpt_path).resolve(), map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n_train = x_train_t.shape[0]
    idx_all = np.arange(n_train)
    best_state = None
    best_valid_loss = float("inf")
    no_improve = 0
    epochs_trained = 0

    for epoch in range(epochs):
        np.random.shuffle(idx_all)
        model.train()
        for start in range(0, n_train, batch_size):
            idx = idx_all[start : start + batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]
            mb = y_mask_t[idx]

            pred = model(xb)
            sqe = (pred - yb) ** 2
            loss = (sqe * mb).sum() / (mb.sum() + 1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_valid_t)
            val_sqe = (val_pred - y_valid_t) ** 2
            val_loss = (val_sqe * y_valid_mask_t).sum() / (y_valid_mask_t.sum() + 1e-6)
            val_loss_value = float(val_loss.detach().cpu().item())

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
        pred_test = model(torch.tensor(x_test, device=device)).cpu().numpy()

    rows = []
    for target_index, target_name in enumerate(target_cols):
        mask = y_test_mask[:, target_index]
        if not np.any(mask):
            continue
        truth = y_test[mask, target_index]
        pred = pred_test[mask, target_index]
        finite_mask = np.isfinite(truth) & np.isfinite(pred)
        if not np.any(finite_mask):
            continue
        truth = truth[finite_mask]
        pred = pred[finite_mask]
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        mae = float(np.mean(np.abs(pred - truth)))
        denom = float(np.sum((truth - truth.mean()) ** 2)) + 1e-12
        r2 = float(1.0 - np.sum((pred - truth) ** 2) / denom)
        rows.append({"target": target_name, "rmse": rmse, "mae": mae, "r2": r2, "n_test": int(finite_mask.sum())})

    out_dir = ensure_dir(artifacts / "results")
    if rows:
        metrics = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    else:
        metrics = pd.DataFrame(columns=["target", "rmse", "mae", "r2", "n_test"])
    metrics.to_csv(out_dir / "metrics_pretrained.csv", index=False)

    summary = {
        "device": device,
        "targets": target_cols,
        "n_features": len(feature_columns),
        "hidden_dim": hidden_dim,
        "epochs_requested": epochs,
        "epochs_trained": int(epochs_trained),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "batch_size": int(batch_size),
        "mean_rmse": float(np.nanmean(metrics["rmse"])) if not metrics.empty else None,
        "mean_mae": float(np.nanmean(metrics["mae"])) if not metrics.empty else None,
        "mean_r2": float(np.nanmean(metrics["r2"])) if not metrics.empty else None,
    }
    (out_dir / "summary_pretrained.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_dir = ensure_dir(artifacts / "models")
    torch.save(model.state_dict(), model_dir / "pretrained_multitask.pt")
