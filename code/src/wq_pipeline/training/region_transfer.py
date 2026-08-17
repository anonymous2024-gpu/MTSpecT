from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wq_pipeline.config import PipelineConfig
from wq_pipeline.data.io import ensure_dir
from wq_pipeline.features import select_feature_columns
from wq_pipeline.training.pretrained import _require_torch


def _evaluate(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for target_index, target_name in enumerate(targets):
        target_mask = mask[:, target_index]
        if not np.any(target_mask):
            continue
        y_true = truth[target_mask, target_index]
        y_pred = pred[target_mask, target_index]
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        denom = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
        r2 = float(1.0 - np.sum((y_pred - y_true) ** 2) / denom)
        rows.append({"target": target_name, "rmse": rmse, "mae": mae, "r2": r2, "n_test": int(target_mask.sum())})
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def run_region_transfer(cfg: PipelineConfig, root: Path) -> None:
    torch, nn, optim = _require_torch()

    artifacts = ensure_dir((root / str(cfg.artifacts_dir)).resolve())
    prep_dir = artifacts / "prepared"
    if not (prep_dir / "train.csv").exists() or not (prep_dir / "test.csv").exists():
        raise FileNotFoundError("Prepared splits not found. Run `prepare` first.")

    train = pd.read_csv(prep_dir / "train.csv")
    valid = pd.read_csv(prep_dir / "valid.csv")
    test = pd.read_csv(prep_dir / "test.csv")
    train_full = pd.concat([train, valid], axis=0, ignore_index=True)

    rtc = dict(cfg.raw.get("region_transfer", {}))
    region_column = str(rtc.get("region_column", "lake_id"))
    target_region = rtc.get("target_region")
    if target_region is None:
        raise ValueError("Missing `region_transfer.target_region` in config.")

    if region_column not in train_full.columns or region_column not in test.columns:
        raise ValueError(f"Region column `{region_column}` not found in prepared data.")

    source_train = train_full[train_full[region_column].astype(str) != str(target_region)].copy()
    target_train = train_full[train_full[region_column].astype(str) == str(target_region)].copy()
    target_test = test[test[region_column].astype(str) == str(target_region)].copy()

    if source_train.empty or target_test.empty:
        raise ValueError("Insufficient rows for region transfer experiment. Check target_region and splits.")

    feature_columns = select_feature_columns(
        train_full,
        targets=cfg.targets,
        exclude_columns=list(cfg.features.get("exclude_columns", [])),
    )
    target_cols = [target for target in cfg.targets if target in train_full.columns]
    if not target_cols:
        raise ValueError("No target columns available for region transfer.")

    conf = cfg.pretrained
    hidden_dim = int(conf.get("hidden_dim", 256))
    dropout = float(conf.get("dropout", 0.1))
    pretrain_epochs = int(rtc.get("pretrain_epochs", conf.get("epochs", 40)))
    finetune_epochs = int(rtc.get("finetune_epochs", 20))
    batch_size = int(conf.get("batch_size", 256))
    learning_rate = float(conf.get("learning_rate", 1e-3))
    finetune_lr = float(rtc.get("finetune_learning_rate", 1e-4))
    weight_decay = float(conf.get("weight_decay", 1e-5))

    def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = df[feature_columns].to_numpy(dtype=np.float32)
        y = df[target_cols].to_numpy(dtype=np.float32)
        mask = np.isfinite(y).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0)
        return x, y, mask

    x_source, y_source, m_source = to_xy(source_train)
    x_target_train, y_target_train, m_target_train = to_xy(target_train)
    x_target_test, y_target_test, m_target_test = to_xy(target_test)

    feat_mean = np.nanmean(x_source, axis=0)
    feat_std = np.nanstd(x_source, axis=0) + 1e-6

    x_source = (x_source - feat_mean) / feat_std
    x_target_train = (x_target_train - feat_mean) / feat_std
    x_target_test = (x_target_test - feat_mean) / feat_std

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

        def forward(self, x):
            return self.head(self.encoder(x))

    model = MultiTaskNet(len(feature_columns), hidden_dim, len(target_cols), dropout).to(device)

    x_source_t = torch.tensor(x_source, device=device)
    y_source_t = torch.tensor(y_source, device=device)
    m_source_t = torch.tensor(m_source, device=device)

    opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    idx_source = np.arange(x_source_t.shape[0])

    for _ in range(pretrain_epochs):
        np.random.shuffle(idx_source)
        model.train()
        for start in range(0, len(idx_source), batch_size):
            idx = idx_source[start : start + batch_size]
            pred = model(x_source_t[idx])
            loss = (((pred - y_source_t[idx]) ** 2) * m_source_t[idx]).sum() / (m_source_t[idx].sum() + 1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        zero_shot = model(torch.tensor(x_target_test, device=device)).cpu().numpy()
    zero_metrics = _evaluate(zero_shot, y_target_test, m_target_test.astype(bool), target_cols)

    if not target_train.empty:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

        x_target_train_t = torch.tensor(x_target_train, device=device)
        y_target_train_t = torch.tensor(y_target_train, device=device)
        m_target_train_t = torch.tensor(m_target_train, device=device)
        idx_target = np.arange(x_target_train_t.shape[0])

        head_opt = optim.AdamW(model.head.parameters(), lr=finetune_lr, weight_decay=weight_decay)
        for _ in range(finetune_epochs):
            np.random.shuffle(idx_target)
            model.train()
            for start in range(0, len(idx_target), batch_size):
                idx = idx_target[start : start + batch_size]
                pred = model(x_target_train_t[idx])
                loss = (((pred - y_target_train_t[idx]) ** 2) * m_target_train_t[idx]).sum() / (m_target_train_t[idx].sum() + 1e-6)
                head_opt.zero_grad(set_to_none=True)
                loss.backward()
                head_opt.step()

    model.eval()
    with torch.no_grad():
        finetuned = model(torch.tensor(x_target_test, device=device)).cpu().numpy()
    finetuned_metrics = _evaluate(finetuned, y_target_test, m_target_test.astype(bool), target_cols)

    out_dir = ensure_dir(artifacts / "region_transfer")
    zero_metrics.to_csv(out_dir / "metrics_zero_shot.csv", index=False)
    finetuned_metrics.to_csv(out_dir / "metrics_head_finetune.csv", index=False)

    summary = {
        "region_column": region_column,
        "target_region": str(target_region),
        "n_source_train": int(len(source_train)),
        "n_target_train": int(len(target_train)),
        "n_target_test": int(len(target_test)),
        "targets": target_cols,
        "device": device,
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
