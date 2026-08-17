from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from wq_pipeline.config import PipelineConfig
from wq_pipeline.data.io import ensure_dir, load_eo_features, load_in_situ
from wq_pipeline.data.matchup import create_matchups
from wq_pipeline.data.qc import apply_basic_qc
from wq_pipeline.features import select_feature_columns
from wq_pipeline.models.baseline import train_and_evaluate_per_target


def _resolve(root: Path, p: str) -> Path:
    return (root / p).resolve()


def _split_by_group(
    df: pd.DataFrame,
    group_column: str,
    test_size: float,
    valid_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[group_column].astype(str)
    splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train_valid, idx_test = next(splitter_1.split(df, groups=groups))

    train_valid = df.iloc[idx_train_valid].reset_index(drop=True)
    test = df.iloc[idx_test].reset_index(drop=True)

    groups_tv = train_valid[group_column].astype(str)
    adjusted_valid = valid_size / max(1e-8, (1.0 - test_size))
    adjusted_valid = min(max(adjusted_valid, 0.05), 0.5)

    splitter_2 = GroupShuffleSplit(n_splits=1, test_size=adjusted_valid, random_state=seed + 1)
    idx_train, idx_valid = next(splitter_2.split(train_valid, groups=groups_tv))

    train = train_valid.iloc[idx_train].reset_index(drop=True)
    valid = train_valid.iloc[idx_valid].reset_index(drop=True)
    return train, valid, test


def run_prepare(cfg: PipelineConfig, root: Path) -> None:
    if not cfg.paths:
        raise ValueError("`paths` section is required for prepare/train.")
    if not cfg.targets:
        raise ValueError("`targets` section is required for prepare/train.")

    artifacts = ensure_dir(_resolve(root, str(cfg.artifacts_dir)))

    in_situ = load_in_situ(_resolve(root, cfg.paths["in_situ_csv"]))
    eo = load_eo_features(_resolve(root, cfg.paths["eo_features_csv"]))

    matched = create_matchups(
        in_situ_df=in_situ,
        eo_df=eo,
        date_tolerance_days=int(cfg.matchup.get("date_tolerance_days", 3)),
        require_same_station=bool(cfg.matchup.get("require_same_station", True)),
    )

    qc = apply_basic_qc(
        matched,
        targets=cfg.targets,
        min_targets_per_row=int(cfg.matchup.get("min_targets_per_row", 1)),
    )

    train, valid, test = _split_by_group(
        qc,
        group_column=str(cfg.split.get("group_column", "lake_id")),
        test_size=float(cfg.split.get("test_size", 0.2)),
        valid_size=float(cfg.split.get("valid_size", 0.1)),
        seed=cfg.seed,
    )

    prep_dir = ensure_dir(artifacts / "prepared")
    train.to_csv(prep_dir / "train.csv", index=False)
    valid.to_csv(prep_dir / "valid.csv", index=False)
    test.to_csv(prep_dir / "test.csv", index=False)

    stats = {
        "n_total": int(len(qc)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "targets": cfg.targets,
        "target_non_null": {target: int(qc[target].notna().sum()) for target in cfg.targets if target in qc.columns},
    }
    (prep_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def run_train(cfg: PipelineConfig, root: Path) -> None:
    if not cfg.paths:
        raise ValueError("`paths` section is required for prepare/train.")
    if not cfg.targets:
        raise ValueError("`targets` section is required for prepare/train.")

    artifacts = ensure_dir(_resolve(root, str(cfg.artifacts_dir)))
    prep_dir = artifacts / "prepared"

    if not (prep_dir / "train.csv").exists() or not (prep_dir / "test.csv").exists():
        run_prepare(cfg, root)

    train = pd.read_csv(prep_dir / "train.csv")
    valid = pd.read_csv(prep_dir / "valid.csv")
    test = pd.read_csv(prep_dir / "test.csv")

    train_full = pd.concat([train, valid], axis=0, ignore_index=True)
    feature_columns = select_feature_columns(
        train_full,
        targets=cfg.targets,
        exclude_columns=list(cfg.features.get("exclude_columns", [])),
    )

    train_params = dict(cfg.training)
    train_params["seed"] = cfg.seed

    metrics_df, models = train_and_evaluate_per_target(
        train_df=train_full,
        test_df=test,
        feature_columns=feature_columns,
        targets=cfg.targets,
        train_params=train_params,
    )

    out_dir = ensure_dir(artifacts / "results")
    metrics_df = metrics_df.sort_values("target").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    summary = {
        "targets_trained": metrics_df["target"].tolist() if not metrics_df.empty else [],
        "mean_rmse": float(np.nanmean(metrics_df["rmse"])) if not metrics_df.empty else None,
        "mean_mae": float(np.nanmean(metrics_df["mae"])) if not metrics_df.empty else None,
        "mean_r2": float(np.nanmean(metrics_df["r2"])) if not metrics_df.empty else None,
        "n_features": len(feature_columns),
        "features": feature_columns,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_dir = ensure_dir(artifacts / "models")
    for target, model in models.items():
        joblib.dump(model, model_dir / f"baseline_{target}.joblib")
