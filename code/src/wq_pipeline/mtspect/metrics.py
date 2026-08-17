from __future__ import annotations

import math

import numpy as np


def _safe_masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr) & mask
    return arr[valid]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray, target_names: list[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    var_eps = 1e-8
    for idx, target in enumerate(target_names):
        mask = y_mask[:, idx].astype(bool)
        truth = _safe_masked(y_true[:, idx], mask)
        pred = _safe_masked(y_pred[:, idx], mask)
        n = min(len(truth), len(pred))
        if n == 0:
            continue
        truth = truth[:n]
        pred = pred[:n]
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        mae = float(np.mean(np.abs(pred - truth)))
        denom = float(np.sum((truth - truth.mean()) ** 2))
        r2 = float("nan") if denom <= var_eps else float(1.0 - np.sum((pred - truth) ** 2) / denom)
        mape = float(np.mean(np.abs((pred - truth) / np.clip(np.abs(truth), 1e-6, None))))
        rows.append({"target": target, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "n_test": int(n)})

    if not rows:
        return rows, {"mean_rmse": None, "mean_mae": None, "mean_r2": None, "mean_mape": None}

    summary = {
        "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
        "mean_mae": float(np.mean([r["mae"] for r in rows])),
        "mean_r2": float(np.nanmean([r["r2"] for r in rows])) if np.isfinite(np.nanmean([r["r2"] for r in rows])) else float("nan"),
        "mean_mape": float(np.mean([r["mape"] for r in rows])),
    }
    return rows, summary


def bootstrap_metric_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_logvar: np.ndarray,
    y_mask: np.ndarray,
    target_names: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Return mean ± std strings for regression and uncertainty metrics via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    reg_keys = ["rmse", "mae", "r2", "mape"]
    unc_keys = ["nll", "crps", "picp", "mpiw", "ece_regression"]
    accum: dict[str, list[float]] = {k: [] for k in reg_keys + unc_keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ylv, ym = y_true[idx], y_pred[idx], y_logvar[idx], y_mask[idx]

        _, bsum = regression_metrics(yt, yp, ym, target_names)
        for k in reg_keys:
            v = bsum.get(f"mean_{k}")
            if v is not None and np.isfinite(v):
                accum[k].append(float(v))

        flat_mask = ym.astype(bool).any(axis=1)
        for fn, key in [
            (lambda: gaussian_nll(yt, yp, ylv, ym), "nll"),
            (lambda: gaussian_crps(yt, yp, ylv, ym), "crps"),
            (lambda: regression_ece(yt, yp, ylv, ym), "ece_regression"),
        ]:
            v = fn()
            if np.isfinite(v):
                accum[key].append(float(v))
        pi = prediction_interval_metrics(yt, yp, ylv, ym)
        for k in ["picp", "mpiw"]:
            v = pi[k]
            if np.isfinite(v):
                accum[k].append(float(v))

    result: dict[str, str] = {}
    for k in reg_keys + unc_keys:
        vals = np.array(accum[k])
        if vals.size == 0:
            result[k] = "nan ± nan"
        else:
            result[k] = f"{vals.mean():.4f} ± {vals.std():.4f}"
    return result


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, logvar: np.ndarray, y_mask: np.ndarray) -> float:
    valid = y_mask.astype(bool)
    yt = y_true[valid]
    mp = mu[valid]
    lv = logvar[valid]
    if yt.size == 0:
        return float("nan")
    return float(np.mean(0.5 * np.exp(-lv) * (yt - mp) ** 2 + 0.5 * lv))


def gaussian_crps(y_true: np.ndarray, mu: np.ndarray, logvar: np.ndarray, y_mask: np.ndarray) -> float:
    valid = y_mask.astype(bool)
    yt = y_true[valid]
    mp = mu[valid]
    lv = logvar[valid]
    if yt.size == 0:
        return float("nan")

    sigma = np.sqrt(np.exp(lv))
    z = (yt - mp) / np.clip(sigma, 1e-6, None)
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def prediction_interval_metrics(y_true: np.ndarray, mu: np.ndarray, logvar: np.ndarray, y_mask: np.ndarray, z_score: float = 1.96) -> dict:
    valid = y_mask.astype(bool)
    yt = y_true[valid]
    mp = mu[valid]
    lv = logvar[valid]
    if yt.size == 0:
        return {"picp": float("nan"), "mpiw": float("nan")}
    sigma = np.sqrt(np.exp(lv))
    lo = mp - z_score * sigma
    hi = mp + z_score * sigma
    covered = ((yt >= lo) & (yt <= hi)).astype(np.float32)
    return {"picp": float(np.mean(covered)), "mpiw": float(np.mean(hi - lo))}


def regression_ece(y_true: np.ndarray, mu: np.ndarray, logvar: np.ndarray, y_mask: np.ndarray, bins: int = 10) -> float:
    valid = y_mask.astype(bool)
    yt = y_true[valid]
    mp = mu[valid]
    lv = logvar[valid]
    if yt.size == 0:
        return float("nan")
    sigma = np.sqrt(np.exp(lv))
    conf = 1.0 / (1.0 + sigma)
    abs_err = np.abs(yt - mp)
    quantiles = np.quantile(conf, np.linspace(0.0, 1.0, bins + 1))
    ece = 0.0
    total = len(conf)
    for i in range(bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        if i == bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(1.0 / (1.0 + abs_err[mask].mean()))
        ece += (n / total) * abs(avg_acc - avg_conf)
    return float(ece)
