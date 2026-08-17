"""
Tolerance-Aware Multimodal Water-Quality Benchmark — benchmark builder.

Self-contained reproduction of the *data* pipeline: it matches in-situ
measurements to Sentinel-2 acquisitions within a date tolerance, applies quality
control, produces lake-grouped train/validation/test splits, and emits every
dataset table reported in the paper (benchmark scale, per-target observation
rates, and the conductivity distribution-shift analysis).

Only pandas, numpy, and scikit-learn are required. Multimodal image/temporal
extraction and model training are documented separately in the README; they are
not needed to reproduce the tables below.

Usage
-----
    python build_benchmark.py \
        --data-dir data \
        --out-dir artifacts \
        --tolerances 1,3,5,7 \
        --test-size 0.2 --valid-size 0.1 --seed 42

Outputs (under --out-dir)
-------------------------
    prepared/tolerance_{t}d/{train,valid,test}.csv   per-tolerance splits
    tables/benchmark_scale.{csv,tex}                 N / visits / chips / scenes
    tables/target_rates.{csv,tex}                    per-target observation rate
    tables/conductivity_shift.{csv,tex}              train vs test range
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TARGETS = ["chl_a", "tn", "tp", "turbidity", "conductivity"]
TARGET_LABELS = {
    "chl_a": r"chl-\textit{a}",
    "tn": "TN",
    "tp": "TP",
    "turbidity": "Turbidity",
    "conductivity": "Conductivity",
}


# --------------------------------------------------------------------------- #
# Core pipeline steps (match the full pipeline's matchup / QC / split logic)
# --------------------------------------------------------------------------- #
def create_matchups(in_situ: pd.DataFrame, eo: pd.DataFrame, tolerance_days: int) -> pd.DataFrame:
    """Join each in-situ sample to its nearest same-station acquisition within tolerance."""
    merged = in_situ.merge(eo, on="station_id", how="inner", suffixes=("", "_eo"))
    merged["abs_day_diff"] = (merged["sample_date"] - merged["acquisition_date"]).abs().dt.days
    matched = merged[merged["abs_day_diff"] <= int(tolerance_days)].copy()
    # Nearest acquisition per sample; deterministic tie-break by acquisition id.
    matched = matched.sort_values(["sample_id", "abs_day_diff", "eo_id"])
    matched = matched.drop_duplicates(subset=["sample_id"], keep="first")
    return matched.reset_index(drop=True)


def apply_qc(df: pd.DataFrame, min_targets_per_row: int = 1) -> pd.DataFrame:
    """Drop invalid dates/coords, coerce targets numeric, remove negatives, require labels."""
    cleaned = df.dropna(subset=["sample_date", "acquisition_date"]).copy()
    cleaned = cleaned[cleaned["lat"].between(-90, 90) & cleaned["lon"].between(-180, 180)]
    for target in TARGETS:
        if target in cleaned.columns:
            cleaned[target] = pd.to_numeric(cleaned[target], errors="coerce")
            cleaned.loc[cleaned[target] < 0, target] = pd.NA
    present = cleaned[TARGETS].notna().sum(axis=1)
    cleaned = cleaned[present >= min_targets_per_row]
    return cleaned.reset_index(drop=True)


def split_by_group(
    df: pd.DataFrame, group_column: str, test_size: float, valid_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lake-grouped split: no water body appears in more than one partition."""
    groups = df[group_column].astype(str)
    split1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tv, idx_test = next(split1.split(df, groups=groups))
    train_valid = df.iloc[idx_tv].reset_index(drop=True)
    test = df.iloc[idx_test].reset_index(drop=True)

    adjusted_valid = min(max(valid_size / max(1e-8, 1.0 - test_size), 0.05), 0.5)
    groups_tv = train_valid[group_column].astype(str)
    split2 = GroupShuffleSplit(n_splits=1, test_size=adjusted_valid, random_state=seed + 1)
    idx_train, idx_valid = next(split2.split(train_valid, groups=groups_tv))
    train = train_valid.iloc[idx_train].reset_index(drop=True)
    valid = train_valid.iloc[idx_valid].reset_index(drop=True)
    return train, valid, test


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def scale_row(matched: pd.DataFrame, tol: int) -> dict:
    return {
        "tau_days": tol,
        "N": len(matched),
        "visits": matched.groupby(["station_id", "sample_date"]).ngroups,
        "chips": matched.groupby(["station_id", "s2_item_id"]).ngroups,
        "scenes": matched["s2_item_id"].nunique(),
    }


def target_rates(matched: pd.DataFrame, tol: int) -> dict:
    row = {"tau_days": tol}
    for target in TARGETS:
        row[target] = round(100 * matched[target].notna().mean(), 1)
    return row


def conductivity_shift(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    def stats(df: pd.DataFrame, name: str) -> dict:
        c = pd.to_numeric(df["conductivity"], errors="coerce").dropna()
        return {
            "split": name,
            "lakes": df["lake_id"].nunique(),
            "n_obs": len(c),
            "median": round(c.median(), 1),
            "mean": round(c.mean(), 1),
            "p95": round(c.quantile(0.95), 1),
            "max": round(c.max(), 1),
        }

    tr = pd.concat([train, valid], ignore_index=True)
    train_p95 = pd.to_numeric(tr["conductivity"], errors="coerce").quantile(0.95)
    test_c = pd.to_numeric(test["conductivity"], errors="coerce").dropna()
    rows = [stats(tr, "train+valid"), stats(test, "test")]
    out = pd.DataFrame(rows)
    out.attrs["pct_test_above_train_p95"] = round(100 * (test_c > train_p95).mean(), 1)
    return out


# --------------------------------------------------------------------------- #
# LaTeX emitters (booktabs; numbers only, no identifying content)
# --------------------------------------------------------------------------- #
def _grp(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def scale_to_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}crrrr@{}}", r"\toprule",
        r"\textbf{$\tau$} & \textbf{$N$} & \textbf{Visits} & \textbf{Chips} & \textbf{Scenes} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r.tau_days)}\\,d & {_grp(int(r.N))} & {_grp(int(r.visits))} "
            f"& {_grp(int(r.chips))} & {_grp(int(r.scenes))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def rates_to_latex(df: pd.DataFrame, tols: list[int]) -> str:
    header = " & ".join([r"\textbf{Target}"] + [f"\\textbf{{{t}\\,d}}" for t in tols])
    lines = [r"\begin{tabular}{@{}l" + "r" * len(tols) + r"@{}}", r"\toprule", header + r" \\", r"\midrule"]
    order = sorted(TARGETS, key=lambda t: -df.set_index("tau_days").loc[tols[-1], t])
    by_tol = df.set_index("tau_days")
    for target in order:
        cells = " & ".join(f"{by_tol.loc[t, target]:.1f}" for t in tols)
        lines.append(f"{TARGET_LABELS[target]} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tolerance-aware WQ benchmark tables.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--in-situ-file", default="in_situ_subset.csv",
                        help="In-situ CSV under --data-dir. Point at the full file for exact paper numbers.")
    parser.add_argument("--eo-file", default="eo_features_subset.csv",
                        help="EO metadata CSV under --data-dir.")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--tolerances", default="1,3,5,7")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column", default="lake_id")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    tols = [int(t) for t in args.tolerances.split(",") if t.strip()]

    in_situ = pd.read_csv(data_dir / args.in_situ_file, parse_dates=["sample_date"])
    eo = pd.read_csv(data_dir / args.eo_file, parse_dates=["acquisition_date"])
    print(f"Loaded {len(in_situ):,} in-situ rows over {in_situ.lake_id.nunique()} lakes "
          f"and {len(eo):,} EO metadata rows.")

    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    scale_rows, rate_rows = [], []

    for tol in tols:
        matched = create_matchups(in_situ, eo, tol)
        qc = apply_qc(matched)
        train, valid, test = split_by_group(
            qc, args.group_column, args.test_size, args.valid_size, args.seed
        )
        prep = out_dir / "prepared" / f"tolerance_{tol}d"
        prep.mkdir(parents=True, exist_ok=True)
        train.to_csv(prep / "train.csv", index=False)
        valid.to_csv(prep / "valid.csv", index=False)
        test.to_csv(prep / "test.csv", index=False)

        scale_rows.append(scale_row(qc, tol))
        rate_rows.append(target_rates(qc, tol))
        print(f"  tau={tol}d: N={len(qc):,}  train/valid/test="
              f"{len(train)}/{len(valid)}/{len(test)}  test_lakes={test.lake_id.nunique()}")

        if tol == tols[-1]:
            shift = conductivity_shift(train, valid, test)
            shift.to_csv(out_dir / "tables" / "conductivity_shift.csv", index=False)
            print(f"  conductivity shift (tau={tol}d): "
                  f"{shift.attrs['pct_test_above_train_p95']}% of test > train p95")

    scale_df = pd.DataFrame(scale_rows)
    rate_df = pd.DataFrame(rate_rows)
    scale_df.to_csv(out_dir / "tables" / "benchmark_scale.csv", index=False)
    rate_df.to_csv(out_dir / "tables" / "target_rates.csv", index=False)
    (out_dir / "tables" / "benchmark_scale.tex").write_text(scale_to_latex(scale_df), encoding="utf-8")
    (out_dir / "tables" / "target_rates.tex").write_text(rates_to_latex(rate_df, tols), encoding="utf-8")

    n0, n1 = scale_df.iloc[0].N, scale_df.iloc[-1].N
    print(f"\nCorpus growth {tols[0]}d->{tols[-1]}d: "
          f"{n0:,} -> {n1:,} ({100 * (n1 - n0) / n0:.1f}%)")
    print(f"Tables written to {out_dir / 'tables'}")


if __name__ == "__main__":
    main()
