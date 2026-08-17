"""
Load one multimodal sample and print the shape of every input stream.

Loads all six modalities of Table 1 for one record (tabular, temporal band
sequence, quality scalars, RGB chip, 12-band NPZ chip, station-context text
prompt) and prints their shapes. Run from the repository root:

    python scripts/10_load_multimodal_sample.py --split train --index 0

Requires numpy and pandas; Pillow is used for the RGB chip if available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["chl_a", "tn", "tp", "turbidity", "conductivity"]
TEMPORAL_BANDS = ["b02", "b03", "b04", "b08"]      # -> X_temp (T, 4)
QUALITY_FIELDS = ["cloud", "day_diff", "offset_days"]  # -> Q (T, 3)


def temporal_steps(columns) -> list[int]:
    steps = set()
    for c in columns:
        if c.startswith("img_t") and c.endswith("_b02"):
            steps.add(int(c[len("img_t"):].split("_")[0]))
    return sorted(steps)


def build_prompt(row: pd.Series) -> str:
    """Human-readable station-context prompt (tokenised inside the model)."""
    month = pd.to_datetime(row["sample_date"]).month
    season = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
              5: "spring", 6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}[month]
    cloud = row.get("s2_cloud_cover_chip", float("nan"))
    dt = row.get("abs_day_diff_chip", float("nan"))
    return (f"Station {row['station_id']} on lake {row['lake_id']}, {season}; "
            f"Sentinel-2 match {dt:.0f} d away at {cloud:.0f}% cloud.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect one multimodal sample.")
    ap.add_argument("--data-dir", default="data/multimodal")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.data_dir)
    df = pd.read_csv(base / f"multimodal_{args.split}.csv")
    row = df.iloc[args.index]
    steps = temporal_steps(df.columns)
    T = len(steps)

    # 1. tabular targets (+ station metadata would be appended in the full model)
    y = row[TARGETS].to_numpy(dtype=float)

    # 2. temporal band sequence X_temp (T, 4)
    x_temp = np.array([[row[f"img_t{s}_{b}"] for b in TEMPORAL_BANDS] for s in steps], dtype=float)

    # 3. quality scalars Q (T, 3)
    q = np.array([[row[f"img_t{s}_{f}"] for f in QUALITY_FIELDS] for s in steps], dtype=float)

    # 4. RGB chip X_rgb (3, H, W)
    rgb_path = base / row["chip_image_path"]
    try:
        from PIL import Image
        x_rgb = np.asarray(Image.open(rgb_path).convert("RGB")).transpose(2, 0, 1)
        rgb_desc = f"{x_rgb.shape} {x_rgb.dtype}"
    except Exception as exc:  # noqa: BLE001 - Pillow optional
        rgb_desc = f"(not decoded: {exc}); file present={rgb_path.exists()}"

    # 5. 12-band NPZ chip X_npz (12, H, W)
    npz = np.load(base / row["chip_path"], allow_pickle=True)
    x_npz = npz["chip"]
    bands = [str(b) for b in npz["bands"]]

    # 6. station-context text prompt x_txt
    prompt = build_prompt(row)

    print(f"Sample {row['sample_id']}  (split={args.split}, lake={row['lake_id']})")
    print("-" * 60)
    print(f"  y (targets)        : shape {y.shape}  values {np.round(y, 2)}")
    print(f"  X_temp (T,4)       : shape {x_temp.shape}  bands {TEMPORAL_BANDS}")
    print(f"  Q (T,3)            : shape {q.shape}  fields {QUALITY_FIELDS}")
    print(f"  X_rgb (3,H,W)      : {rgb_desc}")
    print(f"  X_npz (12,H,W)     : shape {x_npz.shape} {x_npz.dtype}  bands {bands}")
    print(f"  x_txt (prompt)     : \"{prompt}\"")
    print("-" * 60)
    print(f"Temporal steps T={T}  (offset grid indices {steps})")


if __name__ == "__main__":
    main()
