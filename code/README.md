# MT-SpecT — Reproducibility Guide

This folder contains the source, driver scripts, and data needed to reproduce
the experiments and figures in the paper. It ships (i) the complete source
package and numbered pipeline scripts, (ii) point tables and the benchmark
builder, and (iii) a multimodal data sample with all six input streams. See
[§ Reproducing without downloads](#reproducing-without-downloads) for what runs
directly from this bundle, and the step-by-step sections for the full pipeline.

---

## Contents

```
reproducibility_code/
├── README.md               ← this file
├── requirements.txt        ← pinned dependencies
├── pyproject.toml          ← package definition (install with pip install -e .)
├── configs/
│   └── default.yaml        ← pipeline configuration
├── data/                   ← shipped subset (see § Data)
│   ├── in_situ_subset.csv          ← in-situ measurements (40-lake sample)
│   ├── eo_features_subset.csv      ← paired Sentinel-2 acquisition metadata
│   └── multimodal/                 ← 22-sample all-modalities example (12-band chips + RGB + temporal)
│       ├── multimodal_{train,valid,test}.csv
│       └── chips/{npz,rgb}/        ← per-sample 12-band NPZ and RGB chips
├── src/
│   └── wq_pipeline/        ← full source package
└── scripts/
    ├── 00_build_benchmark_tables.py   ← reproduce dataset tables (pandas only, no GPU/download)
    ├── 01_prepare_data.py          ← data download + split generation
    ├── 02_train_baselines.py       ← all baseline model families
    ├── 03_train_mtspect.py         ← MT-SpecT++ tolerance/batch sweep
    ├── 04_select_best_model.py     ← select best checkpoint per tolerance
    ├── 05_run_ablations.py         ← 20-variant ablation studies
    ├── 06_run_ablations_normalised.py  ← normalised ablation metrics (nRMSE/nMAE/R²)
    ├── 07_run_kfold.py             ← 5-fold lake-grouped cross-validation
    ├── 08_export_predictions.py    ← per-sample test predictions with coordinates
    ├── 09_build_visualizations.py  ← all paper figures
    └── 10_load_multimodal_sample.py   ← load one sample across all six modalities
```

---

## Environment Setup

**Python:** 3.11+  
**CUDA:** 12.4+ (for RTX 5090 / Ada Lovelace)

```bash
# 1. Clone or extract the code
cd reproducibility_code

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode
pip install -e .

# 4. Verify
python -c "import wq_pipeline; print('OK')"
```

Key dependencies:

- `torch >= 2.10` with CUDA 12.4
- `transformers >= 4.46` — Qwen2.5-VL-3B and MiniLM text encoder
- `planetary-computer >= 1.0` — Sentinel-2 STAC access (free, no API key)
- `peft >= 0.13` — LoRA adapters for the text encoder
- `rasterio >= 1.4`, `geopandas >= 0.14` — geospatial chip extraction

---

## Data Access

### In-situ water quality (SYKE/VESLA)

Finnish lake measurements are publicly available from the Finnish Environment Institute (SYKE).  
The pipeline fetches them automatically via the VESLA OData API in Step 1.  
No account or API key is required.

### Sentinel-2 imagery

S2 Level-2A imagery is accessed via the Microsoft Planetary Computer STAC API.  
No authentication is required for read access.

### Shipped data (`data/`)

- **`data/in_situ_subset.csv`** + **`data/eo_features_subset.csv`** — point
  tables (deterministic sample, seed 42, stratified by lake-median conductivity)
  that drive `00_build_benchmark_tables.py`.
- **`data/multimodal/`** — a sample of 22 records with **all six modalities**:
  12-band NPZ chips (`chips/npz/`), RGB chips (`chips/rgb/`), the nine-step
  temporal stack and quality scalars (`img_t*` columns in the CSVs), tabular
  targets, and the fields used to build the station prompt.

Chip paths in the multimodal CSVs are **relative** (`chips/npz/<id>.npz`), so the
sample is portable and self-contained.

---

### Reproducing the full corpus

The complete point tables (19,386 in-situ records, 138 stations) and the full
image corpus are regenerated from public sources — the SYKE/VESLA OData API and
the Microsoft Planetary Computer Sentinel-2 STAC (both free, no key) — via
`01_prepare_data.py`.

---

## Reproducing without downloads

Two things run **directly from this bundle** with no GPU and no network — only
`pandas`, `numpy`, `scikit-learn` (and Pillow for the RGB chip):

```bash
# (a) Dataset tables — matchup, QC, lake-grouped split, scale/rate/shift tables.
#     Writes CSV + drop-in booktabs .tex under artifacts/tables/.
python scripts/00_build_benchmark_tables.py --data-dir data --out-dir artifacts \
    --tolerances 1,3,5,7 --test-size 0.2 --valid-size 0.1 --seed 42

# (b) Multimodal data path — load one sample across all six input streams.
python scripts/10_load_multimodal_sample.py --split train --index 0
```

For the full-network table values, run the same command on the complete point
tables (see [Reproducing the full corpus](#reproducing-the-full-corpus)).

---

## Reproducing Results — Step by Step

### Step 1 — Prepare Data

```bash
# Full pipeline (downloads ~500 MB, extracts chips)
python scripts/01_prepare_data.py --tolerances 1,3,5,7

# Skip download if you already have data/raw/in_situ.csv and eo_features.csv
python scripts/01_prepare_data.py --tolerances 1,3,5,7 --skip-fetch --skip-s2
```

Outputs: `artifacts/tolerance_{N}d/prepared/{train,valid,test}.csv` + multimodal CSVs, chips, temporal sequences.

---

### Step 2 — Train Baselines

```bash
# All baseline families (~2–4 hours on CPU for tabular; GPU needed for image/HF)
python scripts/02_train_baselines.py --tolerances 1,3,5,7

# Tabular only (fast, ~20 min)
python scripts/02_train_baselines.py --tolerances 1,3,5,7 --families tabular,tabular_family
```

Outputs: `artifacts/prepared/baseline_metric_summary.{json,csv}`

---

### Step 3 — Train MT-SpecT++

```bash
# Paper grid: 4 tolerances × 4 batch sizes = 16 runs (~24–48 hours)
python scripts/03_train_mtspect.py \
    --tolerances 1,3,5,7 \
    --batch-sizes 32,64,128,256 \
    --epochs 100 \
    --use-amp \
    --no-wandb

# Reproduce exact paper checkpoints (best config per tolerance)
#   τ=1d: BS=128,  τ=3d: BS=64,  τ=5d: BS=32,  τ=7d: BS=32
python scripts/03_train_mtspect.py --tolerances 1 --batch-sizes 128 --epochs 100 --no-wandb
python scripts/03_train_mtspect.py --tolerances 3 --batch-sizes 64  --epochs 100 --no-wandb
python scripts/03_train_mtspect.py --tolerances 5 --batch-sizes 32  --epochs 100 --no-wandb
python scripts/03_train_mtspect.py --tolerances 7 --batch-sizes 32  --epochs 100 --no-wandb
```

Outputs: `artifacts/tolerance_{N}d/results/mtspect/mtspect_tol{N}_bs{K}/checkpoint.pt`

---

### Step 4 — Select Best Model

```bash
python scripts/04_select_best_model.py \
    --tolerances 1,3,5,7 \
    --output-json artifacts/prepared/mtspect_best_per_tolerance.json \
    --output-csv  artifacts/prepared/mtspect_best_per_tolerance.csv
```

Selection uses a 9-metric weighted composite score (equal weights):  
RMSE, MAE, R², MAPE, NLL, CRPS, |PICP−0.95|, MPIW, ECE.

---

### Step 5 — Ablation Studies

```bash
python scripts/05_run_ablations.py \
    --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
    --tolerances 1,3,5,7 \
    --epochs 100 \
    --allow-missing-modalities \
    --no-wandb
```

Runs 20 variants per tolerance (80 total).  
Outputs: `artifacts/tolerance_{N}d/results/mtspect/ablations/ablation_{name}.json`  
Combined: `artifacts/prepared/mtspect_ablations.json`

---

### Step 6 — Normalised Ablation Metrics

```bash
python scripts/06_run_ablations_normalised.py \
    --tolerances 1 3 5 7 \
    --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
    --no-wandb \
    --allow-missing-modalities
```

Computes nRMSE = RMSE / σ*train per target for every ablation variant.  
Outputs: `artifacts/tolerance*{N}d/results/mtspect/ablations/ablation_normalised_metrics.json`

---

### Step 7 — K-Fold Cross-Validation

```bash
python scripts/07_run_kfold.py \
    --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
    --n-folds 5 \
    --epochs 100 \
    --allow-missing-modalities \
    --no-wandb
```

5-fold lake-grouped CV at all 4 tolerances (20 training runs).  
Outputs:

- `artifacts/tolerance_{N}d/results/mtspect/kfold/fold_{k}/metrics.json`
- `artifacts/prepared/mtspect_kfold_all_tolerances.json`

---

### Step 8 — Export Test Predictions

```bash
python scripts/08_export_predictions.py \
    --selection-json artifacts/prepared/mtspect_best_per_tolerance.json \
    --allow-missing-modalities
```

Outputs: `artifacts/tolerance_{N}d/results/mtspect/test_predictions_with_coords.csv`  
Columns: `lat, lon, station_id, lake_id, municipality, target, y_true, y_pred, y_std`

---

### Step 9 — Generate Figures

```bash
# All figures
python scripts/09_build_visualizations.py --tolerances 1 3 5 7

# Calibration / reliability diagrams only
python scripts/09_build_visualizations.py --calibration-only

# Maps only (requires Step 8 predictions)
python scripts/09_build_visualizations.py --maps-only --tolerances 1 3 5 7
```

Outputs:

- `artifacts/figures/phase_i/` — coverage, performance, uncertainty, delta maps
- `artifacts/figures/calibration/reliability_4panel.png`
- `artifacts/figures/calibration/reliability_macro.png`
- `artifacts/visualization/phase_j/` — metric-curve panels

---

## Quick Run (Minimal Reproduction)

To reproduce only the main MT-SpecT++ results at τ=7d:

```bash
# 1. Prepare splits
python scripts/01_prepare_data.py --tolerances 7 --skip-fetch --skip-s2

# 2. Train MT-SpecT++ (best config)
python scripts/03_train_mtspect.py --tolerances 7 --batch-sizes 32 --epochs 100 --no-wandb

# 3. Select best model
python scripts/04_select_best_model.py \
    --tolerances 7 \
    --output-json artifacts/prepared/mtspect_best_per_tolerance.json \
    --output-csv  artifacts/prepared/mtspect_best_per_tolerance.csv

# 4. Export predictions
python scripts/08_export_predictions.py \
    --selection-json artifacts/prepared/mtspect_best_per_tolerance.json

# 5. Reliability diagram
python scripts/09_build_visualizations.py --calibration-only
```

---

## Optimal Hyperparameters

| Tolerance | Batch size | LR   | Dropout |
| --------- | ---------- | ---- | ------- |
| τ=1d      | 128        | 1e-4 | 0.10    |
| τ=3d      | 64         | 4e-4 | 0.15    |
| τ=5d      | 32         | 7e-4 | 0.30    |
| τ=7d      | 32         | 3e-4 | 0.10    |

Common settings:

- Hidden dimension: 256
- Optimizer: AdamW
- Scheduler: cosine annealing
- Early stopping: patience=15, min_delta=1e-4
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2` + LoRA (r=16, α=64, dropout=0.05)
- Vision RGB encoder: `Qwen/Qwen2.5-VL-3B-Instruct` (frozen)
- Vision multispectral: SpectralCNN (12-band Sentinel-2)
- Max epochs: 100, AMP enabled
- Loss weighting: $\lambda_{h}=1.0$, $\lambda_{a}=0.5$, $\lambda_{r}=0.5$, $\lambda_{p} \in [0.01,0.05]$

---

## Troubleshooting

**`ImportError: wq_pipeline`** — run `pip install -e .` from the `reproducibility_code/` directory.

**CUDA out of memory** — reduce batch size (`--batch-sizes 16`) or add `--allow-missing-modalities`
to disable the full multimodal requirement.

**Planetary Computer STAC errors** — check your internet connection; no API key is needed but
the service is occasionally throttled. Retry after a few minutes.

**Missing `temporal/` or `chips/` directories** — Step 1 must complete fully for all
tolerances before Steps 3–8.
