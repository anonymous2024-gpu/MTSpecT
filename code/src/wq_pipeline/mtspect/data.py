from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


TARGETS = ["chl_a", "tn", "tp", "turbidity", "conductivity"]
DEFAULT_NPZ_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


@dataclass
class BatchTensors:
    tabular_x: torch.Tensor
    temporal_x: torch.Tensor
    quality_x: torch.Tensor
    rgb_x: torch.Tensor
    npz_x: torch.Tensor
    text_tokens: torch.Tensor
    text_mask: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor


def _safe_numeric(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


def _infer_temporal_steps(columns: Iterable[str]) -> list[int]:
    steps: set[int] = set()
    for col in columns:
        if col.startswith("img_t"):
            rest = col[len("img_t") :]
            idx_str = ""
            for ch in rest:
                if ch.isdigit():
                    idx_str += ch
                else:
                    break
            if idx_str:
                steps.add(int(idx_str))
    return sorted(steps)


def _text_prompt(row: pd.Series) -> str:
    parts = [
        f"municipality={str(row.get('municipality', 'unknown'))}",
        f"waterbody_type={str(row.get('WaterbodyType', 'unknown'))}",
        f"environment={str(row.get('EnvironmentType', 'unknown'))}",
        f"month={str(row.get('sample_date', 'unknown'))[:7]}",
        f"station={str(row.get('station_id', 'unknown'))}",
        f"lake={str(row.get('lake_id', 'unknown'))}",
    ]
    return " | ".join(parts).lower()


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int], unk_idx: int = 1, pad_idx: int = 0) -> None:
        self.vocab = vocab
        self.unk_idx = int(unk_idx)
        self.pad_idx = int(pad_idx)

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2) -> "SimpleTokenizer":
        freq: dict[str, int] = {}
        for text in texts:
            for tok in text.split():
                freq[tok] = freq.get(tok, 0) + 1
        vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for tok, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if count >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        return cls(vocab=vocab)

    def encode(self, text: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        ids = np.full((max_len,), self.pad_idx, dtype=np.int64)
        mask = np.zeros((max_len,), dtype=np.float32)
        toks = text.split()
        n = min(max_len, len(toks))
        for i in range(n):
            ids[i] = self.vocab.get(toks[i], self.unk_idx)
            mask[i] = 1.0
        return ids, mask


class HFPromptTokenizer:
    def __init__(self, model_id: str, max_len: int = 64) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for HF-based text tokenization.") from exc
        self.model_id = str(model_id)
        self.max_len = int(max_len)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def batch_encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.float32)
        return input_ids, attention_mask


@dataclass
class FeatureSpec:
    tabular_columns: list[str]
    temporal_steps: list[int]
    temporal_band_columns: list[str]
    quality_columns: list[str]


@dataclass
class NormalizationStats:
    tabular_mean: np.ndarray
    tabular_std: np.ndarray
    temporal_mean: np.ndarray
    temporal_std: np.ndarray
    quality_mean: np.ndarray
    quality_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def infer_feature_spec(df: pd.DataFrame, targets: list[str] | None = None) -> FeatureSpec:
    targets = targets or TARGETS
    forbidden = set(targets) | {
        "sample_id",
        "sample_date",
        "station_id",
        "lake_id",
        "eo_id",
        "acquisition_date",
    }

    temporal_steps = _infer_temporal_steps(df.columns)
    if not temporal_steps:
        raise ValueError("No temporal image columns found (expected img_t*_*) in temporal dataset.")

    candidate_bands = ["b02", "b03", "b04", "b08"]
    temporal_band_columns: list[str] = []
    for b in candidate_bands:
        has_all = all(f"img_t{s}_{b}" in df.columns for s in temporal_steps)
        if has_all:
            temporal_band_columns.append(b)
    if not temporal_band_columns:
        raise ValueError("No complete temporal band sets found for img_t*_(b02|b03|b04|b08).")

    quality_candidates = ["cloud", "day_diff", "offset_days"]
    quality_columns: list[str] = []
    for q in quality_candidates:
        col_names = [f"img_t{s}_{q}" for s in temporal_steps]
        if all(c in df.columns for c in col_names):
            quality_columns.append(q)
    if not quality_columns:
        quality_columns = ["cloud"] if all(f"img_t{s}_cloud" in df.columns for s in temporal_steps) else []

    tabular_cols: list[str] = []
    for col in df.columns:
        if col in forbidden:
            continue
        if col.startswith("img_t"):
            continue
        if col.endswith("_eo"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            tabular_cols.append(col)

    if not tabular_cols:
        raise ValueError("No numeric tabular columns found for MT-SpecT tabular branch.")

    return FeatureSpec(
        tabular_columns=tabular_cols,
        temporal_steps=temporal_steps,
        temporal_band_columns=temporal_band_columns,
        quality_columns=quality_columns,
    )


class MTSpectDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_spec: FeatureSpec,
        tokenizer: SimpleTokenizer | HFPromptTokenizer,
        max_text_len: int = 64,
        targets: list[str] | None = None,
        npz_band_order: list[str] | None = None,
        shared_chip_dir: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.spec = feature_spec
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)
        self.targets = targets or TARGETS
        self.npz_band_order = npz_band_order or list(DEFAULT_NPZ_BAND_ORDER)
        self.shared_chip_dir = shared_chip_dir

        self.tabular = np.stack([_safe_numeric(self.df[c]) for c in self.spec.tabular_columns], axis=1).astype(np.float32)

        t, b = len(self.spec.temporal_steps), len(self.spec.temporal_band_columns)
        temporal = np.zeros((len(self.df), t, b), dtype=np.float32)
        for ti, step in enumerate(self.spec.temporal_steps):
            for bi, band in enumerate(self.spec.temporal_band_columns):
                temporal[:, ti, bi] = _safe_numeric(self.df[f"img_t{step}_{band}"])
        self.temporal = temporal

        qdim = max(1, len(self.spec.quality_columns))
        quality = np.zeros((len(self.df), t, qdim), dtype=np.float32)
        if self.spec.quality_columns:
            for qi, qname in enumerate(self.spec.quality_columns):
                for ti, step in enumerate(self.spec.temporal_steps):
                    quality[:, ti, qi] = _safe_numeric(self.df[f"img_t{step}_{qname}"])
        self.quality = quality

        y = np.zeros((len(self.df), len(self.targets)), dtype=np.float32)
        y_mask = np.zeros((len(self.df), len(self.targets)), dtype=np.float32)
        for ti, target in enumerate(self.targets):
            if target not in self.df.columns:
                continue
            vals = pd.to_numeric(self.df[target], errors="coerce").to_numpy(dtype=np.float32)
            finite = np.isfinite(vals)
            y_mask[:, ti] = finite.astype(np.float32)
            y[:, ti] = np.nan_to_num(vals, nan=0.0)
        self.y = y
        self.y_mask = y_mask

        self.text_prompts = [_text_prompt(self.df.iloc[i]) for i in range(len(self.df))]
        if isinstance(self.tokenizer, HFPromptTokenizer):
            self.text_input_ids, self.text_attention = self.tokenizer.batch_encode(self.text_prompts)
        else:
            self.text_input_ids = None
            self.text_attention = None

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        p = str(path or "").strip()
        if not p or not Path(p).exists():
            return np.zeros((3, 256, 256), dtype=np.float32)
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            return arr.astype(np.float32)
        except Exception:
            return np.zeros((3, 256, 256), dtype=np.float32)

    def _load_npz(self, path: str, sample_id: str) -> np.ndarray:
        def _read_chip(p: str) -> tuple[np.ndarray, list[str]]:
            data = np.load(p, allow_pickle=False)
            if "chip" in data.files:
                arr_local = data["chip"]
            else:
                arr_local = data[data.files[0]]
            arr_local = np.asarray(arr_local, dtype=np.float32)
            if arr_local.ndim == 2:
                arr_local = arr_local[None, :, :]
            bands = [str(b) for b in data.get("bands", np.array([f"B{i:02d}" for i in range(arr_local.shape[0])])).tolist()]
            return arr_local.astype(np.float32), bands

        source_path = str(path or "").strip()
        if not source_path or not Path(source_path).exists():
            return np.zeros((len(self.npz_band_order), 256, 256), dtype=np.float32)
        try:
            arr, bands = _read_chip(source_path)
        except Exception:
            return np.zeros((len(self.npz_band_order), 256, 256), dtype=np.float32)

        if len(bands) < len(self.npz_band_order) and self.shared_chip_dir is not None:
            shared_path = self.shared_chip_dir / f"{sample_id}.npz"
            if shared_path.exists():
                arr_s, bands_s = _read_chip(str(shared_path))
                if len(bands_s) >= len(bands):
                    arr, bands = arr_s, bands_s

        h, w = int(arr.shape[-2]), int(arr.shape[-1])
        out = np.zeros((len(self.npz_band_order), h, w), dtype=np.float32)
        band_to_idx = {b.upper(): i for i, b in enumerate(bands)}
        for oi, ob in enumerate(self.npz_band_order):
            idx = band_to_idx.get(ob.upper())
            if idx is not None and idx < arr.shape[0]:
                out[oi] = arr[idx]
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = float(np.max(np.abs(out))) if out.size > 0 else 0.0
        if max_abs > 2.0:
            out = out / 10000.0
        out = np.clip(out, -2.0, 2.0)
        return out.astype(np.float32)

    def apply_normalization(self, stats: NormalizationStats, normalize_targets: bool = True) -> None:
        self.tabular = ((self.tabular - stats.tabular_mean) / stats.tabular_std).astype(np.float32)
        self.temporal = ((self.temporal - stats.temporal_mean) / stats.temporal_std).astype(np.float32)
        self.quality = ((self.quality - stats.quality_mean) / stats.quality_std).astype(np.float32)
        if normalize_targets:
            self.y = ((self.y - stats.target_mean) / stats.target_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if self.text_input_ids is not None and self.text_attention is not None:
            text_ids = self.text_input_ids[idx]
            text_mask = self.text_attention[idx]
        else:
            text_ids, text_mask = self.tokenizer.encode(self.text_prompts[idx], max_len=self.max_text_len)
        row = self.df.iloc[idx]
        sample_id = str(row.get("sample_id", ""))
        return {
            "tabular_x": self.tabular[idx],
            "temporal_x": self.temporal[idx],
            "quality_x": self.quality[idx],
            "rgb_x": self._load_rgb(str(row.get("chip_image_path", ""))),
            "npz_x": self._load_npz(str(row.get("chip_path", "")), sample_id=sample_id),
            "text_tokens": text_ids,
            "text_mask": text_mask,
            "y": self.y[idx],
            "y_mask": self.y_mask[idx],
        }


def collate_batch(items: list[dict[str, np.ndarray]]) -> BatchTensors:
    def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(np.stack([it[name] for it in items], axis=0), dtype=dtype)

    return BatchTensors(
        tabular_x=stack("tabular_x", torch.float32),
        temporal_x=stack("temporal_x", torch.float32),
        quality_x=stack("quality_x", torch.float32),
        rgb_x=stack("rgb_x", torch.float32),
        npz_x=stack("npz_x", torch.float32),
        text_tokens=stack("text_tokens", torch.long),
        text_mask=stack("text_mask", torch.float32),
        y=stack("y", torch.float32),
        y_mask=stack("y_mask", torch.float32),
    )


def load_split_frames(prepared_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temporal_dir = prepared_dir / "temporal"
    train_path = temporal_dir / "train_temporal.csv"
    valid_path = temporal_dir / "valid_temporal.csv"
    test_path = temporal_dir / "test_temporal.csv"
    if not train_path.exists() or not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing temporal split files in {temporal_dir}. Expected train_temporal.csv, valid_temporal.csv, test_temporal.csv"
        )
    return pd.read_csv(train_path), pd.read_csv(valid_path), pd.read_csv(test_path)


def load_multimodal_split_frames(
    prepared_dir: Path,
    require_all_modalities: bool = True,
    min_rows: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_path = prepared_dir / "multimodal_train.csv"
    valid_path = prepared_dir / "multimodal_valid.csv"
    test_path = prepared_dir / "multimodal_test.csv"
    if not train_path.exists() or not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing multimodal split files in {prepared_dir}. Expected multimodal_train.csv, multimodal_valid.csv, multimodal_test.csv"
        )

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        req_cols = ["chip_path", "chip_image_path", "chip_status", "temporal_feature_status"]
        for c in req_cols:
            if c not in df.columns:
                raise ValueError(f"Required multimodal column missing: {c}")

        out = df.copy()
        if require_all_modalities:
            out = out[(out["chip_status"] == "ok") & (out["temporal_feature_status"] == "ok")]
            out = out[out["chip_path"].astype(str).str.len() > 0]
            out = out[out["chip_image_path"].astype(str).str.len() > 0]
            out = out[out["chip_path"].apply(lambda p: Path(str(p)).exists())]
            out = out[out["chip_image_path"].apply(lambda p: Path(str(p)).exists())]
        return out.reset_index(drop=True)

    train_f = _filter(train)
    valid_f = _filter(valid)
    test_f = _filter(test)

    coverage = {
        "train_rows_raw": int(len(train)),
        "valid_rows_raw": int(len(valid)),
        "test_rows_raw": int(len(test)),
        "train_rows_kept": int(len(train_f)),
        "valid_rows_kept": int(len(valid_f)),
        "test_rows_kept": int(len(test_f)),
    }

    if require_all_modalities and (len(train_f) < min_rows or len(valid_f) < 1 or len(test_f) < 1):
        raise ValueError(
            f"Insufficient strict multimodal rows after filtering: {coverage}. "
            "Check chip/temporal coverage for this tolerance."
        )

    return train_f, valid_f, test_f, coverage


def _safe_stats(arr: np.ndarray, mask: np.ndarray | None = None, axis: int | tuple[int, ...] = 0) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        mean = np.nanmean(arr, axis=axis)
        std = np.nanstd(arr, axis=axis)
    else:
        valid = mask.astype(bool)
        safe = np.where(valid, arr, np.nan)
        mean = np.nanmean(safe, axis=axis)
        std = np.nanstd(safe, axis=axis)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_normalization_stats(train_ds: MTSpectDataset) -> NormalizationStats:
    tab_mean, tab_std = _safe_stats(train_ds.tabular, axis=0)
    tmp_mean, tmp_std = _safe_stats(train_ds.temporal, axis=(0, 1))
    q_mean, q_std = _safe_stats(train_ds.quality, axis=(0, 1))
    tgt_mean, tgt_std = _safe_stats(train_ds.y, mask=train_ds.y_mask, axis=0)

    return NormalizationStats(
        tabular_mean=tab_mean,
        tabular_std=tab_std,
        temporal_mean=tmp_mean.reshape(1, 1, -1),
        temporal_std=tmp_std.reshape(1, 1, -1),
        quality_mean=q_mean.reshape(1, 1, -1),
        quality_std=q_std.reshape(1, 1, -1),
        target_mean=tgt_mean.reshape(1, -1),
        target_std=tgt_std.reshape(1, -1),
    )


def invert_target_scaling(mu: np.ndarray, logvar: np.ndarray, stats: NormalizationStats) -> tuple[np.ndarray, np.ndarray]:
    mu_unscaled = mu * stats.target_std + stats.target_mean
    logvar_unscaled = logvar + 2.0 * np.log(np.clip(stats.target_std, 1e-6, None))
    return mu_unscaled.astype(np.float32), logvar_unscaled.astype(np.float32)
