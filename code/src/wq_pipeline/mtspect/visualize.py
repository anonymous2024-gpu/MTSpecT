from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_training_curves(history_csv: Path, output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(history_csv)
    produced: list[Path] = []

    if "train_loss" in hist.columns and "val_loss" in hist.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        ax.plot(hist["epoch"], hist["val_loss"], label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MT-SpecT Training Loss")
        ax.legend()
        path = output_dir / "loss_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        produced.append(path)

    if "val_r2" in hist.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(hist["epoch"], hist["val_r2"], label="val_r2", color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R2 (accuracy proxy)")
        ax.set_title("MT-SpecT Validation R2")
        ax.legend()
        path = output_dir / "val_r2_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        produced.append(path)

    if "epoch_sec" in hist.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(hist["epoch"], hist["epoch_sec"], label="epoch_sec", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")
        ax.set_title("MT-SpecT Epoch Training Time")
        ax.legend()
        path = output_dir / "epoch_time_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        produced.append(path)

    return produced
