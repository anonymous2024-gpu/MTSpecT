from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.raw.get("artifacts_dir", "artifacts"))

    @property
    def paths(self) -> dict[str, str]:
        return dict(self.raw.get("paths", {}))

    @property
    def targets(self) -> list[str]:
        return list(self.raw.get("targets", []))

    @property
    def matchup(self) -> dict[str, Any]:
        return dict(self.raw.get("matchup", {}))

    @property
    def split(self) -> dict[str, Any]:
        return dict(self.raw.get("split", {}))

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.raw.get("training", {}))

    @property
    def features(self) -> dict[str, Any]:
        return dict(self.raw.get("features", {}))

    @property
    def data_ingest(self) -> dict[str, Any]:
        return dict(self.raw.get("data_ingest", {}))

    @property
    def fetch(self) -> dict[str, Any]:
        return dict(self.raw.get("fetch", {}))

    @property
    def pretrained(self) -> dict[str, Any]:
        return dict(self.raw.get("pretrained", {}))

    @property
    def fusion(self) -> dict[str, Any]:
        return dict(self.raw.get("fusion", {}))

    @property
    def hf_pretrained(self) -> dict[str, Any]:
        return dict(self.raw.get("hf_pretrained", {}))

    @property
    def llm(self) -> dict[str, Any]:
        return dict(self.raw.get("llm", {}))

    @property
    def finland_syke(self) -> dict[str, Any]:
        return dict(self.raw.get("finland_syke", {}))

    @property
    def s2_metadata(self) -> dict[str, Any]:
        return dict(self.raw.get("s2_metadata", {}))

    @property
    def temporal_features(self) -> dict[str, Any]:
        return dict(self.raw.get("temporal_features", {}))

    @property
    def image_chips(self) -> dict[str, Any]:
        return dict(self.raw.get("image_chips", {}))

    @property
    def multimodal_index(self) -> dict[str, Any]:
        return dict(self.raw.get("multimodal_index", {}))

    @property
    def vesla_odata(self) -> dict[str, Any]:
        return dict(self.raw.get("vesla_odata", {}))


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a YAML mapping.")

    return PipelineConfig(raw=raw)
