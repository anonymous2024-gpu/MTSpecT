from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class WandbHandle:
    run: Any | None

    def log(self, payload: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_wandb(enabled: bool, project: str, run_name: str, config: dict[str, Any]) -> WandbHandle:
    if not enabled:
        return WandbHandle(run=None)
    try:
        import wandb
    except ImportError:
        return WandbHandle(run=None)

    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    try:
        run = wandb.init(project=project, name=run_name, config=config, mode=mode)
    except Exception:
        run = wandb.init(project=project, name=run_name, config=config, mode="offline")
    return WandbHandle(run=run)
