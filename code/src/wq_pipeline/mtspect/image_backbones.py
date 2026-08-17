from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RGBBackbone(nn.Module):
    _QWEN_PROCESSOR_CACHE: dict[str, object] = {}
    _QWEN_MODEL_CACHE: dict[str, nn.Module] = {}

    def __init__(
        self,
        model_name: str = "resnet18",
        hidden_dim: int = 256,
        pretrained: bool = True,
        backend: str = "timm",
        vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = int(hidden_dim)
        self.pretrained = bool(pretrained)
        self.backend = str(backend).strip().lower()
        self.vlm_model_id = str(vlm_model_id)

        self.backbone = None
        self.proj = None
        self.fallback_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fallback_proj = nn.Linear(128, hidden_dim)
        self._qwen_processor = None
        self._qwen_model = None

        if self.backend in {"qwen_vl", "qwen2.5_vl", "qwen2_5_vl"}:
            try:
                from transformers import AutoModel, AutoProcessor

                if self.vlm_model_id in self._QWEN_PROCESSOR_CACHE:
                    self._qwen_processor = self._QWEN_PROCESSOR_CACHE[self.vlm_model_id]
                else:
                    self._qwen_processor = AutoProcessor.from_pretrained(
                        self.vlm_model_id,
                        trust_remote_code=True,
                        use_fast=False,
                    )
                    self._QWEN_PROCESSOR_CACHE[self.vlm_model_id] = self._qwen_processor

                if self.vlm_model_id in self._QWEN_MODEL_CACHE:
                    self._qwen_model = self._QWEN_MODEL_CACHE[self.vlm_model_id]
                else:
                    self._qwen_model = AutoModel.from_pretrained(self.vlm_model_id, trust_remote_code=True)
                    for param in self._qwen_model.parameters():
                        param.requires_grad = False
                    self._qwen_model.eval()
                    self._QWEN_MODEL_CACHE[self.vlm_model_id] = self._qwen_model

                qwen_hidden = int(getattr(self._qwen_model.config, "hidden_size", 1024))
                self.proj = nn.Linear(qwen_hidden, hidden_dim)
            except Exception:
                self.backend = "timm"

        if self._qwen_model is None:
            try:
                import timm

                self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
                out_dim = int(getattr(self.backbone, "num_features", hidden_dim))
                self.proj = nn.Linear(out_dim, hidden_dim)
            except Exception:
                self.backbone = self.fallback_backbone
                if self.proj is None:
                    self.proj = self.fallback_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._qwen_model is not None and self._qwen_processor is not None:
            try:
                x_cpu = x.detach().cpu().clamp(0.0, 1.0)
                pil_images = []
                for i in range(x_cpu.shape[0]):
                    arr_f = x_cpu[i].permute(1, 2, 0).numpy()
                    arr_f = np.nan_to_num(arr_f, nan=0.0, posinf=1.0, neginf=0.0)
                    arr = (arr_f * 255.0).clip(0, 255).astype("uint8")
                    from PIL import Image

                    pil_images.append(Image.fromarray(arr))
                qinp = self._qwen_processor(images=pil_images, return_tensors="pt")
                qinp = {k: v.to(x.device) for k, v in qinp.items()}
                out = self._qwen_model(**qinp)
                if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    pooled = out.last_hidden_state.mean(dim=1)
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    pooled = out.pooler_output
                else:
                    pooled = out[0].mean(dim=1)
                return self.proj(pooled)
            except Exception:
                if self.fallback_backbone is not None:
                    z_fallback = self.fallback_backbone(x)
                    return self.fallback_proj(z_fallback)
        z = self.backbone(x)
        return self.proj(z)


class NPZSpectralBackbone(nn.Module):
    def __init__(self, in_channels: int = 12, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpectralBandBridge(nn.Module):
    def __init__(self, in_channels: int = 12) -> None:
        super().__init__()
        self.to_rgb = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(3)
        with torch.no_grad():
            self.to_rgb.weight.zero_()
            if in_channels >= 4:
                self.to_rgb.weight[0, 3, 0, 0] = 1.0
                self.to_rgb.weight[1, 2, 0, 0] = 1.0
                self.to_rgb.weight[2, 1, 0, 0] = 1.0

    def forward(self, npz_x: torch.Tensor) -> torch.Tensor:
        rgb = self.to_rgb(npz_x)
        return self.norm(rgb)
