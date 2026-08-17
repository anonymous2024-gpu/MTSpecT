from __future__ import annotations


def require_torch() -> tuple:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for fusion training. Install it first, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu126"
        ) from exc
    return torch, nn, optim


def resolve_profile(profile: str | None) -> dict[str, float | int]:
    name = str(profile or "small").strip().lower()
    profiles: dict[str, dict[str, float | int]] = {
        "small": {
            "hidden_dim": 192,
            "tabular_hidden_dim": 128,
            "temporal_hidden_dim": 128,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.1,
            "batch_size": 256,
            "epochs": 30,
            "learning_rate": 8e-4,
            "weight_decay": 1e-5,
        },
        "medium": {
            "hidden_dim": 256,
            "tabular_hidden_dim": 192,
            "temporal_hidden_dim": 192,
            "num_heads": 6,
            "num_layers": 2,
            "dropout": 0.1,
            "batch_size": 192,
            "epochs": 45,
            "learning_rate": 6e-4,
            "weight_decay": 1e-5,
        },
        "large": {
            "hidden_dim": 320,
            "tabular_hidden_dim": 256,
            "temporal_hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 3,
            "dropout": 0.15,
            "batch_size": 128,
            "epochs": 60,
            "learning_rate": 4e-4,
            "weight_decay": 1e-5,
        },
    }
    return profiles.get(name, profiles["small"])


def build_fusion_model(
    tabular_dim: int,
    temporal_input_dim: int,
    target_dim: int,
    temporal_steps: int,
    profile: dict[str, float | int],
    fusion_mode: str = "cross_attention",
    pretrained_checkpoint_path: str | None = None,
):
    torch, nn, _ = require_torch()

    class FusionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fusion_mode = str(fusion_mode).strip().lower()
            tab_hid = int(profile["tabular_hidden_dim"])
            temporal_hid = int(profile["temporal_hidden_dim"])
            hidden_dim = int(profile["hidden_dim"])
            n_heads = int(profile["num_heads"])
            n_layers = int(profile["num_layers"])
            dropout = float(profile["dropout"])

            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, tab_hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tab_hid, hidden_dim),
                nn.GELU(),
            )

            self.temporal_projection = nn.Linear(temporal_input_dim, temporal_hid)
            attn_layer = nn.TransformerEncoderLayer(
                d_model=temporal_hid,
                nhead=max(1, n_heads),
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal_encoder = nn.TransformerEncoder(attn_layer, num_layers=max(1, n_layers))
            self.temporal_to_hidden = nn.Linear(temporal_hid, hidden_dim)

            self.cross_attention_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.early_fusion = nn.Sequential(
                nn.Linear(tabular_dim + temporal_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

            self.gated_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

            self.tabular_mean_head = nn.Linear(hidden_dim, target_dim)
            self.temporal_mean_head = nn.Linear(hidden_dim, target_dim)
            self.tabular_logvar_head = nn.Linear(hidden_dim, target_dim)
            self.temporal_logvar_head = nn.Linear(hidden_dim, target_dim)

            self.mean_head = nn.Linear(hidden_dim, target_dim)
            self.logvar_head = nn.Linear(hidden_dim, target_dim)

            self.temporal_steps = int(max(1, temporal_steps))

        def forward(self, tabular_x, temporal_x):
            tab_z = self.tabular_encoder(tabular_x)
            temporal_tokens = self.temporal_projection(temporal_x)
            temporal_tokens = self.temporal_encoder(temporal_tokens)
            temporal_pooled = temporal_tokens.mean(dim=1)
            temporal_z = self.temporal_to_hidden(temporal_pooled)

            mode = self.fusion_mode
            if mode == "early":
                early_in = torch.cat([tabular_x, temporal_x.mean(dim=1)], dim=1)
                fused = self.early_fusion(early_in)
                mean = self.mean_head(fused)
                logvar = self.logvar_head(fused)
            elif mode == "late":
                tab_mean = self.tabular_mean_head(tab_z)
                tab_logvar = self.tabular_logvar_head(tab_z)
                temp_mean = self.temporal_mean_head(temporal_z)
                temp_logvar = self.temporal_logvar_head(temporal_z)
                mean = 0.5 * (tab_mean + temp_mean)
                logvar = 0.5 * (tab_logvar + temp_logvar)
            elif mode == "gated":
                joint = torch.cat([tab_z, temporal_z], dim=1)
                gate = self.gate(joint)
                fused_branch = self.gated_fusion(joint)
                gated = gate * tab_z + (1.0 - gate) * temporal_z
                fused = 0.5 * (gated + fused_branch)
                mean = self.mean_head(fused)
                logvar = self.logvar_head(fused)
            else:
                fused = self.cross_attention_fusion(torch.cat([tab_z, temporal_z], dim=1))
                mean = self.mean_head(fused)
                logvar = self.logvar_head(fused)
            return mean, logvar

    model = FusionBlock()

    if pretrained_checkpoint_path:
        ckpt = torch.load(pretrained_checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)

    return model
