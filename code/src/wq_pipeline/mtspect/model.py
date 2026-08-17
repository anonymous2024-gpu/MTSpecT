from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .image_backbones import NPZSpectralBackbone, RGBBackbone, SpectralBandBridge


@dataclass
class MTSpectConfig:
    tabular_dim: int
    temporal_band_dim: int
    quality_dim: int
    n_targets: int
    vocab_size: int
    text_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_model_trainable: bool = False
    text_max_len: int = 64
    hidden_dim: int = 256
    text_dim: int = 128
    temporal_heads: int = 4
    temporal_layers: int = 2
    dropout: float = 0.1
    covariance_rank: int = 0
    vision_model_name: str = "resnet18"
    vision_pretrained: bool = True
    vision_rgb_model: str = "qwen_vl"
    vision_rgb_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vision_ms_model: str = "spectral_cnn"
    npz_channels: int = 12
    use_vlm_bridge: bool = True
    text_use_lora: bool = False
    text_use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Ablation flags
    enable_modality_tabular: bool = True
    enable_modality_temporal: bool = True
    enable_modality_rgb: bool = True
    enable_modality_npz: bool = True
    enable_modality_text: bool = True
    enable_quality_gates: bool = True
    enable_fusion_mode: str = "concat"
    enable_uncertainty_head: bool = True
    enable_cross_attn_expert: bool = True
    enable_spectral_adapter: bool = True   # If False, replace SSA with linear projection
    enable_band_bridge: bool = True        # If False, disable SpectralBandBridge (NPZ→RGB path for VLM)
    enable_tabular_skip: bool = True       # If False, remove direct tabular→output skip connection


class SpectralAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalQualityTransformer(nn.Module):
    def __init__(self, hidden_dim: int, quality_dim: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.quality_gate = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, n_heads),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, n_layers))
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        gates = self.quality_gate(quality)
        gated = tokens * gates
        out = self.encoder(gated)
        return self.out_norm(out)


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HFTextEncoder(nn.Module):
    _BACKBONE_CACHE: dict[str, nn.Module] = {}

    def __init__(
        self,
        model_id: str,
        hidden_dim: int,
        trainable: bool = False,
        dropout: float = 0.1,
        use_lora: bool = False,
        use_qlora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("transformers is required for HF text encoder.") from exc

        cache_key = str(model_id)
        use_cache = (not bool(trainable)) and (not bool(use_lora)) and (not bool(use_qlora))

        if use_cache and cache_key in self._BACKBONE_CACHE:
            self.backbone = self._BACKBONE_CACHE[cache_key]
        else:
            load_kwargs = {}
            if bool(use_qlora):
                load_kwargs = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    ),
                    "device_map": "auto",
                }
            try:
                self.backbone = AutoModel.from_pretrained(model_id, **load_kwargs)
            except Exception:
                self.backbone = AutoModel.from_pretrained(model_id)
                use_qlora = False

            if use_cache:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()
                self._BACKBONE_CACHE[cache_key] = self.backbone

        if bool(use_lora) or bool(use_qlora):
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError:
                for param in self.backbone.parameters():
                    param.requires_grad = bool(trainable)
            else:
                lora_cfg = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=int(lora_r),
                    lora_alpha=int(lora_alpha),
                    lora_dropout=float(lora_dropout),
                    bias="none",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                )
                self.backbone = get_peft_model(self.backbone, lora_cfg)
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = "lora_" in name
        else:
            for param in self.backbone.parameters():
                param.requires_grad = bool(trainable)
        text_out_dim = int(getattr(self.backbone.config, "hidden_size", hidden_dim))
        self.proj = nn.Sequential(
            nn.Linear(text_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=token_ids, attention_mask=token_mask.long())
        encoded = out.last_hidden_state
        weights = token_mask.unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / (weights.sum(dim=1).clamp_min(1.0))
        return self.proj(pooled)


class QwenVisionLanguageBridge(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=max(1, n_heads), batch_first=True, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, z_txt: torch.Tensor, z_rgb: torch.Tensor, z_spec_rgb: torch.Tensor) -> torch.Tensor:
        q = z_txt.unsqueeze(1)
        kv = torch.stack([z_rgb, z_spec_rgb], dim=1)
        out, _ = self.cross_attn(q, kv, kv)
        return self.out(out.squeeze(1))


class ReliabilityFusion(nn.Module):
    def __init__(self, hidden_dim: int, quality_dim: int, n_heads: int, dropout: float, enable_cross_attn_expert: bool = True) -> None:
        super().__init__()
        self.enable_cross_attn_expert = enable_cross_attn_expert
        self.v_score = nn.Linear(hidden_dim + quality_dim, 1)
        self.t_score = nn.Linear(hidden_dim + quality_dim, 1)
        self.l_score = nn.Linear(hidden_dim + quality_dim, 1)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=max(1, n_heads), batch_first=True, dropout=dropout)
        self.expert_mix = nn.Linear(hidden_dim * 2 + quality_dim, 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_v_seq: torch.Tensor,
        z_rgb: torch.Tensor,
        z_npz: torch.Tensor,
        z_tab: torch.Tensor,
        z_txt: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        q_agg = quality.mean(dim=1)
        z_v = z_v_seq.mean(dim=1)
        z_img = 0.5 * (z_rgb + z_npz)
        z_v = 0.5 * (z_v + z_img)

        v_in = torch.cat([z_v, q_agg], dim=1)
        t_in = torch.cat([z_tab, q_agg], dim=1)
        l_in = torch.cat([z_txt, q_agg], dim=1)
        a_v = torch.sigmoid(self.v_score(v_in))
        a_t = torch.sigmoid(self.t_score(t_in))
        a_l = torch.sigmoid(self.l_score(l_in))

        e_add = a_v * z_v + a_t * z_tab + a_l * z_txt

        if self.enable_cross_attn_expert:
            kv = torch.cat([z_v_seq, z_rgb.unsqueeze(1), z_npz.unsqueeze(1), z_txt.unsqueeze(1)], dim=1)
            q = z_tab.unsqueeze(1)
            e_xattn, attn_weights = self.cross_attn(q, kv, kv)
            e_x = e_xattn.squeeze(1)
            mix_logits = self.expert_mix(torch.cat([e_add, e_x, q_agg], dim=1))
            mix = torch.softmax(mix_logits, dim=1)
            z_fuse = mix[:, :1] * e_add + mix[:, 1:2] * e_x
        else:
            attn_weights = None
            mix = torch.zeros(e_add.shape[0], 2, device=e_add.device)
            mix[:, 0] = 1.0
            z_fuse = e_add
        z_fuse = self.out(z_fuse)

        aux = {
            "alpha_v": a_v.squeeze(-1),
            "alpha_t": a_t.squeeze(-1),
            "alpha_l": a_l.squeeze(-1),
            "mix_add": mix[:, 0],
            "mix_xattn": mix[:, 1],
            "attn_weights": attn_weights.mean(dim=1) if (attn_weights is not None and attn_weights.ndim == 3) else attn_weights,
        }
        return z_fuse, aux


class ProbabilisticHead(nn.Module):
    def __init__(self, hidden_dim: int, n_targets: int, covariance_rank: int = 0) -> None:
        super().__init__()
        self.mu = nn.Linear(hidden_dim, n_targets)
        self.logvar = nn.Linear(hidden_dim, n_targets)
        self.covariance_rank = int(max(0, covariance_rank))
        self.low_rank = nn.Linear(hidden_dim, n_targets * self.covariance_rank) if self.covariance_rank > 0 else None
        self.n_targets = int(n_targets)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mu = self.mu(z)
        logvar = self.logvar(z).clamp(min=-8.0, max=8.0)
        if self.low_rank is None:
            return mu, logvar, None
        u = self.low_rank(z)
        u = u.view(z.shape[0], self.n_targets, self.covariance_rank)
        return mu, logvar, u

    def deterministic_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass that returns only mean predictions."""
        return self.mu(z)


class MTSpectModel(nn.Module):
    def __init__(self, cfg: MTSpectConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ssa = SpectralAdapter(cfg.temporal_band_dim, cfg.hidden_dim, cfg.dropout)
        self.ssa_linear = nn.Linear(cfg.temporal_band_dim, cfg.hidden_dim)  # fallback when enable_spectral_adapter=False
        self.rgb_backbone = RGBBackbone(
            model_name=cfg.vision_model_name,
            hidden_dim=cfg.hidden_dim,
            pretrained=cfg.vision_pretrained,
            backend=cfg.vision_rgb_model,
            vlm_model_id=cfg.vision_rgb_model_id,
        )
        self.spectral_bridge = SpectralBandBridge(in_channels=cfg.npz_channels)
        self.spectral_rgb_backbone = self.rgb_backbone
        self.npz_backbone = NPZSpectralBackbone(in_channels=cfg.npz_channels, hidden_dim=cfg.hidden_dim)
        self.temporal = TemporalQualityTransformer(
            hidden_dim=cfg.hidden_dim,
            quality_dim=cfg.quality_dim,
            n_heads=cfg.temporal_heads,
            n_layers=cfg.temporal_layers,
            dropout=cfg.dropout,
        )
        self.tabular = TabularEncoder(cfg.tabular_dim, cfg.hidden_dim, cfg.dropout)
        self.text = HFTextEncoder(
            model_id=cfg.text_model_id,
            hidden_dim=cfg.hidden_dim,
            trainable=cfg.text_model_trainable,
            dropout=cfg.dropout,
            use_lora=cfg.text_use_lora,
            use_qlora=cfg.text_use_qlora,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )
        self.vlm_bridge = QwenVisionLanguageBridge(cfg.hidden_dim, cfg.temporal_heads, cfg.dropout)
        self.fusion = ReliabilityFusion(cfg.hidden_dim, cfg.quality_dim, cfg.temporal_heads, cfg.dropout,
                                        enable_cross_attn_expert=cfg.enable_cross_attn_expert)
        self.head = ProbabilisticHead(cfg.hidden_dim, cfg.n_targets, cfg.covariance_rank)
        self.tabular_skip = nn.Linear(cfg.tabular_dim, cfg.n_targets)
        nn.init.zeros_(self.tabular_skip.weight)
        nn.init.zeros_(self.tabular_skip.bias)

    def forward(
        self,
        tabular_x: torch.Tensor,
        temporal_x: torch.Tensor,
        quality_x: torch.Tensor,
        rgb_x: torch.Tensor,
        npz_x: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Process modalities conditionally based on ablation flags
        device = tabular_x.device

        # Tabular modality
        if self.cfg.enable_modality_tabular:
            z_tab = self.tabular(tabular_x)
        else:
            z_tab = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)

        # Temporal modality
        if self.cfg.enable_modality_temporal:
            v = self.ssa(temporal_x) if self.cfg.enable_spectral_adapter else self.ssa_linear(temporal_x)
            z_v_seq = self.temporal(v, quality_x if self.cfg.enable_quality_gates else torch.ones_like(quality_x))
        else:
            z_v_seq = torch.zeros(tabular_x.shape[0], 1, self.cfg.hidden_dim, device=device)

        # RGB modality
        z_rgb_raw = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)  # Initialize for VLM bridge
        z_spec_rgb = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)  # Initialize for VLM bridge
        if self.cfg.enable_modality_rgb:
            z_rgb_raw = self.rgb_backbone(rgb_x)
            if self.cfg.enable_modality_npz:
                if self.cfg.enable_band_bridge:
                    npz_rgb = self.spectral_bridge(npz_x)
                    z_spec_rgb = self.spectral_rgb_backbone(npz_rgb)
                else:
                    z_spec_rgb = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)
                z_rgb = (z_rgb_raw + z_spec_rgb) * 0.5
            else:
                z_rgb = z_rgb_raw
        else:
            z_rgb = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)

        # NPZ modality (spectral)
        if self.cfg.enable_modality_npz:
            z_npz = self.npz_backbone(npz_x)
        else:
            z_npz = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)

        # Text modality
        if self.cfg.enable_modality_text:
            z_txt = self.text(text_tokens, text_mask)
            if self.cfg.use_vlm_bridge and self.cfg.enable_modality_rgb:
                z_spec_rgb_for_vlm = z_spec_rgb if self.cfg.enable_modality_npz else torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)
                z_txt = 0.5 * (z_txt + self.vlm_bridge(z_txt, z_rgb_raw, z_spec_rgb_for_vlm))
        else:
            z_txt = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)

        # Fusion
        z_fuse, aux = self.fusion(z_v_seq, z_rgb, z_npz, z_tab, z_txt, quality_x if self.cfg.enable_quality_gates else torch.ones_like(quality_x))

        # Apply fusion mode ablation
        if self.cfg.enable_fusion_mode == "sum":
            # Sum all enabled modalities
            enabled_embeddings = []
            if self.cfg.enable_modality_temporal:
                enabled_embeddings.append(z_v_seq.mean(dim=1))
            if self.cfg.enable_modality_rgb:
                enabled_embeddings.append(z_rgb)
            if self.cfg.enable_modality_npz:
                enabled_embeddings.append(z_npz)
            if self.cfg.enable_modality_tabular:
                enabled_embeddings.append(z_tab)
            if self.cfg.enable_modality_text:
                enabled_embeddings.append(z_txt)
            if enabled_embeddings:
                z_fuse = torch.stack(enabled_embeddings, dim=0).sum(dim=0)
            else:
                z_fuse = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)
        elif self.cfg.enable_fusion_mode == "mean":
            # Mean all enabled modalities
            enabled_embeddings = []
            if self.cfg.enable_modality_temporal:
                enabled_embeddings.append(z_v_seq.mean(dim=1))
            if self.cfg.enable_modality_rgb:
                enabled_embeddings.append(z_rgb)
            if self.cfg.enable_modality_npz:
                enabled_embeddings.append(z_npz)
            if self.cfg.enable_modality_tabular:
                enabled_embeddings.append(z_tab)
            if self.cfg.enable_modality_text:
                enabled_embeddings.append(z_txt)
            if enabled_embeddings:
                z_fuse = torch.stack(enabled_embeddings, dim=0).mean(dim=0)
            else:
                z_fuse = torch.zeros(tabular_x.shape[0], self.cfg.hidden_dim, device=device)
        # else: "concat" mode (default) - already handled by fusion module

        # Head
        if self.cfg.enable_uncertainty_head:
            mu, logvar, low_rank_u = self.head(z_fuse)
        else:
            # Deterministic head - use mean only
            mu = self.head.deterministic_forward(z_fuse)
            logvar = torch.zeros_like(mu)
            low_rank_u = None

        if self.cfg.enable_tabular_skip:
            mu = mu + self.tabular_skip(tabular_x)
        out = {
            "mu": mu,
            "logvar": logvar,
            "z_fuse": z_fuse,
            "z_text": z_txt,
            "z_vision": z_v_seq.mean(dim=1),
            "z_rgb": z_rgb,
            "z_spec_rgb": z_spec_rgb,
            "z_npz": z_npz,
            "z_tabular": z_tab,
            **{k: v for k, v in aux.items()},
        }
        if low_rank_u is not None:
            out["low_rank_u"] = low_rank_u
        return out
