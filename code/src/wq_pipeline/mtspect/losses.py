from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    sq = (y - mu) ** 2
    loss = 0.5 * inv_var * sq + 0.5 * logvar
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def masked_huber(mu: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss = F.huber_loss(mu, y, reduction="none", delta=delta)
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def alignment_infonce(z_multi: torch.Tensor, z_text: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z_m = F.normalize(z_multi, dim=-1)
    z_t = F.normalize(z_text, dim=-1)
    logits = (z_m @ z_t.T) / max(temperature, 1e-6)
    labels = torch.arange(z_m.size(0), device=z_m.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def route_regularizer(mix_add: torch.Tensor, mix_xattn: torch.Tensor, alpha_v: torch.Tensor, alpha_t: torch.Tensor, alpha_l: torch.Tensor) -> torch.Tensor:
    mix = torch.stack([mix_add, mix_xattn], dim=1).clamp(min=1e-6)
    uniform = torch.full_like(mix, 0.5)
    kl = (mix * (mix.log() - uniform.log())).sum(dim=1).mean()

    gates = torch.stack([alpha_v, alpha_t, alpha_l], dim=1).clamp(min=1e-6, max=1 - 1e-6)
    entropy = -(gates * gates.log() + (1.0 - gates) * (1.0 - gates).log()).mean()
    return kl + (0.3 - entropy).relu()


def physics_penalty(mu: torch.Tensor, min_value: float = 0.0) -> torch.Tensor:
    return torch.relu(min_value - mu).mean()


def domain_adversarial_loss(domain_logits: torch.Tensor | None, domain_targets: torch.Tensor | None) -> torch.Tensor:
    if domain_logits is None or domain_targets is None:
        return torch.tensor(0.0, device=domain_logits.device if domain_logits is not None else "cpu")
    return F.cross_entropy(domain_logits, domain_targets)
