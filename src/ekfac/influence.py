"""Stage 3 influence score.

The IF formula sign depends on the training objective:

* **Supervised CE** (toy LOO test):
  perturb +ε · log π(y_m|x_m); ``dθ*/dε = +F⁻¹ s_m``;
  ``I = +∇fᵀ F⁻¹ s_m``.

* **KL-regularised RL** (real PPO; CONTEXT.md §5):
  perturb +ε · [r − β log(π/π_ref)]; ``dθ*/dε = −F⁻¹ s_m``;
  ``I = −∇fᵀ F⁻¹ s_m``.

The two functions below name the regime explicitly so the call site is
self-documenting and there is no flag-shaped foot-gun. Both accept the
pre-applied IHVP ``p = F⁻¹ · g_eval`` (so this step is done once for many
training rollouts) and the per-sample gradient ``s_m`` (both shape
``(d_out, d_in)``).
"""
from __future__ import annotations

import torch


def influence_score_supervised(s_m: torch.Tensor, p: torch.Tensor) -> float:
    """IF score for a model trained with plain supervised cross-entropy.

    ``I(z_m, f) = +∇f(θ*)ᵀ F⁻¹ s_m = +⟨s_m, p⟩_F`` where ``p = F⁻¹ · g_eval``.

    Positive means including z_m in training **increases** ``f``. Self-IF
    (``f = log π(y_m|x_m)``) is non-negative under this convention.
    """
    if s_m.shape != p.shape:
        raise ValueError(f"shape mismatch: s_m {tuple(s_m.shape)} vs p {tuple(p.shape)}")
    return float((s_m * p).sum().item())


def influence_score_klrl(s_m: torch.Tensor, p: torch.Tensor) -> float:
    """IF score for a model trained with KL-regularised RL (PPO/RLHF).

    ``I(z_m, f) = −∇f(θ*)ᵀ F⁻¹ s_m = −⟨s_m, p⟩_F`` where ``p = F⁻¹ · g_eval``.

    Positive means upweighting z_m's KL-RL contribution **increases** ``f``.
    Self-IF (``f = log π(y_m|x_m)``) is non-positive under this convention:
    upweighting z_m's effective reward includes upweighting its KL penalty,
    which pushes log π(y_m|x_m) down.
    """
    if s_m.shape != p.shape:
        raise ValueError(f"shape mismatch: s_m {tuple(s_m.shape)} vs p {tuple(p.shape)}")
    return -float((s_m * p).sum().item())
