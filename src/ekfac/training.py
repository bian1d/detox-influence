"""Stage 3 training + per-sample gradient utilities.

Three things live here:

1. ``set_determinism`` — global state setup so that retraining the toy with
   the same seed produces bit-identical parameters. Used as the determinism
   gate before the leave-one-out loop in tests/test_influence_toy.py.

2. ``train_toy_to_optimum`` — full-batch Adam optimisation on a set of
   rollouts using plain causal-LM cross-entropy with prompt mask and mean
   reduction. The "full batch" choice is deliberate: it removes SGD
   stochasticity so that ``θ*_{-m}`` differs from ``θ*`` only because
   rollout m is missing from the loss sum.

3. ``compute_per_sample_grad`` and ``eval_logprob`` — Stage 3's per-rollout
   gradient ``s_m = ∇_W log π_θ*(y_m | x_m)`` and the scalar
   ``f(θ) = log π_θ(y_eval | x_eval)``. Both apply prompt mask and the
   per-token-mean reduction (CLAUDE.md C1, C3); the gradient is returned
   as the gradient of *log π* (not *−log π*), so that downstream callers
   apply the supervised or KL-RL sign convention at the IF-score site.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekfac.config import EKFACConfig
from ekfac.data import Rollout, build_input_and_labels
from ekfac.factors import _logits_from_model_output


def set_determinism(seed: int) -> None:
    """Set every knob torch exposes for deterministic CPU/GPU runs.

    Must be called *before* model init **and** before every retraining
    iteration in the leave-one-out loop. Tests must verify bit-equivalence
    of two seeded runs before relying on retraining-based ground truth.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only: a few ops (e.g., scatter on some backends) have no
    # deterministic kernel; we never hit them at toy scale on CPU, but
    # raising would break unrelated paths if someone runs this on cuda.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _full_batch_ce_loss(
    model: nn.Module,
    rollouts: Sequence[Rollout],
    cfg: EKFACConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Mean over rollouts of per-rollout per-token-mean NLL.

    Each rollout contributes equal weight (1/N). Within a rollout, response
    tokens are mean-averaged (cross_entropy reduction='mean' with the
    prompt mask). Total scalar loss has gradient w.r.t. θ that is the
    natural mean-of-mean training signal.
    """
    losses = []
    for r in rollouts:
        input_ids, labels = build_input_and_labels(
            r.prompt_ids.to(device), r.response_ids.to(device),
            ignore_index=cfg.ignore_index,
        )
        input_ids = input_ids.unsqueeze(0)
        out = model(input_ids)
        logits = _logits_from_model_output(out)
        shift_logits = logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32))
        shift_labels = labels[1:]
        l = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index=cfg.ignore_index, reduction="mean",
        )
        losses.append(l)
    return torch.stack(losses).mean()


def train_toy_to_optimum(
    model: nn.Module,
    rollouts: Sequence[Rollout],
    cfg: EKFACConfig,
    *,
    n_steps: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Full-batch AdamW training to convergence; returns loss trajectory.

    ``weight_decay > 0`` uses AdamW (decoupled weight decay), which prevents
    the model from reaching the interpolating zero-loss optimum where the
    classical IF approximation breaks down (see
    ``reports/notes/overparameterized_loo_breakdown.md``). Default
    ``weight_decay=0`` reproduces the original Adam behaviour for tests
    that rely on it.

    Caller is responsible for setting determinism state and model init seed
    *before* calling this function (see ``set_determinism``). The optimizer
    is constructed fresh each call so optimizer state doesn't leak between
    retrainings.
    """
    device = device or next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay,
    )
    losses: list[float] = []
    model.train()
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _full_batch_ce_loss(model, rollouts, cfg, device=device)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    return losses


def compute_per_sample_grad(
    model: nn.Module,
    layer: nn.Linear,
    rollout: Rollout,
    cfg: EKFACConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return ``∇_W log π_θ(y_m | x_m)`` for a single rollout.

    Conventions:
        * batch_size = 1 (CLAUDE.md C2/C6).
        * Recorded response labels — *not* pseudo (CLAUDE.md C4: Stage 3
          uses the actual rollout, Stage 1 uses pseudo-labels).
        * Prompt mask via ``ignore_index = -100`` (CLAUDE.md C1).
        * Per-token-mean reduction (CLAUDE.md C3, CONTEXT.md §18.6).
        * Returns the gradient of **log π**, not of **−log π**. Sign of the
          IF score is applied at the influence_score_* site.

    Shape: ``(d_out, d_in)`` matching ``layer.weight``.
    """
    P = rollout.prompt_len
    input_ids, labels = build_input_and_labels(
        rollout.prompt_ids.to(device), rollout.response_ids.to(device),
        ignore_index=cfg.ignore_index,
    )
    input_ids = input_ids.unsqueeze(0)

    if layer.weight.grad is not None:
        layer.weight.grad = None

    out = model(input_ids)
    logits = _logits_from_model_output(out)
    shift_logits = logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32))
    shift_labels = labels[1:]
    nll = F.cross_entropy(
        shift_logits, shift_labels,
        ignore_index=cfg.ignore_index, reduction="mean",
    )
    # ∇_W log π = -∇_W NLL.  cross_entropy(reduction='mean') divides by R
    # automatically, so the resulting gradient is the per-token-mean form.
    (g_neg,) = torch.autograd.grad(nll, layer.weight, retain_graph=False)
    return -g_neg


@torch.no_grad()
def eval_logprob(
    model: nn.Module,
    rollout: Rollout,
    cfg: EKFACConfig,
    *,
    device: torch.device,
) -> float:
    """Return scalar ``f(θ) = log π_θ(y_eval | x_eval)`` with the same
    prompt-mask + per-token-mean convention used everywhere else in the
    pipeline. Used both for the toy LOO ground truth and for any future
    eval objective f."""
    P = rollout.prompt_len
    input_ids, labels = build_input_and_labels(
        rollout.prompt_ids.to(device), rollout.response_ids.to(device),
        ignore_index=cfg.ignore_index,
    )
    input_ids = input_ids.unsqueeze(0)
    out = model(input_ids)
    logits = _logits_from_model_output(out)
    shift_logits = logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32))
    shift_labels = labels[1:]
    nll = F.cross_entropy(
        shift_logits, shift_labels,
        ignore_index=cfg.ignore_index, reduction="mean",
    )
    return float(-nll.item())
