"""Multi-layer EK-FAC over every MLP linear layer, in one pass per stage.

Generalises the single-layer pipeline (``factors.accumulate_AS``,
``eigen.fit_lambda``, ``training.compute_per_sample_grad``) to a *set* of layers
captured simultaneously, so all of GPT-Neo's 12 MLP blocks (each ``c_fc`` = W_1
and ``c_proj`` = W_2, 24 matrices) are factored / scored with the same number of
forward+backward passes as a single layer. Only MLP linears are in scope —
attention's softmax breaks the Kronecker factorisation (Grosse 2023), so QKVO
are deliberately excluded by the caller's layer dict.

Design invariants (the multi-vs-single consistency gate verifies them):

* **Same row-aligned pseudo-labels as the fixed single-layer path.** Stage-1A
  and Stage-1B both call ``factors.sample_row_aligned_labels`` (the single
  source of truth for the off-by-one fix), with a generator seeded identically
  to the single-layer code (``cfg.seed`` for A/S, ``cfg.seed + 1`` for Λ) and
  drawing once per rollout. The hooks are pure observers, so each layer's
  captured ``(m, δ)`` — hence its A, S, Λ — is bit-identical to running the
  single-layer ``accumulate_AS`` / ``fit_lambda`` on that layer alone.
* **Per-token Fisher window** ``[P-1, T-2]`` per layer, matching single-layer.
* **Scores use the full autograd gradient over the real response** (not the
  per-token window, not pseudo-labels) — identical to
  ``training.compute_per_sample_grad``, just fanned out to every layer weight in
  one backward via ``torch.autograd.grad(nll, [w1, ..., wk])``.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekfac.config import EKFACConfig
from ekfac.data import Rollout, build_input_and_labels
from ekfac.factors import _logits_from_model_output, sample_row_aligned_labels
from ekfac.hooks import capture_mlp_layers


def mlp_layer_dict(model: nn.Module, layer_names: Iterable[str]) -> dict[str, nn.Linear]:
    """Resolve ``name -> nn.Linear`` for the given submodule names."""
    out: dict[str, nn.Linear] = {}
    for name in layer_names:
        mod = model.get_submodule(name)
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"{name} is {type(mod).__name__}, not nn.Linear")
        out[name] = mod
    return out


def gptneo_mlp_layer_names(n_layer: int = 12) -> list[str]:
    """Every MLP linear in GPT-Neo: ``c_fc`` (W_1) and ``c_proj`` (W_2) per block."""
    names: list[str] = []
    for i in range(n_layer):
        names.append(f"transformer.h.{i}.mlp.c_fc")
        names.append(f"transformer.h.{i}.mlp.c_proj")
    return names


# --------------------------------------------------------------------------- #
# Stage 1A: per-layer A, S (one pass, row-aligned pseudo-labels)
# --------------------------------------------------------------------------- #
def accumulate_AS_multi(
    model: nn.Module,
    layers: dict[str, nn.Linear],
    rollouts: Iterable[Rollout],
    cfg: EKFACConfig,
    *,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, int]]:
    """Per-layer Kronecker factors ``A, S`` over all layers in one data pass.

    Returns ``name -> (A (d_in,d_in) fp64, S (d_out,d_out) fp64, n_tok)``.
    """
    A_sum: dict[str, torch.Tensor] = {}
    S_sum: dict[str, torch.Tensor] = {}
    for name, layer in layers.items():
        d_out, d_in = layer.weight.shape
        A_sum[name] = torch.zeros(d_in, d_in, dtype=cfg.dtype_factors, device=device)
        S_sum[name] = torch.zeros(d_out, d_out, dtype=cfg.dtype_factors, device=device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    n_tok = 0

    was_training = model.training
    model.eval()
    try:
        for rollout in rollouts:
            n_tok += _one_pass_factor(
                model, layers, rollout, cfg, device=device, gen=gen,
                A_sum=A_sum, S_sum=S_sum,
            )
    finally:
        if was_training:
            model.train()
    if n_tok == 0:
        raise RuntimeError("no response tokens accumulated (multi)")

    out: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for name in layers:
        A = A_sum[name] / n_tok
        S = S_sum[name] / n_tok
        out[name] = (0.5 * (A + A.T), 0.5 * (S + S.T), n_tok)
    return out


def _one_pass_factor(
    model, layers, rollout, cfg, *, device, gen, A_sum, S_sum,
) -> int:
    P = rollout.prompt_len
    R = rollout.response_len
    T = P + R
    input_ids = torch.cat([rollout.prompt_ids, rollout.response_ids]).to(device).unsqueeze(0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    with capture_mlp_layers(layers) as caches:
        out = model(input_ids)
        logits = _logits_from_model_output(out)
        labels = sample_row_aligned_labels(
            logits[0], rollout.prompt_ids.to(device), rollout.response_ids.to(device),
            generator=gen, ignore_index=cfg.ignore_index,
        )
        loss = F.cross_entropy(
            logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32)), labels[1:],
            ignore_index=cfg.ignore_index, reduction="sum",
        )
        loss.backward()
    for name in layers:
        cache = caches[name]
        if cache.m is None or cache.delta is None:
            raise RuntimeError(f"capture failed for {name}")
        m_resp = cache.m[0, P - 1 : T - 1, :].to(cfg.dtype_factors)
        d_resp = cache.delta[0, P - 1 : T - 1, :].to(cfg.dtype_factors)
        A_sum[name].add_(m_resp.T @ m_resp)
        S_sum[name].add_(d_resp.T @ d_resp)
    return R


# --------------------------------------------------------------------------- #
# Stage 1B: per-layer corrected Λ (one pass, same row-aligned pseudo-labels)
# --------------------------------------------------------------------------- #
def fit_lambda_multi(
    model: nn.Module,
    layers: dict[str, nn.Linear],
    rollouts: Iterable[Rollout],
    cfg: EKFACConfig,
    *,
    Qs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, int]]:
    """Per-layer corrected diagonal Λ (d_in, d_out) in each layer's (Q_A, Q_S).

    ``Qs[name] = (Q_A, Q_S)``. Returns ``name -> (Λ fp64, n_tok)``.
    """
    Lam_sum: dict[str, torch.Tensor] = {}
    Qd: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, layer in layers.items():
        d_out, d_in = layer.weight.shape
        Lam_sum[name] = torch.zeros(d_in, d_out, dtype=cfg.dtype_factors, device=device)
        Q_A, Q_S = Qs[name]
        Qd[name] = (Q_A.to(cfg.dtype_factors), Q_S.to(cfg.dtype_factors))
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    n_tok = 0

    was_training = model.training
    model.eval()
    try:
        for rollout in rollouts:
            n_tok += _one_pass_lambda(
                model, layers, rollout, cfg, device=device, gen=gen, Qd=Qd, Lam_sum=Lam_sum,
            )
    finally:
        if was_training:
            model.train()
    if n_tok == 0:
        raise RuntimeError("no response tokens accumulated in fit_lambda_multi")
    return {name: (Lam_sum[name] / n_tok, n_tok) for name in layers}


def _one_pass_lambda(model, layers, rollout, cfg, *, device, gen, Qd, Lam_sum) -> int:
    P = rollout.prompt_len
    R = rollout.response_len
    T = P + R
    input_ids = torch.cat([rollout.prompt_ids, rollout.response_ids]).to(device).unsqueeze(0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    with capture_mlp_layers(layers) as caches:
        out = model(input_ids)
        logits = _logits_from_model_output(out)
        labels = sample_row_aligned_labels(
            logits[0], rollout.prompt_ids.to(device), rollout.response_ids.to(device),
            generator=gen, ignore_index=cfg.ignore_index,
        )
        loss = F.cross_entropy(
            logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32)), labels[1:],
            ignore_index=cfg.ignore_index, reduction="sum",
        )
        loss.backward()
    for name in layers:
        cache = caches[name]
        m_resp = cache.m[0, P - 1 : T - 1, :].to(cfg.dtype_factors)
        d_resp = cache.delta[0, P - 1 : T - 1, :].to(cfg.dtype_factors)
        Q_A, Q_S = Qd[name]
        a_proj = m_resp @ Q_A
        d_proj = d_resp @ Q_S
        Lam_sum[name].add_(a_proj.pow(2).T @ d_proj.pow(2))
    return R


# --------------------------------------------------------------------------- #
# Stage 3 score: full per-layer gradient of log π(real y | x), one backward
# --------------------------------------------------------------------------- #
def per_sample_grads_multi(
    model: nn.Module,
    layer_weights: dict[str, torch.Tensor],
    rollout: Rollout,
    cfg: EKFACConfig,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """``s_m[name] = ∇_{W_name} log π_θ(y_m | x_m)`` for every layer, one backward.

    Real recorded response labels (CLAUDE.md C4), prompt mask, per-token mean
    reduction (C1/C3), batch=1 (C2). ``torch.autograd.grad`` over the list of
    weights returns each layer's gradient from a single backward — bit-identical
    to calling ``training.compute_per_sample_grad`` per layer. Returns +grad of
    log π (sign convention applied at the influence-score site). fp32.
    """
    names = list(layer_weights)
    weights = [layer_weights[n] for n in names]
    input_ids, labels = build_input_and_labels(
        rollout.prompt_ids.to(device), rollout.response_ids.to(device),
        ignore_index=cfg.ignore_index,
    )
    for w in weights:
        if w.grad is not None:
            w.grad = None
    out = model(input_ids.unsqueeze(0))
    logits = _logits_from_model_output(out)
    nll = F.cross_entropy(
        logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32)), labels[1:],
        ignore_index=cfg.ignore_index, reduction="mean",
    )
    grads = torch.autograd.grad(nll, weights, retain_graph=False)
    return {name: (-g).detach() for name, g in zip(names, grads)}


# --------------------------------------------------------------------------- #
# Stage 3 streaming influence — one cache-free pass, all targets, all layers
# --------------------------------------------------------------------------- #
def influence_streaming(
    model: nn.Module,
    layer_weights: dict[str, torch.Tensor],
    rollouts: Iterable[Rollout],
    targets: dict[str, dict[str, torch.Tensor]],
    cfg: EKFACConfig,
    *,
    device: torch.device,
    sign: float = -1.0,
    log_every: int = 2000,
) -> dict[str, torch.Tensor]:
    """Per-rollout influence for every eval target, in ONE pass, NO disk cache.

    Replaces the 184 GB ``g_scaled_z`` cache: instead of persisting each
    rollout's scaled gradient, we recompute its per-layer score ``s_m`` on the
    fly (one backward, all layers) and immediately reduce against every target's
    precomputed per-layer left-vector.

    ``targets[name][layer] = w`` is the eval-side vector that pairs with the
    score for that layer (e.g. ``p_f = F^-1 g_f`` for the paper IF, or
    ``p_f + q_f`` for the Δ-corrected IF; build it per layer beforehand). The
    influence summed over the block-diagonal MLP subspace is

        I_name[m] = sign * Σ_layer ⟨w_name[layer], s_m[layer]⟩

    with ``sign = -1`` realising the KL-RL convention (``influence_score_klrl``)
    and ``sign = +1`` the supervised convention.

    Args:
        layer_weights: ``name -> weight`` (requires_grad leaves) to score.
        targets: ``target_name -> {layer_name -> left_vector (d_out,d_in)}``.
            Every target must cover exactly ``layer_weights``' keys.
        sign: -1 KL-RL (default), +1 supervised.

    Returns:
        ``target_name -> fp64 tensor (N_rollouts,)`` of influence scores.
    """
    names = list(layer_weights)
    for tname, per_layer in targets.items():
        missing = set(names) - set(per_layer)
        if missing:
            raise ValueError(f"target {tname!r} missing layers {sorted(missing)}")
    # Pre-cast left vectors to fp64 on device.
    tv = {tn: {ln: v.to(device).to(torch.float64) for ln, v in pl.items()}
          for tn, pl in targets.items()}
    rollouts = list(rollouts)
    out = {tn: torch.zeros(len(rollouts), dtype=torch.float64) for tn in targets}

    was_training = model.training
    model.eval()
    try:
        for m, rollout in enumerate(rollouts):
            s_m = per_sample_grads_multi(model, layer_weights, rollout, cfg, device=device)
            s64 = {ln: s.to(torch.float64) for ln, s in s_m.items()}
            for tn in targets:
                acc = 0.0
                for ln in names:
                    acc += float((tv[tn][ln] * s64[ln]).sum().item())
                out[tn][m] = sign * acc
            if log_every and (m + 1) % log_every == 0:
                print(f"    [influence_streaming] {m + 1}/{len(rollouts)}")
    finally:
        if was_training:
            model.train()
    return out
