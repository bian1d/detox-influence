"""Stage 1A: Kronecker factor (A, S) accumulation under pseudo-labels.

For target layer ``W = layer.weight`` of shape ``(d_out, d_in)``::

    A = E_t[m_t m_tᵀ]    shape (d_in,  d_in)
    S = E_t[δ_t δ_tᵀ]    shape (d_out, d_out)

where ``m_t`` is the c_proj input at response-token position ``t`` and
``δ_t`` is the gradient at the c_proj output at position ``t`` under a
summed-loss backward (CONTEXT.md §13.2, §14 Stage-1A reference code).

Convention choices (each backed by a hard constraint):

* **Pseudo-labels from π_θ** (not the recorded response). This realises the
  expectation ``E_{y~π_θ}`` in the Fisher definition (CLAUDE.md C4).
* **Prompt-position mask** via ``ignore_index = -100`` on labels. Forward
  is over the full ``[prompt | response]`` sequence, but the cross-entropy
  contribution and the index range used to slice ``m, δ`` skip prompt
  positions (CLAUDE.md C1, CONTEXT.md §14 difference one).
* **Per-token normalisation**: divide ``A_sum`` and ``S_sum`` by total
  response token count, not by sequence count (MDA Stage 1A; CONTEXT.md
  §14 / §18.6).
* **fp64 accumulation** for numerical stability before eigendecomposition.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekfac.config import EKFACConfig
from ekfac.data import Rollout, build_input_and_labels
from ekfac.hooks import capture_c_proj


@torch.no_grad()
def sample_pseudo_labels(
    logits: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-position multinomial draw from ``softmax(logits)``.

    Args:
        logits: ``(..., V)``.
        generator: optional generator for deterministic sampling.

    Returns:
        Long tensor of shape ``logits.shape[:-1]``.
    """
    head_shape = logits.shape[:-1]
    V = logits.shape[-1]
    probs = torch.softmax(logits.reshape(-1, V).float(), dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return sampled.reshape(head_shape)


def sample_row_aligned_labels(
    logits_row: torch.Tensor,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    *,
    generator: torch.Generator | None,
    ignore_index: int,
) -> torch.Tensor:
    """Row-aligned Fisher pseudo-labels for one sequence.

    The Fisher expectation ``E_{y~π_θ}`` requires that the loss row at position
    ``t`` (which uses ``logits[t]`` to predict the token at ``t+1``) be paired
    with a pseudo-label sampled from *that same row's* conditional
    ``softmax(logits[t])``. After the shift-by-one CE alignment
    (``logits[:-1]`` vs ``labels[1:]``), row ``t`` reads ``labels[t+1]``; so the
    masked label vector must satisfy ``labels[t+1] = pseudo[t]`` for the
    response rows ``t in [P-1, T-2]``, i.e. ``labels[P:T] = pseudo[P-1:T-1]``.

    Using ``pseudo[P:]`` instead (the previous behaviour) is an off-by-one: it
    pairs row ``t`` with a label drawn from ``softmax(logits[t+1])`` — the NEXT
    position's distribution — which is not the Fisher and biases S/Λ
    (see reports/fisher_inverse_independent_audit.md). This helper is the single
    source of truth for the alignment so it cannot drift across call sites.

    Args:
        logits_row: ``(T, V)`` logits for one sequence (``logits[0]``).
        prompt_ids: ``(P,)``.
        response_ids: ``(R,)`` actual recorded response tokens (forward context).
        generator: optional generator for deterministic sampling.
        ignore_index: cross-entropy sentinel for prompt positions.

    Returns:
        ``(P+R,)`` long labels: ``[-100]*P`` then the R row-aligned pseudo tokens.
    """
    P = int(prompt_ids.shape[0])
    T = P + int(response_ids.shape[0])
    pseudo = sample_pseudo_labels(logits_row, generator=generator)  # (T,)
    _, labels = build_input_and_labels(
        prompt_ids,
        response_ids,
        response_labels=pseudo[P - 1 : T - 1],   # row-aligned (NOT pseudo[P:])
        ignore_index=ignore_index,
    )
    return labels


def _logits_from_model_output(out: object) -> torch.Tensor:
    """HuggingFace models return a ``ModelOutput`` wrapper; toy returns the
    tensor directly. Normalise."""
    if isinstance(out, torch.Tensor):
        return out
    logits = getattr(out, "logits", None)
    if logits is None:
        raise TypeError(f"could not extract logits from {type(out).__name__}")
    return logits


def accumulate_AS(
    model: nn.Module,
    layer: nn.Linear,
    rollouts: Iterable[Rollout],
    cfg: EKFACConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate per-token Kronecker factors A, S over pseudo-labeled rollouts.

    One forward + one backward per rollout (batch size 1). The hook captures
    ``m`` and ``δ`` at every position; we slice to response positions before
    accumulating, matching the prompt-mask convention in ``build_input_and_labels``.

    Returns:
        A: ``(d_in,  d_in)``  fp64, symmetrised.
        S: ``(d_out, d_out)`` fp64, symmetrised.
        n_tok: total response tokens contributed (= sum of R_i across rollouts).
    """
    d_out, d_in = layer.weight.shape
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    A_sum = torch.zeros(d_in, d_in, dtype=cfg.dtype_factors, device=device)
    S_sum = torch.zeros(d_out, d_out, dtype=cfg.dtype_factors, device=device)
    n_tok = 0

    was_training = model.training
    model.eval()
    try:
        for rollout in rollouts:
            n_tok += _accumulate_one(
                model, layer, rollout, cfg,
                device=device, A_sum=A_sum, S_sum=S_sum, gen=gen,
                d_in=d_in, d_out=d_out,
            )
    finally:
        if was_training:
            model.train()

    if n_tok == 0:
        raise RuntimeError("no response tokens accumulated")
    A = A_sum / n_tok
    S = S_sum / n_tok
    # Symmetrise to guard eigh against float-roundoff asymmetry.
    A = 0.5 * (A + A.T)
    S = 0.5 * (S + S.T)
    return A, S, n_tok


def _accumulate_one(
    model: nn.Module,
    layer: nn.Linear,
    rollout: Rollout,
    cfg: EKFACConfig,
    *,
    device: torch.device,
    A_sum: torch.Tensor,
    S_sum: torch.Tensor,
    gen: torch.Generator,
    d_in: int,
    d_out: int,
) -> int:
    """Forward + backward one rollout; add response-token outer products."""
    P = rollout.prompt_len
    R = rollout.response_len
    T = P + R

    input_ids = torch.cat([rollout.prompt_ids, rollout.response_ids]).to(device)
    input_ids = input_ids.unsqueeze(0)  # (1, T)

    # Zero stale grads on the model.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    with capture_c_proj(layer) as cache:
        out = model(input_ids)
        logits = _logits_from_model_output(out)  # (1, T, V)

        # Row-aligned Fisher pseudo-labels: row t's label ~ softmax(logits[t]).
        # (Single source of truth for the alignment; see sample_row_aligned_labels.)
        labels = sample_row_aligned_labels(
            logits[0],
            rollout.prompt_ids.to(device),
            rollout.response_ids.to(device),
            generator=gen,
            ignore_index=cfg.ignore_index,
        )

        # Shift-by-one CE: logits[:-1] predict labels[1:].
        shift_logits = logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32))      # (T-1, V)
        shift_labels = labels[1:]                  # (T-1,)
        loss = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index=cfg.ignore_index,
            reduction="sum",                       # per-token normalisation handled outside
        )
        loss.backward()

    if cache.m is None or cache.delta is None:
        raise RuntimeError("hook capture failed: m or delta is None")
    if cache.m.shape != (1, T, d_in):
        raise RuntimeError(f"unexpected m shape {tuple(cache.m.shape)}, want (1, {T}, {d_in})")
    if cache.delta.shape != (1, T, d_out):
        raise RuntimeError(f"unexpected δ shape {tuple(cache.delta.shape)}, want (1, {T}, {d_out})")

    # Response positions in the unshifted m/δ view are [P-1, T-2] (inclusive),
    # matching the shift positions where shift_labels != -100 (rows of m and
    # δ are indexed identically to logits).
    m_resp = cache.m[0, P - 1 : T - 1, :].to(cfg.dtype_factors)
    d_resp = cache.delta[0, P - 1 : T - 1, :].to(cfg.dtype_factors)
    if m_resp.shape[0] != R or d_resp.shape[0] != R:
        raise RuntimeError(
            f"response slice length {m_resp.shape[0]} != R={R} "
            f"(P={P}, T={T})"
        )

    A_sum.add_(m_resp.T @ m_resp)
    S_sum.add_(d_resp.T @ d_resp)
    return R
