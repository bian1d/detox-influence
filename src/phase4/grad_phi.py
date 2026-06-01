"""Per-sample score in the phi-subspace: s = grad_phi log pi_theta*(y|x).

phi = layer-9 MLP W_2 = ``transformer.h.9.mlp.c_proj.weight``, shape
(d_out, d_in) = (768, 3072).  This is the same orientation EK-FAC uses
(``ekfac.eigen``: layer.weight is (d_out, d_in), A is (d_in, d_in), S is
(d_out, d_out)), so a score produced here drops straight into ``inverse_hvp``
without any transpose.

Three hard constraints (CLAUDE.md 1/2/3) realised here:
  * prompt mask: prompt positions get label -100, so only response tokens
    enter log pi(y|x);
  * batch_size = 1: one rollout, one backward, one score — never a batched
    autograd.grad (which would return sum_i grad_i, unrecoverable per-sample);
  * reduction = 'mean': per-token average over response tokens, matching
    Stage 1/3, so variable-length rollouts are not length-biased.

Sign: cross-entropy is -log pi, so the score is ``s = -grad(CE_mean)``.  The
sign matters for G = E[s*(R_tilde - beta)] (E2) even though it cancels in the
Fisher F = E[s s^T]; we return the +log-pi gradient explicitly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekfac.data import build_input_and_labels


def get_phi_weight(model: nn.Module, layer_name: str = "transformer.h.9.mlp.c_proj") -> nn.Parameter:
    """The (d_out, d_in) weight matrix that is phi.  Leaves its requires_grad
    untouched; callers decide the grad regime."""
    return model.get_submodule(layer_name).weight


def score_logpi_phi(
    model: nn.Module,
    weight: torch.Tensor,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: str | torch.device,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """s = grad_W [ (1/R) sum_{t in response} log pi_theta*(y_t | x, y_<t) ].

    Returns a detached (d_out, d_in) tensor on ``weight``'s device/dtype.
    ``weight`` must be the parameter returned by :func:`get_phi_weight` with
    ``requires_grad=True``; all other params may be frozen (the backward graph
    only needs the path from the loss back to this weight).
    """
    input_ids, labels = build_input_and_labels(
        prompt_ids.to(device),
        response_ids.to(device),
        response_labels=response_ids.to(device),   # real recorded tokens
        ignore_index=ignore_index,
    )
    input_ids = input_ids.unsqueeze(0)              # (1, P+R)

    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out
    # Upcast fp16/bf16 -> fp32 for stable cross-entropy, but never DOWNcast:
    # promote_types(fp32,fp32)=fp32 (production no-op), promote_types(fp64,fp32)
    # =fp64 (keeps toy precision so finite-difference validation resolves).
    ce_dtype = torch.promote_types(logits.dtype, torch.float32)
    shift_logits = logits[0, :-1].to(ce_dtype)       # (P+R-1, V)
    shift_labels = labels[1:]                        # (P+R-1,)
    ce_mean = F.cross_entropy(
        shift_logits, shift_labels,
        ignore_index=ignore_index, reduction="mean",
    )
    (grad,) = torch.autograd.grad(ce_mean, weight, retain_graph=False)
    return (-grad).detach()                          # +grad of mean log pi
