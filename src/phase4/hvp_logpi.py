"""Hessian-vector products of log pi w.r.t. phi, via double backward.

E3's Delta-vector product needs, per on-policy sample,
    s = grad_phi log pi(y|x)                       (first-order score)
    HVP(v) = (grad^2_phi log pi(y|x)) v            (curvature times v)
for one or more directions v (the per-target IHVP outputs p_f).

``score_and_hvps`` computes the forward + first backward ONCE (with
create_graph=True) and reuses that graph to produce the HVP for every v, so a
sample costs 1 forward + 1 first-backward + len(vs) second-backwards rather
than re-forwarding per target.

Conventions match grad_phi.score_logpi_phi exactly (prompt mask, per-token
MEAN reduction, batch=1, sign = +grad of log pi): the s returned here is the
same s_m that went into the production EK-FAC cache, so Delta is built in the
same metric as F.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekfac.data import build_input_and_labels


def _mean_logpi(model: nn.Module, weight: torch.Tensor, prompt_ids, response_ids,
                device, ignore_index: int) -> torch.Tensor:
    input_ids, labels = build_input_and_labels(
        prompt_ids.to(device), response_ids.to(device),
        response_labels=response_ids.to(device), ignore_index=ignore_index,
    )
    out = model(input_ids.unsqueeze(0))
    logits = out.logits if hasattr(out, "logits") else out
    ce_dtype = torch.promote_types(logits.dtype, torch.float32)
    shift_logits = logits[0, :-1].to(ce_dtype)
    shift_labels = labels[1:]
    ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction="mean")
    return -ce                                        # mean log pi (scalar)


def score_and_hvps(
    model: nn.Module,
    weight: torch.Tensor,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    vs: Sequence[torch.Tensor],
    device: str | torch.device,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return (s, [HVP(v) for v in vs]) for one rollout, sharing one graph.

    s = grad_phi log pi  (detached, (d_out,d_in)).
    HVP(v) = grad^2_phi log pi @ v, computed as d(<s, v>)/d phi with the
    second-order graph carried only through s (v is detached).
    """
    logpi = _mean_logpi(model, weight, prompt_ids, response_ids, device, ignore_index)
    (g,) = torch.autograd.grad(logpi, weight, create_graph=True)   # graph retained
    s = g.detach()
    hvps: list[torch.Tensor] = []
    n = len(vs)
    for k, v in enumerate(vs):
        vv = v.detach().to(g.dtype)
        gv = (g * vv).sum()                                        # <s, v>
        (hvp,) = torch.autograd.grad(gv, weight, retain_graph=(k < n - 1))
        hvps.append(hvp.detach())
    return s, hvps


def hvp_logpi_phi(
    model: nn.Module,
    weight: torch.Tensor,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    v: torch.Tensor,
    device: str | torch.device,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Single-direction HVP (thin wrapper over score_and_hvps), for tests."""
    _, hvps = score_and_hvps(model, weight, prompt_ids, response_ids, [v], device,
                             ignore_index=ignore_index)
    return hvps[0]
