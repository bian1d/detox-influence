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


def _sum_logpi_capture(model, layer, prompt_ids, response_ids, device, ignore_index):
    """Forward [prompt|response]; return (logpi_sum scalar with graph, captured
    layer input m (1,T,d_in) detached, captured layer OUTPUT tensor with graph).

    SUM reduction over response tokens (prompt-masked) — this is the genuine
    sequence log pi (= sum of per-token log pi), the object whose Hessian is the
    exact sum of per-token Hessians (no cross-token terms). The captured output
    tensor lets us read per-token output-gradients delta_u = d logpi_sum/d out_u
    (matching accumulate_AS's per-token delta, sum-loss) for the diagonal score
    term, without relying on full_backward_hook firing under autograd.grad."""
    cap: dict = {}

    def fwd(_m, args, out):
        cap["m"] = args[0].detach()
        cap["out"] = out

    h = layer.register_forward_hook(fwd)
    try:
        input_ids, labels = build_input_and_labels(
            prompt_ids.to(device), response_ids.to(device),
            response_labels=response_ids.to(device), ignore_index=ignore_index,
        )
        out = model(input_ids.unsqueeze(0))
        logits = out.logits if hasattr(out, "logits") else out
        ce_dtype = torch.promote_types(logits.dtype, torch.float32)
        shift_logits = logits[0, :-1].to(ce_dtype)
        ce_sum = F.cross_entropy(shift_logits, labels[1:],
                                 ignore_index=ignore_index, reduction="sum")
        logpi_sum = -ce_sum                                   # log pi(y|x), SUM over tokens
    finally:
        h.remove()
    return logpi_sum, cap["m"], cap["out"]


def score_and_hvps_pertoken(
    model: nn.Module,
    layer: nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    vs: Sequence[torch.Tensor],
    device: str | torch.device,
    *,
    ignore_index: int = -100,
) -> tuple[int, list[torch.Tensor]]:
    """Per-token-granularity pieces of the Delta-vector product for one rollout.

    Returns ``(R, [contrib(v) for v in vs])`` where, for each direction v,
        contrib(v) = diag_score(v) + HVP_sum(v)
    with (both summed over the R response-token positions):
      * diag_score(v) = sum_u g_u (g_u . v),  g_u = outer(delta_u, m_u),
        delta_u = d logpi_sum / d (layer output)_u, m_u = layer input_u.
        This is the PER-TOKEN DIAGONAL of the sequence score outer product — it
        drops the cross-token (u != u') terms, exactly as the EK-FAC Fisher
        F_tok does (so Delta and F use the same Kronecker/per-token
        approximation). Computed from captured (m_u, delta_u); no extra backward.
      * HVP_sum(v) = grad^2_phi (sum_t log pi(y_t)) v — the Hessian of the SUM
        log pi (exact = sum of per-token Hessians, no cross terms) applied to v;
        one double-backward per v (SUM reduction, NOT mean).

    Multiply by the sample's advantage A and normalise by TOTAL response tokens
    outside (delta_vp_per_prompt_tok) to match F_tok's per-token normalisation.
    The caller does NOT divide by R here.
    """
    P = int(prompt_ids.shape[0])
    T = P + int(response_ids.shape[0])
    logpi_sum, m_cap, out_tensor = _sum_logpi_capture(
        model, layer, prompt_ids, response_ids, device, ignore_index)

    # delta_u (per-token output grad of log pi, sum reduction) AND the weight
    # score g (for the HVPs), in one create_graph backward.
    delta_full, g = torch.autograd.grad(
        logpi_sum, [out_tensor, layer.weight], create_graph=True)
    m_resp = m_cap[0, P - 1 : T - 1].to(torch.float64)        # (R, d_in)
    d_resp = delta_full.detach()[0, P - 1 : T - 1].to(torch.float64)  # (R, d_out)
    R = int(m_resp.shape[0])

    contribs: list[torch.Tensor] = []
    n = len(vs)
    for k, v in enumerate(vs):
        v64 = v.detach().to(torch.float64)
        # diagonal score term: sum_u (delta_u . (v m_u)) outer(delta_u, m_u)
        c = (d_resp * (m_resp @ v64.T)).sum(dim=1)            # (R,)
        diag = (c[:, None] * d_resp).T @ m_resp               # (d_out, d_in) fp64
        # HVP of the SUM log pi (model-precision), promoted to fp64
        vv = v.detach().to(g.dtype)
        gv = (g * vv).sum()
        (hvp,) = torch.autograd.grad(gv, layer.weight, retain_graph=(k < n - 1))
        contribs.append(diag + hvp.detach().to(torch.float64))
    return R, contribs
