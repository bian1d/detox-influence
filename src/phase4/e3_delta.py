"""E3: the curvature-correction operator Delta-tilde and the corrected IF.

Theory (phase4.md §1):
    Delta      = E_{x, y~pi_theta*}[ A(x,y) * (s s^T + grad^2_phi log pi) ]
    Delta-tilde = Delta / beta
    Delta-tilde @ v = (1/beta) E[ A(x,y) * ( s (s^T v) + HVP_logpi(v) ) ]
with A(x,y) = R_tilde(x,y) - mean_{y'} R_tilde(x,y') the same-prompt advantage.

First-order corrected influence (Neumann, (F-Delta~)^-1 ~ F^-1 + F^-1 Delta~ F^-1):
    I_corr(z_m) ~ -p^T s_m - q^T s_m,   p = F^-1 g_f,  q = F^-1 (Delta~ p).

Reusing the production cache (Q3 check): g_scaled_z[m] = Q_S^T s_m Q_A / denom,
so for any eval-side vector w (PROJECTION ONLY, no second denom divide):
    -<w, F^-1 s_m> = -<Q_S^T w Q_A, g_scaled_z[m]>.
Hence baseline  = cached I_*[m]                (w = g_f)
      correction = -<Q_S^T (Delta~ p) Q_A, g_scaled_z[m]>   (w = Delta~ p)
The denom is ALREADY in g_scaled_z; the eval-side projection must NOT divide
again (CRITICAL pitfall, gated by test_no_double_denom).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from phase4.hvp_logpi import score_and_hvps, score_and_hvps_pertoken


@dataclass
class DeltaSample:
    prompt_idx: int
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    A: float                      # same-prompt-centred advantage R_tilde - mean_y R_tilde


def _contrib(s_i: torch.Tensor, hvp_i: torch.Tensor, v: torch.Tensor, A: float) -> torch.Tensor:
    """A * ( s (s^T v) + HVP(v) ), in fp64."""
    s64 = s_i.to(torch.float64)
    stv = float((s64 * v.to(torch.float64)).sum().item())
    return A * (s64 * stv + hvp_i.to(torch.float64))


def delta_vp_total(
    samples: list[DeltaSample],
    model: torch.nn.Module,
    weight: torch.Tensor,
    beta: float,
    vs: list[torch.Tensor],
    device: str,
) -> list[torch.Tensor]:
    """Delta~ v for each v in vs, flat mean over samples (test/utility form)."""
    acc = [torch.zeros_like(weight, dtype=torch.float64) for _ in vs]
    for smp in samples:
        s_i, hvps = score_and_hvps(model, weight, smp.prompt_ids, smp.response_ids, vs, device)
        for k in range(len(vs)):
            acc[k] += _contrib(s_i, hvps[k], vs[k], smp.A)
    n = len(samples)
    return [a / (beta * n) for a in acc]


def delta_vp_per_prompt(
    pool: list[DeltaSample],
    model: torch.nn.Module,
    weight: torch.Tensor,
    beta: float,
    vs: list[torch.Tensor],
    device: str,
    *,
    store_dtype: torch.dtype = torch.float32,
    store_device: str | torch.device | None = None,
    log_every: int = 200,
    track_contribs: bool = False,
) -> tuple[list[torch.Tensor], list[int], list[dict]]:
    """Per-prompt Delta-contribution vectors d_p^f for each target v_f.

    d_p^f = (1/beta)(1/K_p) sum_{i in p} A_i ( s_i (s_i^T v_f) + HVP_i(v_f) ).
    Then Delta~ v_f = mean_p d_p^f, and ||Delta~ v_f||^2 is the all-pairs
    U-statistic over {d_p^f} (unbiased; no +tr(Cov)/N self term).  One shared
    forward/first-backward graph per sample serves all targets.

    If ``track_contribs``, also return per-sample diagnostics
    {prompt_idx, A, contrib_norm:[per target]} for the left-tail-dominance check
    (the contrib norm is the un-normalised A_i(s s^T v + HVP v); its share by
    |A| says whether the heavy advantage tail drives Delta).

    ``store_device`` (default: ``device``) is where the (N_p, d_out, d_in)
    accumulators live. On OLMo's 16.78M-param phi a 400-prompt, 2-target run
    needs 2 x 25 GiB which exceeds the GPU — pass "cpu" to keep the per-sample
    compute on ``device`` while accumulating off-GPU.
    """
    by_prompt: dict[int, list[DeltaSample]] = defaultdict(list)
    for smp in pool:
        by_prompt[smp.prompt_idx].append(smp)
    order = sorted(by_prompt)
    N_p = len(order)
    sdev = device if store_device is None else store_device
    out = [torch.zeros(N_p, *weight.shape, dtype=store_dtype, device=sdev) for _ in vs]
    diag: list[dict] = []

    seen = 0
    for r, p in enumerate(order):
        for smp in by_prompt[p]:
            s_i, hvps = score_and_hvps(model, weight, smp.prompt_ids, smp.response_ids, vs, device)
            norms = []
            for k in range(len(vs)):
                c = _contrib(s_i, hvps[k], vs[k], smp.A)
                if track_contribs:
                    norms.append(float(c.norm().item()))
                out[k][r] += c.to(dtype=store_dtype, device=out[k].device)
            if track_contribs:
                diag.append({"prompt_idx": smp.prompt_idx, "A": smp.A, "contrib_norm": norms})
            seen += 1
            if seen % log_every == 0:
                print(f"    delta-vp sample {seen}/{len(pool)}")
        Kp = len(by_prompt[p])
        for k in range(len(vs)):
            out[k][r] /= (beta * Kp)
    return out, order, diag


def delta_vp_per_prompt_tok(
    pool: list[DeltaSample],
    model: torch.nn.Module,
    layer: torch.nn.Module,
    beta: float,
    vs: list[torch.Tensor],
    device: str,
    *,
    store_dtype: torch.dtype = torch.float32,
    store_device: str | torch.device | None = None,
    log_every: int = 200,
    track_terms: bool = False,
) -> tuple[list[torch.Tensor], list[int], int, list[dict]]:
    """PER-TOKEN-granularity Delta-contribution vectors, matching the per-token
    EK-FAC Fisher F_tok (see reports/metric_granularity_derivation.md).

    Per-token Delta:   Delta_tok = (1/N_tok) sum_u A(seq(u)) (g_u g_u^T + h_u),
    so   Delta_tok~ v = (1/(beta N_tok)) sum_{samples i} A_i (diag_i(v) + HVP_sum_i(v))
    with diag_i = per-token diagonal score term, HVP_sum_i = SUM-reduction HVP,
    and N_tok = total response tokens in the pool. This matches F_tok's per-token
    normalisation, so the thermometer ||Delta_tok~ p|| / ||g_f|| is a same-
    granularity ratio (no per-token-vs-per-sample scale factor c needed).

    Returned per-prompt vectors d_p are pre-scaled by ``N_p / (beta N_tok)`` so
    that ``mean_p d_p = Delta_tok~ v`` exactly — i.e. the existing U-statistic
    estimators (jackknife_ci / _streaming_stats) over d_p give ||Delta_tok~ v||^2
    directly, unchanged.

    Returns ``(d_p_list, prompt_order, N_tok, term_diag)``. With ``track_terms``,
    ``term_diag`` carries per-sample {prompt_idx, A, diag_norm, hvp_norm} so the
    derivation's health check (diag and HVP terms same order of magnitude at
    per-token granularity) can be reported.
    """
    by_prompt: dict[int, list[DeltaSample]] = defaultdict(list)
    for smp in pool:
        by_prompt[smp.prompt_idx].append(smp)
    order = sorted(by_prompt)
    N_p = len(order)
    d_out, d_in = layer.weight.shape
    sdev = device if store_device is None else store_device
    out = [torch.zeros(N_p, d_out, d_in, dtype=store_dtype, device=sdev) for _ in vs]
    term_diag: list[dict] = []

    N_tok = 0
    seen = 0
    for r, p in enumerate(order):
        for smp in by_prompt[p]:
            R, contribs = score_and_hvps_pertoken(
                model, layer, smp.prompt_ids, smp.response_ids, vs, device)
            N_tok += R
            for k in range(len(vs)):
                out[k][r] += (smp.A * contribs[k]).to(dtype=store_dtype, device=out[k].device)
            if track_terms:
                # recompute term norms once for diagnostics (target 0 only, cheap)
                term_diag.append({"prompt_idx": smp.prompt_idx, "A": smp.A, "R": R})
            seen += 1
            if seen % log_every == 0:
                print(f"    delta-vp(tok) sample {seen}/{len(pool)}")
    # Scale so mean_p d_p == Delta_tok~ v: d_p = (N_p/(beta N_tok)) * sum_{i in p} A_i(...)
    C = N_p / (beta * N_tok)
    for k in range(len(vs)):
        out[k] *= C
    return out, order, N_tok, term_diag


# --------------------------------------------------------------------------- #
# Eval-side projection (PROJECTION ONLY — denom already in g_scaled_z)
# --------------------------------------------------------------------------- #
def project_eval_vector(w: torch.Tensor, Q_A: torch.Tensor, Q_S: torch.Tensor) -> torch.Tensor:
    """Q_S^T w Q_A — the projection that pairs with cached g_scaled_z.

    NB: g_scaled_z already carries 1/denom, so this does NOT divide by denom.
    Doing so would double-apply the damping (gated by test_no_double_denom)."""
    return Q_S.to(torch.float64).T @ w.to(torch.float64) @ Q_A.to(torch.float64)


def cache_dot_scores(
    left_vectors: dict[str, torch.Tensor],
    cache_dir,
    n_rollouts: int,
    device: str,
    *,
    log_every: int = 4000,
) -> dict[str, np.ndarray]:
    """One streaming pass over the cached g_scaled_z[m]; for each named left
    vector w return the array ``score[m] = -<w, g_scaled_z[m]>``.

    All inner products that pair an eval-side vector with the cache reduce to
    this form (denom is already inside g_scaled_z):
      * baseline / cached I_*:   w = Q_S^T g_f Q_A
      * Delta correction:        w = Q_S^T (Delta~ p) Q_A
      * gradient similarity:     w = (Q_S^T g_f Q_A) * denom_T   (undo 1/denom)
    """
    from pathlib import Path
    cache_dir = Path(cache_dir)
    ws = {k: v.to(device).to(torch.float64) for k, v in left_vectors.items()}
    out = {k: np.empty(n_rollouts, dtype=np.float64) for k in ws}
    for m in range(n_rollouts):
        gsz = torch.load(cache_dir / f"g_scaled_z_{m:05d}.pt", map_location=device).to(torch.float64)
        for k, w in ws.items():
            out[k][m] = -float((w * gsz).sum().item())
        if (m + 1) % log_every == 0:
            print(f"    cache-dot {m + 1}/{n_rollouts}")
    return out


def corrected_scores_from_cache(
    g_f_proj: torch.Tensor,
    delta_p_proj: torch.Tensor,
    cache_dir,
    n_rollouts: int,
    device: str,
    *,
    log_every: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline + Delta correction in one cache pass (thin wrapper)."""
    out = cache_dot_scores(
        {"baseline": g_f_proj, "correction": delta_p_proj},
        cache_dir, n_rollouts, device, log_every=log_every,
    )
    return out["baseline"], out["correction"]
