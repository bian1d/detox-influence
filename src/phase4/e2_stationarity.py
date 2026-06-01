"""E2: first-order non-stationarity ||G(theta*)|| and the natural-gradient step.

G(theta) = grad_phi J = E_{x, y~pi_theta}[ s * (R_tilde - beta) ].  At a true
optimum of J, G = 0 and the IFT yielding the paper's IF formula applies.
theta* = step_0650 is a Pareto checkpoint, not a stationary point, so G != 0.

Estimating ||G||^2 in the 2.36M-dim phi subspace
------------------------------------------------
G is a sample mean of per-prompt mean score-contributions
``gbar_p = (1/K) sum_{i in p} s_i * w_i`` (E_{y|x}[s]=0 makes any
per-prompt-constant baseline drop out, so G = E_x[Cov_{y|x}(s, R_tilde)]).
The naive ||G_hat||^2 carries a +tr(Cov)/N self-product bias that, in 2.36M
dims, can dominate (CLAUDE.md Hard Constraint 8).

We use the **all-pairs U-statistic** — the exactly-unbiased estimator and the
limit of averaging over all disjoint-half splits Hard Constraint 8 prescribes:

    ||G||^2  =  (1/(N(N-1))) [ ||S||^2 - sum_p ||gbar_p||^2 ],   S = sum_p gbar_p

and likewise G^T M G with ||.||^2 -> the M-quadratic form.  The CI is a
**jackknife over prompts** (leave-one-prompt-out), which propagates the
between-prompt sampling variance that a split-*choice* bootstrap misses.  (A
sample-level split-half is kept as a cross-check.)

Two baselines for the scalar weight w_i, both UNBIASED for the same G:
  * primary: (R_tilde - beta), exactly per spec — but the large between-prompt
    R_tilde level (~1.9) multiplies the per-prompt score-mean noise, inflating
    the estimator variance;
  * loo_advantage: A_i^LOO = R_tilde_i - mean_{j!=i, same prompt} R_tilde_j,
    which removes that level and targets Cov_{y|x}(s,R_tilde) directly -> much
    lower variance.  This is the trustworthy estimate; agreement of the two
    (within CI) is the check.

0.5 * G^T F^-1 G is the KL the policy moves on one unit natural-gradient/Newton
step toward the phi-restricted stationary point; compared to the training KL it
says whether the residual non-stationarity is small.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ekfac.eigen import inverse_hvp
from phase4.grad_phi import score_logpi_phi
from phase4.sampling import OnPolicySample


def loo_prompt_advantage(pool: list[OnPolicySample]) -> np.ndarray:
    """A_i^LOO = R_tilde_i - mean_{j != i in same prompt} R_tilde_j."""
    by_prompt: dict[int, list[int]] = defaultdict(list)
    for idx, s in enumerate(pool):
        by_prompt[s.prompt_idx].append(idx)
    w = np.zeros(len(pool), dtype=np.float64)
    for idxs in by_prompt.values():
        Rs = np.array([pool[i].R_tilde for i in idxs], dtype=np.float64)
        K = len(Rs)
        tot = Rs.sum()
        for k, i in enumerate(idxs):
            loo_mean = (tot - Rs[k]) / (K - 1) if K > 1 else 0.0
            w[i] = Rs[k] - loo_mean
    return w


def accumulate_prompt_means(
    pool: list[OnPolicySample],
    model: torch.nn.Module,
    weight: torch.Tensor,
    device: str,
    weight_sets: dict[str, np.ndarray],
    *,
    store_dtype: torch.dtype = torch.float32,
    log_every: int = 256,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """ONE streaming score pass.  Returns, per baseline, a (N_prompt, d_out,
    d_in) tensor of per-prompt MEAN score-contributions gbar_p, and the list of
    prompt ids in row order.

    s_i is computed once and fanned out to every baseline (only the scalar
    weight differs), halving backward cost versus a pass per baseline.
    """
    by_prompt: dict[int, list[int]] = defaultdict(list)
    for idx, s in enumerate(pool):
        by_prompt[s.prompt_idx].append(idx)
    prompt_ids = sorted(by_prompt)
    row_of = {p: r for r, p in enumerate(prompt_ids)}
    N_p = len(prompt_ids)
    d_out, d_in = weight.shape

    sums = {k: torch.zeros(N_p, d_out, d_in, dtype=store_dtype, device=device)
            for k in weight_sets}
    counts = torch.zeros(N_p, dtype=torch.float64, device=device)

    for i, s in enumerate(pool):
        s_i = score_logpi_phi(model, weight, s.prompt_ids, s.response_ids, device)
        r = row_of[s.prompt_idx]
        counts[r] += 1
        for k, wv in weight_sets.items():
            sums[k][r] += (s_i * float(wv[i])).to(store_dtype)
        if (i + 1) % log_every == 0:
            print(f"    score pass {i + 1}/{len(pool)}")

    gbar = {}
    cinv = (1.0 / counts).reshape(N_p, 1, 1)
    for k in weight_sets:
        gbar[k] = sums[k].to(torch.float64) * cinv          # mean over prompt's K
    return gbar, prompt_ids


# --------------------------------------------------------------------------- #
# Unbiased U-statistic quadratic form + jackknife CI
# --------------------------------------------------------------------------- #
def _flat(gbar: torch.Tensor) -> torch.Tensor:
    """(N, d_out, d_in) -> (N, D) fp64."""
    return gbar.reshape(gbar.shape[0], -1).to(torch.float64)


def ustat_quadratic(gbar: torch.Tensor, Mg: torch.Tensor | None = None) -> float:
    """All-pairs U-statistic for G^T M G (M=identity if Mg is None).

    ``Mg`` is M applied to each gbar_p, shape (N, d_out, d_in); for the
    identity Mg = gbar.  Uses
        sum_{p!=q} <gbar_p, M gbar_q> = <S, M_S> - sum_p <gbar_p, M gbar_p>
    with S = sum_p gbar_p, divided by N(N-1).
    """
    G = _flat(gbar)
    MG = G if Mg is None else _flat(Mg)
    N = G.shape[0]
    S = G.sum(0)
    S_M = MG.sum(0)
    cross = float(torch.dot(S, S_M).item())
    diag = float((G * MG).sum().item())                     # sum_p <gbar_p, M gbar_p>
    return (cross - diag) / (N * (N - 1))


def jackknife_ci(gbar: torch.Tensor, Mg: torch.Tensor | None = None) -> dict:
    """Leave-one-prompt-out jackknife SE + 95% normal CI for the U-statistic.

    U_(-q) recomputed in closed form from S, the cross terms <S, M gbar_q>, and
    the diagonal terms <gbar_q, M gbar_q>."""
    G = _flat(gbar)
    MG = G if Mg is None else _flat(Mg)
    N = G.shape[0]
    S = G.sum(0)
    S_M = MG.sum(0)
    cross_all = float(torch.dot(S, S_M).item())
    diag_each = (G * MG).sum(1)                              # (N,) <gbar_q, M gbar_q>
    diag_all = float(diag_each.sum().item())
    # <S, M gbar_q> and <gbar_q, M S> (M symmetric on the relevant ops; for
    # F^-1 both forms equal; for identity trivially equal). Use both halves.
    SMgq = (G @ S_M)                                        # <gbar_q, M S>  (N,)
    gqMS = (MG @ S)                                         # <M gbar_q, S>  (N,)

    U_full = (cross_all - diag_all) / (N * (N - 1))
    loo = np.empty(N, dtype=np.float64)
    for q in range(N):
        cross_q = cross_all - float(SMgq[q].item()) - float(gqMS[q].item()) + float(diag_each[q].item())
        diag_q = diag_all - float(diag_each[q].item())
        m = N - 1
        loo[q] = (cross_q - diag_q) / (m * (m - 1))
    # Jackknife variance of the U-statistic.
    loo_mean = loo.mean()
    var = (N - 1) / N * float(((loo - loo_mean) ** 2).sum())
    se = float(np.sqrt(max(var, 0.0)))
    return {
        "point": U_full,
        "jackknife_se": se,
        "ci95": [U_full - 1.96 * se, U_full + 1.96 * se],
        "z": U_full / se if se > 0 else float("inf"),
    }


def subsample_se(
    gbar: torch.Tensor, Mg: torch.Tensor | None, point: float, *,
    b_frac: float = 0.5, n_sub: int = 300, seed: int = 1000
) -> dict:
    """WITHOUT-replacement subsampling SE for the degree-2 U-statistic.

    A resample-WITH-replacement bootstrap is INVALID here: duplicated prompts
    create p==q self-pairs that the U-statistic formula does not remove, biasing
    it back toward the naive plug-in.  Subsampling b<N prompts without
    replacement has no duplicate self-pairs.  For a degree-2 U-statistic
    Var(U_N) ~ (b/N) Var(U_b), so SE(U_N) = sqrt(b/N) * SD(U_b over subsamples).
    CI is the normal interval around the full-sample point estimate.
    """
    G = _flat(gbar)
    MG = G if Mg is None else _flat(Mg)
    N = G.shape[0]
    b = max(4, int(round(b_frac * N)))
    rng = np.random.RandomState(seed)
    us = np.empty(n_sub, dtype=np.float64)
    for t in range(n_sub):
        idx = torch.from_numpy(rng.choice(N, size=b, replace=False)).to(G.device)
        Gb = G[idx]; MGb = MG[idx]
        S = Gb.sum(0); S_M = MGb.sum(0)
        us[t] = (float(torch.dot(S, S_M).item()) - float((Gb * MGb).sum().item())) / (b * (b - 1))
    se = float(np.sqrt(b / N) * us.std(ddof=1))
    return {
        "subsample_se": se,
        "ci95": [point - 1.96 * se, point + 1.96 * se],
        "z": point / se if se > 0 else float("inf"),
        "subsample_point_mean": float(us.mean()),   # should track the U-stat point
    }


def naive_quadratic(gbar: torch.Tensor, Mg: torch.Tensor | None = None) -> float:
    """Biased plug-in ||G_hat||^2 (or G_hat^T M G_hat) = quadratic form of the
    full-sample mean — carries the +tr(Cov)/N self term."""
    G = _flat(gbar)
    MG = G if Mg is None else _flat(Mg)
    N = G.shape[0]
    Ghat = G.mean(0)
    MGhat = MG.mean(0)
    return float(torch.dot(Ghat, MGhat).item())


def stationarity_report(
    pool: list[OnPolicySample],
    model: torch.nn.Module,
    weight: torch.Tensor,
    device: str,
    beta: float,
    Q_A: torch.Tensor,
    Q_S: torch.Tensor,
    Lambda: torch.Tensor,
    *,
    damping_floor: float = 1e-5,
    damping_alpha: float = 0.1,
    store_dtype: torch.dtype = torch.float32,
) -> dict:
    weight_sets = {
        "primary_Rtilde_minus_beta": np.array([s.R_tilde for s in pool]) - beta,
        "loo_advantage": loo_prompt_advantage(pool),
    }
    print("  [E2] single shared score pass for all baselines ...")
    gbar, prompt_ids = accumulate_prompt_means(
        pool, model, weight, device, weight_sets, store_dtype=store_dtype
    )

    def apply_ihvp_each(g: torch.Tensor) -> torch.Tensor:
        """F^-1 applied to every prompt's gbar_p; returns (N, d_out, d_in)."""
        out = torch.empty_like(g, dtype=torch.float64)
        for r in range(g.shape[0]):
            out[r] = inverse_hvp(
                g[r].to(torch.float64), Q_A, Q_S, Lambda,
                damping_floor=damping_floor, damping_alpha=damping_alpha,
            )
        return out

    results: dict[str, dict] = {}
    for name in weight_sets:
        g = gbar[name]
        Fi_g = apply_ihvp_each(g)
        # ||G||^2: U-stat point, jackknife SE, subsampling SE cross-check.
        gnorm = jackknife_ci(g, None)
        gnorm_sub = subsample_se(g, None, gnorm["point"])
        gnorm_naive = naive_quadratic(g, None)
        # G^T F^-1 G
        gfig = jackknife_ci(g, Fi_g)
        gfig_sub = subsample_se(g, Fi_g, gfig["point"])
        gfig_naive = naive_quadratic(g, Fi_g)
        gtfig = gfig["point"]
        results[name] = {
            "G_norm_sq_ustat": gnorm["point"],
            "G_norm_sq_jackknife_se": gnorm["jackknife_se"],
            "G_norm_sq_jackknife_ci95": gnorm["ci95"],
            "G_norm_sq_subsample_se": gnorm_sub["subsample_se"],
            "G_norm_sq_subsample_ci95": gnorm_sub["ci95"],
            "G_norm_sq_z_jackknife": gnorm["z"],
            "G_norm_sq_z_subsample": gnorm_sub["z"],
            "G_norm": float(np.sqrt(max(gnorm["point"], 0.0))),
            "G_norm_sq_naive_biased": gnorm_naive,
            "GtFiG_ustat": gtfig,
            "GtFiG_jackknife_se": gfig["jackknife_se"],
            "GtFiG_subsample_se": gfig_sub["subsample_se"],
            "GtFiG_subsample_ci95": gfig_sub["ci95"],
            "GtFiG_z_jackknife": gfig["z"],
            "GtFiG_z_subsample": gfig_sub["z"],
            "GtFiG_naive_biased": gfig_naive,
            "natgrad_step_kl": 0.5 * gtfig if gtfig > 0 else float("nan"),
            "natgrad_norm": float(np.sqrt(max(gtfig, 0.0))),
        }
    results["_meta"] = {
        "n_samples": len(pool),
        "n_prompts": len(prompt_ids),
        "beta": beta,
        "damping_floor": damping_floor,
        "damping_alpha": damping_alpha,
        "estimator": "all-pairs U-statistic, jackknife-over-prompts CI",
    }
    return results
