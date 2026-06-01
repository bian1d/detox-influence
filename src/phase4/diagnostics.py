"""Split-half unbiased estimators for high-dimensional norms / quadratic forms
(CLAUDE.md Hard Constraint 8).

Why this exists
---------------
We want quantities like ||G||^2 = G^T G and G^T F^-1 G for G in the
~2.36M-dim phi subspace, where G is only available as a noisy sample mean
G_hat = (1/n) sum_i g_i.  The naive plug-in ||G_hat||^2 carries a
``+ (1/n) tr(Cov(g))`` self-product bias.  In 2.36M dimensions tr(Cov) is
enormous and swamps the true ||G||^2 — the estimate is almost all noise.

The fix: split the n samples into two disjoint halves A, B; the half-means
G_hat_A, G_hat_B are independent and each unbiased for G, so
``<G_hat_A, M G_hat_B>`` is unbiased for ``G^T M G`` with no self-product
term (the cross term of two independent noises averages to 0, unlike a
square which is always positive).  Average over many random splits and give a
bootstrap CI over splits.

These helpers are matrix-shape agnostic: g_i and the accumulators are flat
vectors (callers reshape (d_out,d_in) <-> flat as needed); ``apply_M`` is an
arbitrary linear operator (identity for ||G||^2, F^-1 for the natural-gradient
quadratic form).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def split_half_quadratic(
    half_sum_A: list[torch.Tensor],
    half_sum_B: list[torch.Tensor],
    count_A: list[int],
    count_B: list[int],
    apply_M: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    n_boot: int = 2000,
    boot_seed: int = 1000,
) -> dict:
    """Estimate G^T M G from pre-accumulated per-split half-sums.

    For split r, ``G_hat_A = half_sum_A[r] / count_A[r]`` (likewise B), and
    the per-split estimate is ``<G_hat_A, M G_hat_B>``.  We report the mean
    over splits and a bootstrap CI over the per-split estimates.

    Args:
        half_sum_A/B: per-split summed vectors sum_{i in half} g_i (flat).
        count_A/B:    per-split half sizes.
        apply_M:      linear operator M applied to G_hat_B; None = identity
                      (gives G^T G = ||G||^2).
    Returns dict with per-split estimates, mean, std, and bootstrap 95% CI.
    """
    n_splits = len(half_sum_A)
    estimates = np.empty(n_splits, dtype=np.float64)
    for r in range(n_splits):
        g_a = (half_sum_A[r] / count_A[r]).to(torch.float64)
        g_b = (half_sum_B[r] / count_B[r]).to(torch.float64)
        mg_b = apply_M(g_b) if apply_M is not None else g_b
        estimates[r] = float(torch.dot(g_a.reshape(-1), mg_b.reshape(-1).to(g_a.dtype)).item())

    boot = _bootstrap_ci(estimates, n_boot=n_boot, seed=boot_seed)
    return {
        "n_splits": n_splits,
        "per_split": estimates.tolist(),
        "mean": float(estimates.mean()),
        "std": float(estimates.std(ddof=1)) if n_splits > 1 else float("nan"),
        "ci95": boot,
        # sqrt of the clamped-positive mean, for reporting a "length"/"step".
        "sqrt_mean_pos": float(np.sqrt(max(estimates.mean(), 0.0))),
    }


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    n = values.size
    if n < 2:
        return [float("nan"), float("nan")]
    means = np.array([values[rng.randint(0, n, size=n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def make_split_assignments(
    n_samples: int, n_splits: int, seed: int
) -> np.ndarray:
    """(n_splits, n_samples) bool matrix: True => sample in half A for that
    split, else half B.  Each split is an independent random ~50/50 partition
    (exactly balanced when n_samples is even)."""
    rng = np.random.RandomState(seed)
    assign = np.zeros((n_splits, n_samples), dtype=bool)
    half = n_samples // 2
    for r in range(n_splits):
        perm = rng.permutation(n_samples)
        assign[r, perm[:half]] = True
    return assign
