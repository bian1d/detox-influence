"""E1 variance-decomposition correctness.

The decomposition
    Var_y[R_tilde] = Var_y[r] + beta^2 Var_y[KL] - 2 beta Cov_y[r, KL]
is an algebraic identity for sample (co)variance *provided* every term uses
the same ddof.  These tests pin that the implementation realises the identity
exactly on synthetic data with known r, KL (so a future refactor that, say,
switches one term to ddof=0 is caught), and that per-prompt shares sum to 1.

A separate test confirms assert_records_consistent fires when stored R_tilde
does not match reward - beta*KL (the bit-exactness guard against a beta drift
between Phase 1 and Phase 4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase4.e1_advantage import (  # noqa: E402
    assert_records_consistent,
    pooled_advantages,
    variance_decomposition,
)
from phase4.e1_icc_ci import select_icc_subset  # noqa: E402

BETA = 0.23652004338891985


def _make_records(n_prompts: int, k: int, seed: int) -> list[dict]:
    """Synthetic per-prompt records with correlated (r, KL) and the *exact*
    R_tilde = r - beta*KL Phase 1 would have stored."""
    rng = np.random.RandomState(seed)
    records = []
    for p in range(n_prompts):
        # Give each prompt its own correlated (r, KL) Gaussian.
        mean = rng.uniform(-1, 5, size=2)
        a = rng.uniform(-1.5, 1.5, size=(2, 2))
        cov = a @ a.T + np.eye(2) * 0.3
        sample = rng.multivariate_normal(mean, cov, size=k)
        r = sample[:, 0]
        kl = sample[:, 1]
        R = r - BETA * kl
        L = rng.randint(10, 31, size=k).astype(float)
        records.append(
            {
                "prompt_key": p,
                "prompt_text": f"prompt {p}",
                "reward_samples": r.tolist(),
                "k1_kl_samples": kl.tolist(),
                "R_tilde_samples": R.tolist(),
                "response_lens": L.tolist(),
            }
        )
    return records


def test_decomposition_reconstructs_var_exactly():
    records = _make_records(n_prompts=12, k=32, seed=0)
    out = variance_decomposition(records, BETA)
    # The per-prompt reconstruction error is the strongest statement: each
    # prompt's three terms must sum to its Var_y[R_tilde] to float precision.
    assert out["reconstruction_max_abs_err"] < 1e-9, out["reconstruction_max_abs_err"]


def test_per_prompt_shares_sum_to_one():
    records = _make_records(n_prompts=8, k=32, seed=1)
    out = variance_decomposition(records, BETA)
    for p in out["per_prompt"]:
        s = p["share_reward"] + p["share_kl"] + p["share_cov"]
        assert abs(s - 1.0) < 1e-9, (p["prompt_key"], s)


def test_weighted_aggregate_shares_sum_to_one():
    records = _make_records(n_prompts=20, k=32, seed=2)
    agg = variance_decomposition(records, BETA)["aggregate"]
    s = (
        agg["weighted_share_reward"]
        + agg["weighted_share_kl"]
        + agg["weighted_share_cov"]
    )
    assert abs(s - 1.0) < 1e-12, s


def test_pure_reward_variance_gives_unit_reward_share():
    """If KL is constant across responses, all within-variance is reward and
    the reward share must be exactly 1 (KL and cov terms vanish)."""
    rng = np.random.RandomState(7)
    records = []
    for p in range(5):
        r = rng.normal(2.0, 1.0, size=32)
        kl = np.full(32, 3.0)            # constant KL within prompt
        R = r - BETA * kl
        records.append(
            {
                "prompt_key": p,
                "reward_samples": r.tolist(),
                "k1_kl_samples": kl.tolist(),
                "R_tilde_samples": R.tolist(),
                "response_lens": rng.randint(10, 31, size=32).astype(float).tolist(),
            }
        )
    agg = variance_decomposition(records, BETA)["aggregate"]
    assert abs(agg["weighted_share_reward"] - 1.0) < 1e-9
    assert abs(agg["weighted_share_kl"]) < 1e-9
    assert abs(agg["weighted_share_cov"]) < 1e-9


def test_pooled_advantage_is_mean_zero():
    records = _make_records(n_prompts=10, k=32, seed=3)
    A = pooled_advantages(records)
    # Equal K per prompt => exact zero pooled mean up to float error.
    assert abs(A.mean()) < 1e-12


def test_icc_subset_includes_top3_and_has_right_size():
    rng = np.random.RandomState(11)
    records = [
        {"prompt_key": p, "var_R_tilde": float(v)}
        for p, v in enumerate(rng.uniform(0.3, 6.0, size=100))
    ]
    n = 20
    idx = select_icc_subset(records, n)
    assert len(idx) == n
    assert len(set(idx)) == n                      # no duplicates
    # The three highest-variance prompts must be present.
    top3 = sorted(range(100), key=lambda i: -records[i]["var_R_tilde"])[:3]
    for t in top3:
        assert t in idx, (t, idx)


def test_icc_subset_spans_variance_range():
    """The non-top picks should cover low variance too, not cluster at the
    top — the min selected variance should be near the global min."""
    var = np.linspace(0.5, 5.5, 100)
    records = [{"prompt_key": p, "var_R_tilde": float(v)} for p, v in enumerate(var)]
    idx = select_icc_subset(records, 20)
    selected_var = [records[i]["var_R_tilde"] for i in idx]
    assert min(selected_var) < 1.0                 # reaches the low end
    assert max(selected_var) == var.max()          # top-3 guarantee hits the max


def test_consistency_guard_fires_on_beta_drift():
    records = _make_records(n_prompts=3, k=16, seed=4)
    # Correct beta passes.
    assert_records_consistent(records, BETA)
    # A wrong beta must raise (stored R_tilde no longer matches).
    with pytest.raises(ValueError, match="inconsistent"):
        assert_records_consistent(records, BETA + 0.05)
