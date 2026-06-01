"""E3 correctness gates (toy / synthetic). ALL must be green before the real
model — a bug here silently corrupts the central Delta error-budget.

Gates:
  1. HVP double-backward == dense Hessian @ v on the toy transformer.
  2. matrix-free Delta~ @ v == brute-force dense Delta~ @ v.
  3. per-prompt accumulation averages to the flat-total Delta~.
  4. project-but-NO-double-denom: the cached-g_scaled path reproduces the
     brute -g_f^T F^-1 s_m, and dividing the eval side by denom AGAIN gives a
     DIFFERENT (wrong) answer (locks the critical pitfall the user flagged).
  5. MINRES converges on a symmetric INDEFINITE (F - Delta~); CG does not
     (negative control).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ekfac.eigen import inverse_hvp_additive  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402
from phase4.e3_delta import (  # noqa: E402
    DeltaSample,
    delta_vp_per_prompt,
    delta_vp_total,
    project_eval_vector,
)
from phase4.grad_phi import get_phi_weight, score_logpi_phi  # noqa: E402
from phase4.hvp_logpi import hvp_logpi_phi  # noqa: E402

BETA = 0.2365


def _toy():
    torch.manual_seed(0)
    model = ToyTransformer(max_len=8).double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    w = get_phi_weight(model, "transformer.h.1.mlp.c_proj")
    w.requires_grad_(True)
    return model, w


def _rand_sample(model, P=3, R=4, seed=0, A=1.0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.vocab_size, (P + R,), generator=g)
    return DeltaSample(prompt_idx=seed % 3, prompt_ids=ids[:P], response_ids=ids[P:], A=A)


# --------------------------------------------------------------------------- #
# 1. HVP vs dense Hessian
# --------------------------------------------------------------------------- #
def test_hvp_matches_dense_hessian():
    model, w = _toy()
    smp = _rand_sample(model, seed=1)
    d = w.numel()
    v = torch.randn(w.shape, dtype=torch.float64)

    hvp = hvp_logpi_phi(model, w, smp.prompt_ids, smp.response_ids, v, "cpu")

    # Dense Hessian of mean-logpi w.r.t. w, column by column.
    from phase4.hvp_logpi import _mean_logpi
    logpi = _mean_logpi(model, w, smp.prompt_ids, smp.response_ids, "cpu", -100)
    (g,) = torch.autograd.grad(logpi, w, create_graph=True)
    gflat = g.reshape(-1)
    H = torch.zeros(d, d, dtype=torch.float64)
    for k in range(d):
        (col,) = torch.autograd.grad(gflat[k], w, retain_graph=True)
        H[k] = col.reshape(-1)
    hvp_dense = (H @ v.reshape(-1)).reshape(w.shape)
    assert torch.allclose(hvp, hvp_dense, atol=1e-9), (hvp - hvp_dense).abs().max().item()


# --------------------------------------------------------------------------- #
# 2. matrix-free Delta~ vs brute-force dense Delta~
# --------------------------------------------------------------------------- #
def test_delta_vp_matches_brute_force():
    model, w = _toy()
    samples = [_rand_sample(model, seed=i, A=float(np.sin(i) * 2.0)) for i in range(5)]
    v = torch.randn(w.shape, dtype=torch.float64)

    (dv_mf,) = delta_vp_total(samples, model, w, BETA, [v], "cpu")

    # Brute: Delta~ = (1/beta)(1/n) sum_i A_i (s_i s_i^T + H_i); then Delta~ v.
    from phase4.hvp_logpi import _mean_logpi
    d = w.numel()
    Delta = torch.zeros(d, d, dtype=torch.float64)
    for smp in samples:
        logpi = _mean_logpi(model, w, smp.prompt_ids, smp.response_ids, "cpu", -100)
        (g,) = torch.autograd.grad(logpi, w, create_graph=True)
        s = g.detach().reshape(-1)
        gflat = g.reshape(-1)
        H = torch.zeros(d, d, dtype=torch.float64)
        for k in range(d):
            (col,) = torch.autograd.grad(gflat[k], w, retain_graph=True)
            H[k] = col.reshape(-1)
        Delta += smp.A * (torch.outer(s, s) + H)
    Delta = Delta / (BETA * len(samples))
    dv_brute = (Delta @ v.reshape(-1)).reshape(w.shape)
    assert torch.allclose(dv_mf, dv_brute, atol=1e-8), (dv_mf - dv_brute).abs().max().item()


def test_per_prompt_averages_to_total():
    model, w = _toy()
    # Two prompts, 2 samples each.
    samples = []
    for pi in range(2):
        for j in range(2):
            s = _rand_sample(model, seed=10 * pi + j, A=float(0.5 * pi - 0.3 * j))
            s.prompt_idx = pi
            samples.append(s)
    v = torch.randn(w.shape, dtype=torch.float64)
    (dv_total,) = delta_vp_total(samples, model, w, BETA, [v], "cpu")
    per_prompt, order, _ = delta_vp_per_prompt(samples, model, w, BETA, [v], "cpu", store_dtype=torch.float64)
    dv_pp = per_prompt[0].mean(0)
    assert torch.allclose(dv_total, dv_pp, atol=1e-9), (dv_total - dv_pp).abs().max().item()


# --------------------------------------------------------------------------- #
# 4. project-but-NO-double-denom (the critical cache pitfall)
# --------------------------------------------------------------------------- #
def test_no_double_denom_cache_chain():
    torch.manual_seed(3)
    d_out, d_in = 6, 9
    Q_A = torch.linalg.qr(torch.randn(d_in, d_in, dtype=torch.float64))[0]
    Q_S = torch.linalg.qr(torch.randn(d_out, d_out, dtype=torch.float64))[0]
    Lambda = torch.rand(d_in, d_out, dtype=torch.float64) + 0.05   # (d_in,d_out)
    damping = 0.01
    denom_T = (Lambda + damping).T.contiguous()                    # (d_out,d_in)

    g_f = torch.randn(d_out, d_in, dtype=torch.float64)
    s_m = torch.randn(d_out, d_in, dtype=torch.float64)

    # Ground truth -g_f^T F^-1 s_m.
    Fi_sm = inverse_hvp_additive(s_m, Q_A, Q_S, Lambda, damping=damping)
    truth = -float((g_f * Fi_sm).sum().item())

    # Production cache layout: g_scaled_z = Q_S^T s_m Q_A / denom_T.
    g_scaled = (Q_S.T @ s_m @ Q_A) / denom_T
    g_f_proj = project_eval_vector(g_f, Q_A, Q_S)                  # projection only

    correct = -float((g_f_proj * g_scaled).sum().item())
    assert abs(correct - truth) < 1e-10, (correct, truth)

    # The WRONG double-divide must give a DIFFERENT answer (guards the pitfall).
    g_f_proj_wrong = g_f_proj / denom_T
    wrong = -float((g_f_proj_wrong * g_scaled).sum().item())
    assert abs(wrong - truth) > 1e-6 * (abs(truth) + 1e-12), (wrong, truth)


# --------------------------------------------------------------------------- #
# 5. MINRES on symmetric indefinite; CG fails (negative control)
# --------------------------------------------------------------------------- #
def _relres(M, x, b):
    if x is None or not np.all(np.isfinite(x)):
        return np.inf
    return float(np.linalg.norm(M @ x - b) / np.linalg.norm(b))


def test_minres_solves_indefinite_cg_breaks_down():
    """MINRES is the correct solver for the symmetric INDEFINITE (F - Delta~);
    CG assumes SPD and breaks down.  Two demonstrations:

      (a) Deterministic CG breakdown: A = diag(1,-1), b = [1,1] forces the
          first CG curvature p0^T A p0 = b1^2 - b2^2 = 0 (division by zero).
          MINRES solves it exactly.
      (b) Robustness: across random indefinite systems MINRES always converges.
    """
    from scipy.sparse.linalg import LinearOperator, cg, minres

    # (a) Deterministic breakdown case.
    A = np.diag([1.0, -1.0])
    b = np.array([1.0, 1.0])
    opA = LinearOperator((2, 2), matvec=lambda x: A @ x)
    x_min, _ = minres(opA, b, rtol=1e-10, maxiter=100)
    assert _relres(A, x_min, b) < 1e-8, _relres(A, x_min, b)      # MINRES: exact
    x_cg, _ = cg(opA, b, rtol=1e-10, maxiter=100)
    assert _relres(A, x_cg, b) > 1e-4, _relres(A, x_cg, b)        # CG: breaks down

    # (b) MINRES converges on many random indefinite systems.
    rng = np.random.RandomState(4)
    worst_min = 0.0
    for t in range(15):
        n = 50
        Q = np.linalg.qr(rng.randn(n, n))[0]
        eig = np.concatenate([rng.uniform(0.3, 2.0, n - 10), rng.uniform(-2.0, -0.3, 10)])
        M = 0.5 * ((Q * eig) @ Q.T + ((Q * eig) @ Q.T).T)
        rhs = rng.randn(n)
        op = LinearOperator((n, n), matvec=lambda x, M=M: M @ x)
        xm, _ = minres(op, rhs, rtol=1e-9, maxiter=4000)
        worst_min = max(worst_min, _relres(M, xm, rhs))
    assert worst_min < 1e-6, f"MINRES worst residual {worst_min}"
