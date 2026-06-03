"""Phase 5 Section 2 — re-run the Phase 4 EK-FAC / Delta-VP correctness gates on
a **gated** SwiGLU toy (OLMo-2-shaped), because the model changed (gated MLP)
and the double-backward now runs through silu*gate*up. ALL must be green before
touching the real OLMo-2.

The four Phase-4 gates map as follows:
  1. HVP double-backward == dense Hessian @ v        -> here, on the gated toy.
  2. matrix-free Delta~ @ v == brute dense Delta~ @ v -> here, on the gated toy.
  3. per-prompt accumulation averages to flat total   -> here, on the gated toy.
  4. project-but-NO-double-denom vs brute F^-1        -> model-INDEPENDENT
     (synthetic Q_A/Q_S/Lambda; ``inverse_hvp_additive`` does not touch the
     MLP). Covered by tests/test_phase4_e3.py::test_no_double_denom_cache_chain.
  5. MINRES converges on indefinite, CG breaks down   -> model-INDEPENDENT
     (pure linear algebra). Covered by
     tests/test_phase4_e3.py::test_minres_solves_indefinite_cg_breaks_down.

So the verification command runs BOTH files:
    pytest tests/test_phase5_toy_gates.py tests/test_phase4_e3.py
gates 1-3 get the gated-MLP graph here; gates 4-5 (unchanged by the model) are
re-asserted by the Phase-4 file in the same run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ekfac.toy_gated import ToyGatedTransformer  # noqa: E402
from phase4.e3_delta import (  # noqa: E402
    DeltaSample,
    delta_vp_per_prompt,
    delta_vp_total,
)
from phase4.grad_phi import get_phi_weight  # noqa: E402
from phase4.hvp_logpi import _mean_logpi, hvp_logpi_phi  # noqa: E402

BETA = 0.2365
PHI = "transformer.h.1.mlp.down_proj"   # gated toy's phi (analog of OLMo down_proj)


def _toy():
    torch.manual_seed(0)
    model = ToyGatedTransformer(max_len=8).double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    w = get_phi_weight(model, PHI)
    w.requires_grad_(True)
    return model, w


def _rand_sample(model, P=3, R=4, seed=0, A=1.0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.vocab_size, (P + R,), generator=g)
    return DeltaSample(prompt_idx=seed % 3, prompt_ids=ids[:P], response_ids=ids[P:], A=A)


def _dense_hessian(model, w, smp):
    """Column-by-column dense Hessian of mean-logpi w.r.t. w (fp64)."""
    d = w.numel()
    logpi = _mean_logpi(model, w, smp.prompt_ids, smp.response_ids, "cpu", -100)
    (g,) = torch.autograd.grad(logpi, w, create_graph=True)
    gflat = g.reshape(-1)
    H = torch.zeros(d, d, dtype=torch.float64)
    for k in range(d):
        (col,) = torch.autograd.grad(gflat[k], w, retain_graph=True)
        H[k] = col.reshape(-1)
    return H


# --- Gate 1: HVP double-backward (through SwiGLU) == dense Hessian @ v --------
def test_gated_hvp_matches_dense_hessian():
    model, w = _toy()
    smp = _rand_sample(model, seed=1)
    v = torch.randn(w.shape, dtype=torch.float64)

    hvp = hvp_logpi_phi(model, w, smp.prompt_ids, smp.response_ids, v, "cpu")
    H = _dense_hessian(model, w, smp)
    hvp_dense = (H @ v.reshape(-1)).reshape(w.shape)
    assert torch.allclose(hvp, hvp_dense, atol=1e-9), (hvp - hvp_dense).abs().max().item()


# --- Gate 2: matrix-free Delta~ @ v == brute dense Delta~ @ v -----------------
def test_gated_delta_vp_matches_brute_force():
    model, w = _toy()
    samples = [_rand_sample(model, seed=i, A=float(np.sin(i) * 2.0)) for i in range(5)]
    v = torch.randn(w.shape, dtype=torch.float64)

    (dv_mf,) = delta_vp_total(samples, model, w, BETA, [v], "cpu")

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


# --- Gate 3: per-prompt accumulation averages to the flat total --------------
def test_gated_per_prompt_averages_to_total():
    model, w = _toy()
    samples = []
    for pi in range(2):
        for j in range(2):
            s = _rand_sample(model, seed=10 * pi + j, A=float(0.5 * pi - 0.3 * j))
            s.prompt_idx = pi
            samples.append(s)
    v = torch.randn(w.shape, dtype=torch.float64)
    (dv_total,) = delta_vp_total(samples, model, w, BETA, [v], "cpu")
    per_prompt, _order, _ = delta_vp_per_prompt(
        samples, model, w, BETA, [v], "cpu", store_dtype=torch.float64
    )
    dv_pp = per_prompt[0].mean(0)
    assert torch.allclose(dv_total, dv_pp, atol=1e-9), (dv_total - dv_pp).abs().max().item()
