"""Phase 5 landing check: prove the Phase 4 *numerical* estimators run clean
under the olmo env's numpy 2.x / torch 2.5.1 / transformers 4.49 stack,
*independently* of the TRL packaging coupling.

Context: `phase4.e2_stationarity` (which hosts the headline U-statistic
estimator `ustat_quadratic` + `jackknife_ci`) transitively imports
`phase4.sampling -> phase1.score -> train_ppo -> trl`. TRL is deliberately
absent from the olmo env (Stage 1 SFT quick-test uses no PPO), so a plain
`import phase4.e2_stationarity` raises ModuleNotFoundError('trl'). That is a
packaging coupling, NOT a numpy-2.x incompatibility.

This script stubs out the `trl` module so the import resolves, then exercises
`ustat_quadratic` against a brute-force double sum and checks `jackknife_ci`
agrees at its point estimate -- the same assertions as test_phase4_e2.py, but
reachable here. If these pass, numpy 2.x is compatible with the estimator math;
the only Stage-1 work needed is to decouple the import chain (make train_ppo /
trl lazy), which is left for Stage 1 itself.

Run with the olmo env python:
    /root/miniconda3/envs/olmo/bin/python tests/diag_olmo_estimator_compat.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _stub_trl() -> None:
    """Inject a minimal fake `trl` so train_ppo's top-level import resolves.

    We never call any of these symbols here -- the only goal is to let the
    import chain phase4.e2_stationarity -> ... -> train_ppo succeed so the
    pure-numpy/torch estimator functions become importable.
    """
    trl = types.ModuleType("trl")
    for name in (
        "AutoModelForCausalLMWithValueHead",
        "PPOConfig",
        "PPOTrainer",
        "create_reference_model",
    ):
        setattr(trl, name, object)
    core = types.ModuleType("trl.core")
    core.LengthSampler = object
    sys.modules["trl"] = trl
    sys.modules["trl.core"] = core


def _brute_ustat(g: torch.Tensor, Mg: torch.Tensor | None = None) -> float:
    """Explicit (1/(N(N-1))) sum_{p!=q} <g_p, (M g)_q>; mirrors test_phase4_e2."""
    N = g.shape[0]
    Gm = g.reshape(N, -1).double()
    MGm = Gm if Mg is None else Mg.reshape(N, -1).double()
    tot = 0.0
    for p in range(N):
        for q in range(N):
            if p != q:
                tot += float(torch.dot(Gm[p], MGm[q]))
    return tot / (N * (N - 1))


def main() -> int:
    print("numpy", np.__version__, "| torch", torch.__version__)
    _stub_trl()

    # This import is what fails on a bare olmo env (ModuleNotFoundError: trl).
    from phase4.e2_stationarity import jackknife_ci, ustat_quadratic

    print("imported phase4.e2_stationarity (with trl stubbed): OK")

    # gbar is a stack of per-prompt mean score *matrices* (N, d_out, d_in).
    torch.manual_seed(5)
    g = torch.randn(7, 4, 3, dtype=torch.float64)
    Mg = torch.randn(7, 4, 3, dtype=torch.float64)  # stand-in for F^-1 g_p precomputed

    # Gate 1: identity-metric U-statistic == brute force.
    u0, b0 = ustat_quadratic(g, None), _brute_ustat(g, None)
    print(f"ustat(g)     = {u0:.10f}  brute = {b0:.10f}  |diff| = {abs(u0-b0):.2e}")
    assert abs(u0 - b0) < 1e-9, "U-stat (identity) disagrees with brute force"

    # Gate 2: with an explicit Mg array == brute force.
    uM, bM = ustat_quadratic(g, Mg), _brute_ustat(g, Mg)
    print(f"ustat(g, Mg) = {uM:.10f}  brute = {bM:.10f}  |diff| = {abs(uM-bM):.2e}")
    assert abs(uM - bM) < 1e-9, "U-stat (metric) disagrees with brute force"

    # Gate 3: jackknife point estimate == the U-statistic itself, point in CI.
    torch.manual_seed(6)
    g2 = torch.randn(9, 4, 3, dtype=torch.float64)
    jk = jackknife_ci(g2, None)
    u2 = ustat_quadratic(g2, None)
    print(f"jackknife point = {jk['point']:.12f}  vs ustat {u2:.12f}  ci95 = "
          f"({jk['ci95'][0]:.4f}, {jk['ci95'][1]:.4f})")
    assert abs(jk["point"] - u2) < 1e-12, "jackknife point estimate != U-statistic"
    assert jk["ci95"][0] <= jk["point"] <= jk["ci95"][1], "point not inside its own CI"

    print("\nNUMPY-2.x ESTIMATOR COMPAT: OK (math clean; only the trl import is a packaging coupling)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
