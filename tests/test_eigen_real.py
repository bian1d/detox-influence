"""Stage 2 real-data sub-acceptance on GPT-Neo-125M.

Validates the eigen pipeline on the actual model:

* Eigendecomposition of real A, S completes without numerical errors
  (large negative eigenvalues, non-orthonormal Q).
* Q_A Q_Aᵀ ≈ I and Q_S Q_Sᵀ ≈ I to ``< 1e-10``.
* Λ entries are all positive after fit_lambda.
* Print Λ_A, Λ_S, Λ statistics for the eventual phase 2 report.

Does NOT run the full Frobenius gate on real (no F_true ground truth at
real-model scale; that test belongs in Stage 3 via leave-one-out IF rank
correlation on the toy).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import load_rollout_file  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402


REAL_ROLLOUT_PATH = Path("data/rollouts/step_0500.pt")
MODEL_NAME = "EleutherAI/gpt-neo-125m"


@pytest.fixture(scope="module")
def gpt_neo_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to("cuda").eval()
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def real_rollouts():
    if not REAL_ROLLOUT_PATH.exists():
        pytest.skip(f"{REAL_ROLLOUT_PATH} not present")
    return load_rollout_file(REAL_ROLLOUT_PATH)


@pytest.fixture(scope="module")
def real_AS_eigen(gpt_neo_on_gpu, real_rollouts):
    """Heavy fixture: A, S, eigendecomp, Λ. Reused by multiple tests."""
    cfg = EKFACConfig(layer_name="transformer.h.9.mlp.c_proj")
    layer = gpt_neo_on_gpu.get_submodule(cfg.layer_name)
    device = torch.device("cuda")
    t0 = time.time()
    A, S, n_tok_AS = accumulate_AS(gpt_neo_on_gpu, layer, real_rollouts, cfg, device=device)
    t_AS = time.time() - t0
    t0 = time.time()
    Q_A, Λ_A, Q_S, Λ_S = eigendecompose(A, S)
    t_eig = time.time() - t0
    t0 = time.time()
    Lambda, n_tok_L = fit_lambda(
        gpt_neo_on_gpu, layer, real_rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=device,
    )
    t_lambda = time.time() - t0
    return {
        "cfg": cfg, "layer": layer, "A": A, "S": S,
        "Q_A": Q_A, "Λ_A": Λ_A, "Q_S": Q_S, "Λ_S": Λ_S, "Lambda": Lambda,
        "n_tok_AS": n_tok_AS, "n_tok_L": n_tok_L,
        "t_AS": t_AS, "t_eig": t_eig, "t_lambda": t_lambda,
    }


def test_real_Q_orthonormal(real_AS_eigen) -> None:
    for name in ("Q_A", "Q_S"):
        Q = real_AS_eigen[name]
        I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
        err = (Q @ Q.T - I).abs().max().item()
        assert err < 1e-10, f"{name}: ||QQᵀ - I||∞ = {err:.3e}"


def test_real_lambda_positive(real_AS_eigen) -> None:
    Lambda = real_AS_eigen["Lambda"]
    min_v = Lambda.min().item()
    assert min_v > 0, f"Λ has non-positive entries: min = {min_v:.3e}"


def test_real_pipeline_statistics(real_AS_eigen) -> None:
    """Print real-model A/S eigenvalue + Lambda statistics for the report."""
    Λ_A = real_AS_eigen["Λ_A"]
    Λ_S = real_AS_eigen["Λ_S"]
    Lambda = real_AS_eigen["Lambda"]
    cfg = real_AS_eigen["cfg"]

    def stats(name: str, x: torch.Tensor) -> None:
        flat = x.reshape(-1).double()
        min_v = float(flat.min().item())
        med_v = float(flat.median().item())
        max_v = float(flat.max().item())
        cond = float(flat.max().item() / max(flat.min().item(), 1e-30)) if min_v > 0 else float("inf")
        # Condition with production damping floor + adaptive (alpha · mean).
        eff_floor = max(cfg.damping_floor, cfg.damping_alpha * flat.mean().item())
        cond_damped = float((flat.max().item() + eff_floor) / (flat.min().item() + eff_floor))
        print(f"  {name:14s} shape={tuple(x.shape)} min={min_v:.3e} med={med_v:.3e} max={max_v:.3e} "
              f"cond_raw={cond:.3e} cond_with_damping={cond_damped:.3e}")

    print()
    print(f"  Real-model accumulation timings:")
    print(f"    Stage 1A (A, S accumulate)   = {real_AS_eigen['t_AS']:.2f} s")
    print(f"    Stage 1B eigendecomposition  = {real_AS_eigen['t_eig']:.2f} s")
    print(f"    Stage 1B fit_lambda          = {real_AS_eigen['t_lambda']:.2f} s")
    print(f"    n_tok (AS, Lambda)            = ({real_AS_eigen['n_tok_AS']}, {real_AS_eigen['n_tok_L']})")
    print(f"  Eigenvalue + Lambda statistics:")
    stats("Λ_A (d_mlp)",   Λ_A)
    stats("Λ_S (d_model)", Λ_S)
    stats("Λ (d_in,d_out)", Lambda)
    n_below_floor = int((Lambda < cfg.damping_floor).sum().item())
    print(f"  Λ entries below damping_floor ({cfg.damping_floor:.0e}): "
          f"{n_below_floor} / {Lambda.numel()}")
    assert n_below_floor >= 0  # always true; print is the real artefact here.
