"""Stage 1A real-data acceptance on GPT-Neo-125M.

Per phases/phase2.md Stage 1 real-data acceptance:
    * Accumulate over the 32 rollouts in ``step_0050.pt``.
    * A, S still symmetric and PSD.
    * Eigenvalue ranges: largest eigenvalue of A is O(1) to O(100),
      largest of S is O(0.001) to O(0.1).
    * Stage 1 on 32 rollouts takes < 5 minutes on GPU.

The archived run (``data/archive/run_no_min_length/rollouts/step_0050.pt``)
is used while the re-run is in progress. Schema is identical.

GPU sharing: PPO uses ~27 GB on the same physical GPU. GPT-Neo-125M fp32
is ~500 MB; with response-length ≤30 and 32 rollouts the workspace stays
well under 1 GB. Verified before running this file.
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
from ekfac.factors import accumulate_AS  # noqa: E402


REAL_ROLLOUT_PATH = Path("data/archive/run_no_min_length/rollouts/step_0050.pt")
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
def step_0050_rollouts():
    if not REAL_ROLLOUT_PATH.exists():
        pytest.skip(f"{REAL_ROLLOUT_PATH} not present")
    rollouts = load_rollout_file(REAL_ROLLOUT_PATH)
    assert len(rollouts) == 32, f"expected 32 rollouts, got {len(rollouts)}"
    return rollouts


def test_real_stage1_acceptance(gpt_neo_on_gpu, step_0050_rollouts) -> None:
    cfg = EKFACConfig(layer_name="transformer.h.9.mlp.c_proj")
    layer = gpt_neo_on_gpu.get_submodule(cfg.layer_name)
    assert layer.weight.shape == (768, 3072), tuple(layer.weight.shape)

    device = torch.device("cuda")
    t0 = time.time()
    A, S, n_tok = accumulate_AS(
        gpt_neo_on_gpu, layer, step_0050_rollouts, cfg, device=device,
    )
    elapsed = time.time() - t0

    # --- Sanity ---
    asym_A = (A - A.T).abs().max().item()
    asym_S = (S - S.T).abs().max().item()
    eigs_A = torch.linalg.eigvalsh(A)
    eigs_S = torch.linalg.eigvalsh(S)
    min_A = float(eigs_A.min().item())
    min_S = float(eigs_S.min().item())
    max_A = float(eigs_A.max().item())
    max_S = float(eigs_S.max().item())
    med_A = float(eigs_A.median().item())
    med_S = float(eigs_S.median().item())
    diag_A_min = float(torch.diagonal(A).min().item())
    diag_S_min = float(torch.diagonal(S).min().item())

    print()
    print(f"  n_tok            = {n_tok}")
    print(f"  elapsed (s)      = {elapsed:.2f}")
    print(f"  A shape          = {tuple(A.shape)}")
    print(f"  S shape          = {tuple(S.shape)}")
    print(f"  A max_asymmetry  = {asym_A:.2e}")
    print(f"  S max_asymmetry  = {asym_S:.2e}")
    print(f"  A eigval [min,med,max] = [{min_A:.3e}, {med_A:.3e}, {max_A:.3e}]")
    print(f"  S eigval [min,med,max] = [{min_S:.3e}, {med_S:.3e}, {max_S:.3e}]")
    print(f"  diag(A).min      = {diag_A_min:.3e}")
    print(f"  diag(S).min      = {diag_S_min:.3e}")

    assert asym_A < 1e-10, f"A asymmetry {asym_A:.2e} >= 1e-10"
    assert asym_S < 1e-10, f"S asymmetry {asym_S:.2e} >= 1e-10"
    assert min_A >= -1e-8, f"A min eigval {min_A:.2e} < -1e-8 (not PSD)"
    assert min_S >= -1e-8, f"S min eigval {min_S:.2e} < -1e-8 (not PSD)"
    assert diag_A_min > 0, f"A has non-positive diagonal: min = {diag_A_min:.3e}"
    assert diag_S_min > 0, f"S has non-positive diagonal: min = {diag_S_min:.3e}"

    # Spec ranges from phase2.md (treat as soft sanity, not hard fail —
    # surfaced numbers go to the report and the user decides on borderline).
    A_max_in_range = 1.0 <= max_A <= 100.0
    S_max_in_range = 1e-3 <= max_S <= 1e-1
    print(f"  A.eig_max in [1, 100]?     {A_max_in_range}")
    print(f"  S.eig_max in [1e-3, 1e-1]? {S_max_in_range}")

    assert elapsed < 300.0, f"Stage 1 on 32 rollouts took {elapsed:.1f}s, > 300s budget"
