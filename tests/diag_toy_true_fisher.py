"""Diagnostic dump for Phase 2 kickoff report.

Prints:
    1. ToyTransformer architecture summary (all parameters and shapes)
    2. The true-Fisher computation function (sourced from this test module
       at runtime, so the printed text is the exact code that runs)
    3. F_true numerical sanity: symmetry, PSD, eigenvalue range,
       condition number, effective rank.

Not a pytest test; run as ``python tests/diag_toy_true_fisher.py``.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402

# Import the function we want to dump source for.
sys.path.insert(0, str(Path(__file__).parent))
from test_eigen_toy import (  # noqa: E402
    build_toy_samples,
    compute_true_fisher_per_token,
    fisher_diagnostics,
)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def main() -> None:
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)

    # 1. Toy architecture
    banner("1. Toy model architecture")
    model = ToyTransformer()
    model.eval()
    print(model.summary())
    target_name = "transformer.h.1.mlp.c_proj"
    target = model.get_submodule(target_name)
    print()
    print(f"Target layer for EK-FAC: {target_name}")
    print(f"  weight.shape       = {tuple(target.weight.shape)}  "
          f"(d_model={model.d_model}, d_mlp={model.d_mlp})")
    print(f"  flatten dim        = {target.weight.numel()}")
    print(f"  true-Fisher matrix = ({target.weight.numel()}, {target.weight.numel()})")

    # 2. Function source
    banner("2. True-Fisher computation (compute_true_fisher_per_token)")
    print(inspect.getsource(compute_true_fisher_per_token))

    # 3. Numerical sanity on F_true
    banner("3. F_true numerical sanity")
    # Bumped from 20 → 150 so n_tok ≥ 2 × Fisher_dim = 256 (cf. user
    # decision on Option (a) — full-rank F_true for the Stage-2 gate).
    samples = build_toy_samples(
        n_samples=150,
        vocab_size=model.vocab_size,
        max_len=model.max_len,
        seed=cfg.seed,
    )
    print(f"Synthetic samples: n={len(samples)}, "
          f"prompt_len range = {min(s['prompt_len'] for s in samples)}..{max(s['prompt_len'] for s in samples)}")
    F, n_tok = compute_true_fisher_per_token(model, target, samples, seed=cfg.seed)
    print(f"Response tokens accumulated: n_tok = {n_tok}")
    diag = fisher_diagnostics(F)
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k:<26s} = {v:.6e}")
        else:
            print(f"  {k:<26s} = {v}")
    # Eigenvalue distribution at multiple thresholds.
    eigs = torch.linalg.eigvalsh(F).sort(descending=True).values
    print()
    print("Eigenvalue tail (largest 5, smallest 20):")
    print(f"  top 5:   {[f'{x:.3e}' for x in eigs[:5].tolist()]}")
    print(f"  bot 20:  {[f'{x:.3e}' for x in eigs[-20:].tolist()]}")
    max_e = eigs.max().item()
    for thr in [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        n = int((eigs > thr * max_e).sum().item())
        print(f"  eigs > {thr:.0e} * max_eig : {n}/128")
    print()
    print(f"Toy damping for the Stage-2 gate: λ = {cfg.toy_damping}")
    print(f"Apples-to-apples F_inv = inv(F_true + {cfg.toy_damping} * I)")


if __name__ == "__main__":
    main()
