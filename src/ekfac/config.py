"""EK-FAC pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class EKFACConfig:
    """All EK-FAC hyperparameters and paths.

    Damping has two regimes that must not be conflated:

      * Production IHVP (Stage 3 on the real model) uses the MDA two-level
        scheme: ``denom = max(Λ + damping_alpha * Λ.mean(), damping_floor)``.
        See CONTEXT.md §13.2 and §18.3.

      * The toy-model gate (Stage 2 acceptance test in tests/test_eigen_toy.py)
        compares EK-FAC's reconstructed inverse against ``np.linalg.inv(F_true
        + λI)``. To isolate EK-FAC approximation error from damping-scheme
        differences, both sides use the same fixed scalar ``toy_damping``.
    """

    # Parameter subspace (provisional; finalized post-PPO).
    # Must resolve via ``model.get_submodule(layer_name)`` on both toy and
    # GPT-Neo. Toy model exposes ``transformer.h.<i>.mlp.c_proj`` identically.
    layer_name: str = "transformer.h.9.mlp.c_proj"

    # Production damping (Stage 3).
    damping_floor: float = 1e-5
    damping_alpha: float = 0.1

    # Toy-gate damping (Stage 2 acceptance test only).
    toy_damping: float = 0.01

    # Determinism (pseudo-label sampling, batch shuffling).
    seed: int = 42

    # Numerics. Factors accumulate in fp64; per-sample gradients in fp32.
    dtype_factors: torch.dtype = torch.float64
    dtype_grads: torch.dtype = torch.float32

    # Cross-entropy mask sentinel (HuggingFace convention).
    ignore_index: int = -100

    # IO.
    rollouts_dir: Path = field(default_factory=lambda: Path("data/rollouts"))
    archived_rollouts_dir: Path = field(
        default_factory=lambda: Path("data/archive/run_no_min_length/rollouts")
    )
    out_dir: Path = field(default_factory=lambda: Path("data/ekfac"))
