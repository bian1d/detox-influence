"""Stage 3 determinism gate.

The single largest silent-failure risk in Stage 3 is non-deterministic
retraining: if ``θ*_{-m}`` is reproducible only up to seeded variance, then
``I_true ≈ f(θ*) - f(θ*_{-m})`` is corrupted by noise indistinguishable
from EK-FAC approximation error, producing a false-negative Kendall τ.

This file gates the LOO loop with three checks:

1. Two retrainings from the same seed produce bit-identical parameters.
2. Training converges (final loss substantially below initial loss).
3. The seeded random rollout generator is reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402
from ekfac.training import set_determinism, train_toy_to_optimum  # noqa: E402


def _toy_cfg() -> EKFACConfig:
    """Toy uses layer 1 (the only non-zero index in 2-layer toy)."""
    return EKFACConfig(layer_name="transformer.h.1.mlp.c_proj")


def _make_random_rollouts(
    n: int, vocab_size: int, max_len: int, *, seed: int
) -> list[Rollout]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        P = int(torch.randint(1, max_len, (1,), generator=g).item())
        R = max_len - P
        out.append(Rollout(
            prompt_ids=torch.randint(0, vocab_size, (P,), generator=g, dtype=torch.long),
            response_ids=torch.randint(0, vocab_size, (R,), generator=g, dtype=torch.long),
            reward=0.0, step=i,
        ))
    return out


def _build_model(init_seed: int) -> ToyTransformer:
    set_determinism(init_seed)
    return ToyTransformer()


def test_training_is_bit_deterministic() -> None:
    """Two retrainings from identical seeds must produce identical params."""
    cfg = _toy_cfg()
    rollouts = _make_random_rollouts(20, vocab_size=20, max_len=4, seed=10)

    model_a = _build_model(init_seed=42)
    set_determinism(123)  # advance state to the train-time seed
    losses_a = train_toy_to_optimum(model_a, rollouts, cfg, n_steps=50, lr=1e-2)
    params_a = {k: v.detach().clone() for k, v in model_a.state_dict().items()}

    model_b = _build_model(init_seed=42)
    set_determinism(123)
    losses_b = train_toy_to_optimum(model_b, rollouts, cfg, n_steps=50, lr=1e-2)
    params_b = {k: v.detach().clone() for k, v in model_b.state_dict().items()}

    # Loss trajectories must match exactly.
    assert losses_a == losses_b, "loss trajectories differ — non-deterministic training"
    # Every parameter must match bit-for-bit.
    for k in params_a:
        if not torch.equal(params_a[k], params_b[k]):
            diff = (params_a[k] - params_b[k]).abs().max().item()
            raise AssertionError(
                f"{k}: max abs diff = {diff:.3e} (non-deterministic retraining)"
            )


def test_training_converges() -> None:
    """Final loss must be substantially below initial loss for the LOO
    ground truth to be meaningful."""
    cfg = _toy_cfg()
    rollouts = _make_random_rollouts(20, vocab_size=20, max_len=4, seed=10)
    model = _build_model(init_seed=42)
    set_determinism(123)
    losses = train_toy_to_optimum(model, rollouts, cfg, n_steps=200, lr=1e-2)
    # Sanity: with 1568 params on 20 random rollouts (each ~2 tokens), the
    # network should drive loss far below the random-init baseline.
    assert losses[-1] < 0.5 * losses[0], (
        f"loss did not converge enough: initial={losses[0]:.3f}, "
        f"final={losses[-1]:.3f}"
    )
    assert losses[-1] < losses[-50], (
        f"loss is still decreasing at end (init={losses[0]:.3f}, "
        f"step-50-from-end={losses[-50]:.3f}, final={losses[-1]:.3f}); "
        f"need more steps or different lr"
    )


def test_rollout_generator_reproducible() -> None:
    """Seeded rollout generation must be reproducible (orthogonal channel
    to model retraining determinism)."""
    a = _make_random_rollouts(20, vocab_size=20, max_len=4, seed=10)
    b = _make_random_rollouts(20, vocab_size=20, max_len=4, seed=10)
    assert len(a) == len(b) == 20
    for ra, rb in zip(a, b):
        assert torch.equal(ra.prompt_ids, rb.prompt_ids)
        assert torch.equal(ra.response_ids, rb.response_ids)
