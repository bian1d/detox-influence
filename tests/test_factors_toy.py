"""Stage 1A toy-model acceptance.

Three groups of checks (per phase2.md Stage 1 acceptance + user's extras):

1. Hook shape verification — for one toy rollout, the gradient assembled
   from hook captures ``δ.T @ m`` must equal ``autograd.grad(loss, W)`` to
   fp32 relative error < 1e-5. Catches transpose / index bugs at the
   source (cheaper than full F⁻¹ comparison).

2. A, S sanity — symmetric to fp64 (max asymmetry < 1e-10), PSD (min eig
   ≥ -1e-8), strictly positive diagonal.

3. Eigenvalue ranges — print (and assert sane positivity for) min, max,
   median eigenvalues of A and S. Numbers are surfaced for the kickoff
   report; assertions here are loose ("max > 0") because the toy has no
   meaningful absolute scale.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.hooks import capture_c_proj  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402


def _make_toy_rollout(model: ToyTransformer, *, p: int = 2, r: int = 2, seed: int = 0) -> Rollout:
    """Random (prompt, response) within toy vocab/length budget."""
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, model.vocab_size, (p,), generator=g)
    response = torch.randint(0, model.vocab_size, (r,), generator=g)
    return Rollout(prompt_ids=prompt, response_ids=response, reward=0.0, step=0)


def _make_toy_rollouts(model: ToyTransformer, n: int, *, seed: int = 0) -> list[Rollout]:
    """Variable-length rollouts within ``max_len`` for n samples."""
    g = torch.Generator().manual_seed(seed)
    out: list[Rollout] = []
    for i in range(n):
        # P in [1, max_len-1], R in [1, max_len-P]
        P = int(torch.randint(1, model.max_len, (1,), generator=g).item())
        R = int(torch.randint(1, model.max_len - P + 1, (1,), generator=g).item())
        prompt = torch.randint(0, model.vocab_size, (P,), generator=g)
        response = torch.randint(0, model.vocab_size, (R,), generator=g)
        out.append(Rollout(prompt_ids=prompt, response_ids=response, reward=0.0, step=i))
    return out


# ---------------------------------------------------------------------------
# 1. Hook shape verification
# ---------------------------------------------------------------------------


def test_hook_grad_matches_autograd() -> None:
    """``δ.T @ m`` (hook captures, response slice) must equal
    ``autograd.grad(loss, W)`` for the same forward+backward.

    Uses the recorded-label (not pseudo) loss path and reduction='sum' so
    that the per-token mathematical equivalence ``∇_W L_total = Σ_t δ_t m_tᵀ``
    is exact (no normaliser to worry about). The Stage-1A accumulation uses
    the same path with pseudo labels, so this test directly validates the
    hook-to-grad correspondence Stage 1A relies on.
    """
    torch.manual_seed(0)
    model = ToyTransformer()
    model.eval()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    W = layer.weight

    rollout = _make_toy_rollout(model, p=2, r=2, seed=0)
    P, R = rollout.prompt_len, rollout.response_len
    T = P + R
    input_ids = torch.cat([rollout.prompt_ids, rollout.response_ids]).unsqueeze(0)
    labels = torch.cat([
        torch.full((P,), -100, dtype=torch.long),
        rollout.response_ids,
    ])

    # Zero existing grads on W.
    if W.grad is not None:
        W.grad = None

    with capture_c_proj(layer) as cache:
        logits = model(input_ids)
        shift_logits = logits[0, :-1].float()
        shift_labels = labels[1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="sum",
        )
        loss.backward()

    g_autograd = W.grad.detach().clone()        # (d_out, d_in)

    # Reassemble via hooks: response slice of m, δ.
    m_resp = cache.m[0, P - 1 : T - 1, :]       # (R, d_in)
    d_resp = cache.delta[0, P - 1 : T - 1, :]   # (R, d_out)
    g_hooks = d_resp.T @ m_resp                  # (d_out, d_in)

    # Hook captures should reproduce autograd.grad to fp32 precision.
    rel_err = (g_hooks - g_autograd).norm() / g_autograd.norm().clamp_min(1e-30)
    assert rel_err.item() < 1e-5, (
        f"hook-vs-autograd relative error {rel_err.item():.2e} exceeds 1e-5; "
        f"||g_autograd||={g_autograd.norm().item():.3e}, "
        f"||g_hooks||={g_hooks.norm().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 2. A, S sanity (symmetric / PSD / positive diag)
# ---------------------------------------------------------------------------


def _accumulate_on_toy(n_rollouts: int = 80) -> tuple[torch.Tensor, torch.Tensor, int]:
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer()
    model.eval()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    rollouts = _make_toy_rollouts(model, n=n_rollouts, seed=cfg.seed)
    device = torch.device("cpu")
    A, S, n_tok = accumulate_AS(model, layer, rollouts, cfg, device=device)
    return A, S, n_tok


def test_AS_symmetric_psd_positive_diagonal() -> None:
    A, S, n_tok = _accumulate_on_toy(n_rollouts=80)
    assert n_tok > 0

    for name, M in [("A", A), ("S", S)]:
        asym = (M - M.T).abs().max().item()
        assert asym < 1e-10, f"{name} asymmetry {asym:.2e} >= 1e-10"
        eigs = torch.linalg.eigvalsh(M)
        min_eig = float(eigs.min().item())
        assert min_eig >= -1e-8, f"{name} min eigenvalue {min_eig:.2e} < -1e-8 (not PSD)"
        diag = torch.diagonal(M)
        assert (diag > 0).all().item(), (
            f"{name} has non-positive diagonal entries: min diag = {diag.min().item():.3e}"
        )


# ---------------------------------------------------------------------------
# 3. Eigenvalue ranges (informational + sanity)
# ---------------------------------------------------------------------------


def test_AS_eigenvalue_ranges_printable() -> None:
    """Surface the toy eigenvalue ranges. Assertions only enforce that the
    leading eigenvalue is finite and positive; absolute scale is not
    meaningful at this size and depends on toy init."""
    A, S, n_tok = _accumulate_on_toy(n_rollouts=80)
    e_A = torch.linalg.eigvalsh(A)
    e_S = torch.linalg.eigvalsh(S)
    summary = {
        "n_tok": n_tok,
        "A.shape": tuple(A.shape),
        "S.shape": tuple(S.shape),
        "A.eig_min": float(e_A.min().item()),
        "A.eig_med": float(e_A.median().item()),
        "A.eig_max": float(e_A.max().item()),
        "S.eig_min": float(e_S.min().item()),
        "S.eig_med": float(e_S.median().item()),
        "S.eig_max": float(e_S.max().item()),
    }
    print()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<14s} = {v:.6e}")
        else:
            print(f"  {k:<14s} = {v}")
    assert math.isfinite(summary["A.eig_max"]) and summary["A.eig_max"] > 0
    assert math.isfinite(summary["S.eig_max"]) and summary["S.eig_max"] > 0
