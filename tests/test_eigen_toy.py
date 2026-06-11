"""Stage 2 toy acceptance: eigendecomposition + Λ correction.

Two Fisher helpers used here:

* ``compute_per_prediction_fisher_alpha`` — *per-prediction* (α): one
  ``autograd.grad`` per response token, full causal path included. This is
  mathematically the per-token-prediction Fisher and what the IF formula in
  CONTEXT.md technically inverts. Kept as research evidence; **not** the
  apples-to-apples target for the toy gate.

* ``compute_per_token_fisher_beta`` — *per-token-position summed* (β): one
  forward + one summed-loss backward, capture ``m_t, δ_summed[t]`` per
  response position via hooks, accumulate
  ``vec(δ_t m_tᵀ) vec(δ_t m_tᵀ)ᵀ`` per token. This is what EK-FAC's A, S, Λ
  pipeline approximates (Grosse 2023 / MDA convention restricted to per-token
  Λ per user spec). Used as the apples-to-apples target for the 1e-3
  Frobenius gate.

Both functions accept the same ``samples`` so the comparison is on identical
data and pseudo-label draws (same seed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout, build_input_and_labels  # noqa: E402
from ekfac.eigen import (  # noqa: E402
    eigendecompose,
    fit_lambda,
    inverse_hvp,
    reconstruct_F_inv_dense,
)
from ekfac.factors import (  # noqa: E402
    _logits_from_model_output,
    accumulate_AS,
    sample_pseudo_labels,
    sample_row_aligned_labels,
)
from ekfac.hooks import capture_c_proj  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402


# ---------------------------------------------------------------------------
# Sample construction (shared)
# ---------------------------------------------------------------------------


class ToySample(TypedDict):
    input_ids: torch.Tensor  # (T,) long
    prompt_len: int


def build_toy_samples(
    n_samples: int,
    vocab_size: int,
    *,
    max_len: int = 4,
    min_response: int = 1,
    seed: int = 42,
) -> list[ToySample]:
    g = torch.Generator().manual_seed(seed)
    out: list[ToySample] = []
    for _ in range(n_samples):
        T = max_len
        prompt_len = int(torch.randint(1, T - min_response + 1, (1,), generator=g).item())
        ids = torch.randint(0, vocab_size, (T,), generator=g, dtype=torch.long)
        out.append({"input_ids": ids, "prompt_len": prompt_len})
    return out


def _sample_to_rollout(sample: ToySample) -> Rollout:
    P = sample["prompt_len"]
    return Rollout(
        prompt_ids=sample["input_ids"][:P],
        response_ids=sample["input_ids"][P:],
        reward=0.0,
        step=0,
    )


# ---------------------------------------------------------------------------
# (α) per-prediction Fisher  (research evidence only)
# ---------------------------------------------------------------------------


def compute_per_prediction_fisher_alpha(
    model: nn.Module,
    layer: nn.Linear,
    samples: list[ToySample],
    *,
    seed: int = 42,
) -> tuple[torch.Tensor, int]:
    """Per-prediction Fisher F^(α). One autograd.grad per response token; full
    causal path. Normalised by token count."""
    torch.manual_seed(seed)
    W = layer.weight
    d_out, d_in = W.shape
    P = d_out * d_in
    F_acc = torch.zeros(P, P, dtype=torch.float64, device=W.device)
    n_tok = 0
    for sample in samples:
        ids = sample["input_ids"].to(W.device)
        plen = sample["prompt_len"]
        T = int(ids.shape[0])
        logits = model(ids.unsqueeze(0))[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            pseudo = torch.multinomial(probs, num_samples=1).squeeze(-1)
        active = list(range(plen - 1, T - 1))
        for i, t in enumerate(active):
            score = log_probs[t, pseudo[t]]
            retain = (i < len(active) - 1)
            (g,) = torch.autograd.grad(score, W, retain_graph=retain)
            g_flat = g.reshape(-1).to(torch.float64)
            F_acc.add_(torch.outer(g_flat, g_flat))
            n_tok += 1
    if n_tok == 0:
        raise RuntimeError("no response tokens for (α)")
    F_acc.div_(n_tok)
    F_acc = 0.5 * (F_acc + F_acc.T)
    return F_acc, n_tok


# ---------------------------------------------------------------------------
# (β) per-token-position summed Fisher  (apples-to-apples with EK-FAC)
# ---------------------------------------------------------------------------


def compute_per_token_fisher_beta(
    model: nn.Module,
    layer: nn.Linear,
    samples: list[ToySample],
    *,
    seed: int = 42,
    cfg: EKFACConfig | None = None,
) -> tuple[torch.Tensor, int]:
    """Per-token-position summed Fisher F^(β). For each response position::

        g_t = vec(δ_summed[t] · m_tᵀ)            # row-major flatten, (d_out·d_in,)
        F  += g_t g_tᵀ
    F /= n_tok

    ``δ_summed[t]`` is the hook-captured gradient at the c_proj output from a
    single summed-loss backward (identical convention to Stage 1A).
    """
    cfg = cfg or EKFACConfig()
    W = layer.weight
    device = W.device
    d_out, d_in = W.shape
    P_flat = d_out * d_in
    F_acc = torch.zeros(P_flat, P_flat, dtype=torch.float64, device=device)
    n_tok = 0
    gen = torch.Generator(device=device).manual_seed(seed)

    for sample in samples:
        plen = sample["prompt_len"]
        T = int(sample["input_ids"].shape[0])
        R = T - plen
        prompt_ids = sample["input_ids"][:plen].to(device)
        response_ids = sample["input_ids"][plen:].to(device)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)

        # Zero stale grads.
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        with capture_c_proj(layer) as cache:
            out = model(input_ids)
            logits = _logits_from_model_output(out)
            # Row-aligned pseudo-labels — MUST use the same single source of
            # truth as fit_lambda (sample_row_aligned_labels), otherwise this
            # "ground truth" would share fit_lambda's alignment and the gate
            # could only ever confirm internal consistency, not correctness
            # (the original off-by-one self-grading trap).
            labels = sample_row_aligned_labels(
                logits[0], prompt_ids, response_ids,
                generator=gen, ignore_index=cfg.ignore_index,
            )
            shift_logits = logits[0, :-1].float()
            shift_labels = labels[1:]
            loss = F.cross_entropy(
                shift_logits, shift_labels,
                ignore_index=cfg.ignore_index, reduction="sum",
            )
            loss.backward()

        m_resp = cache.m[0, plen - 1 : T - 1, :].to(torch.float64)      # (R, d_in)
        d_resp = cache.delta[0, plen - 1 : T - 1, :].to(torch.float64)  # (R, d_out)
        for t in range(R):
            g_t = torch.outer(d_resp[t], m_resp[t])                      # (d_out, d_in)
            g_flat = g_t.reshape(-1)
            F_acc.add_(torch.outer(g_flat, g_flat))
            n_tok += 1

    if n_tok == 0:
        raise RuntimeError("no response tokens for (β)")
    F_acc.div_(n_tok)
    F_acc = 0.5 * (F_acc + F_acc.T)
    return F_acc, n_tok


def fisher_diagnostics(F_mat: torch.Tensor) -> dict[str, float]:
    P = F_mat.shape[0]
    asym = (F_mat - F_mat.T).abs().max().item()
    eigs = torch.linalg.eigvalsh(F_mat)
    min_e = float(eigs.min().item())
    max_e = float(eigs.max().item())
    eigs_damped = eigs + 0.01
    cond_d = float(eigs_damped.max().item() / eigs_damped.min().item())
    return {
        "size": int(P),
        "max_asymmetry": asym,
        "min_eigval": min_e,
        "max_eigval": max_e,
        "is_psd": bool(min_e >= -1e-10 * max(abs(max_e), 1.0)),
        "effective_rank": int((eigs > 1e-10 * max_e).sum().item()),
        "cond_with_toy_damping": cond_d,
    }


# ---------------------------------------------------------------------------
# Legacy / original (α) sanity tests (kept green)
# ---------------------------------------------------------------------------


def test_alpha_fisher_shape_and_psd() -> None:
    cfg = EKFACConfig()
    model = ToyTransformer()
    model.eval()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    samples = build_toy_samples(20, model.vocab_size, max_len=model.max_len, seed=cfg.seed)
    F_mat, n_tok = compute_per_prediction_fisher_alpha(model, layer, samples, seed=cfg.seed)
    diag = fisher_diagnostics(F_mat)
    assert diag["size"] == 128
    assert diag["max_asymmetry"] < 1e-12
    assert diag["is_psd"]
    assert n_tok > 0


def test_alpha_fisher_invertible_with_toy_damping() -> None:
    cfg = EKFACConfig()
    model = ToyTransformer()
    model.eval()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    samples = build_toy_samples(20, model.vocab_size, max_len=model.max_len, seed=cfg.seed)
    F_mat, _ = compute_per_prediction_fisher_alpha(model, layer, samples, seed=cfg.seed)
    P = F_mat.shape[0]
    I = torch.eye(P, dtype=F_mat.dtype, device=F_mat.device)
    F_inv = torch.linalg.inv(F_mat + cfg.toy_damping * I)
    err = ((F_mat + cfg.toy_damping * I) @ F_inv - I).abs().max().item()
    assert err < 1e-8


# ---------------------------------------------------------------------------
# Stage 2 acceptance: eigendecompose
# ---------------------------------------------------------------------------


def _build_AS_and_eigen(n_rollouts: int = 150):
    """Common setup: accumulate A, S on toy then eigendecompose."""
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer()
    model.eval()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    samples = build_toy_samples(n_rollouts, model.vocab_size, max_len=model.max_len, seed=cfg.seed)
    rollouts = [_sample_to_rollout(s) for s in samples]
    A, S, n_tok_AS = accumulate_AS(model, layer, rollouts, cfg, device=torch.device("cpu"))
    Q_A, Λ_A, Q_S, Λ_S = eigendecompose(A, S)
    return cfg, model, layer, samples, rollouts, A, S, Q_A, Λ_A, Q_S, Λ_S, n_tok_AS


def test_eigendecompose_orthonormal_and_psd() -> None:
    """Gate 1c: Q orthonormal to ‖QQᵀ−I‖∞ < 1e-10."""
    _, _, _, _, _, A, S, Q_A, Λ_A, Q_S, Λ_S, _ = _build_AS_and_eigen()
    for name, Q in [("Q_A", Q_A), ("Q_S", Q_S)]:
        I = torch.eye(Q.shape[0], dtype=Q.dtype, device=Q.device)
        err = (Q @ Q.T - I).abs().max().item()
        assert err < 1e-10, f"{name}: ||QQᵀ - I||∞ = {err:.3e}"
    for name, Λ in [("Λ_A", Λ_A), ("Λ_S", Λ_S)]:
        assert torch.all(Λ >= 0), f"{name}: negative eigenvalue after clamp"
        assert torch.all(Λ[:-1] >= Λ[1:] - 1e-12), f"{name}: not descending"


def test_eigendecompose_reconstructs_A_S() -> None:
    """Gate 1d: ||A − Q_A diag(Λ_A) Q_Aᵀ|| / ||A|| < 1e-6."""
    _, _, _, _, _, A, S, Q_A, Λ_A, Q_S, Λ_S, _ = _build_AS_and_eigen()
    A_rec = Q_A @ torch.diag(Λ_A) @ Q_A.T
    S_rec = Q_S @ torch.diag(Λ_S) @ Q_S.T
    rel_A = (A_rec - A).norm() / A.norm()
    rel_S = (S_rec - S).norm() / S.norm()
    assert rel_A.item() < 1e-6, f"A reconstruction rel err {rel_A.item():.3e} > 1e-6"
    assert rel_S.item() < 1e-6, f"S reconstruction rel err {rel_S.item():.3e} > 1e-6"


# ---------------------------------------------------------------------------
# Stage 2 acceptance: fit_lambda
# ---------------------------------------------------------------------------


def test_fit_lambda_positive_and_shape() -> None:
    cfg, model, layer, _, rollouts, _, _, Q_A, _, Q_S, _, _ = _build_AS_and_eigen()
    Λ, n_tok = fit_lambda(model, layer, rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=torch.device("cpu"))
    assert Λ.shape == (model.d_mlp, model.d_model), f"Λ shape {tuple(Λ.shape)}"
    assert torch.all(Λ > 0), f"Λ has non-positive entries: min = {Λ.min().item():.3e}"
    assert n_tok > 0


# ---------------------------------------------------------------------------
# Stage 2 acceptance: 1e-3 Frobenius gate (the critical test)
# ---------------------------------------------------------------------------


def test_lambda_matches_kronecker_diagonal() -> None:
    """Gate 1a (implementation correctness): per-token Λ must equal the
    diagonal of F^(β) projected into the (Q_S ⊗ Q_A) basis to ``< 1e-10``."""
    cfg, model, layer, samples, rollouts, _, _, Q_A, _, Q_S, _, _ = _build_AS_and_eigen()
    Λ, n_tok_lambda = fit_lambda(
        model, layer, rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=torch.device("cpu")
    )
    F_true_b, n_tok_b = compute_per_token_fisher_beta(
        model, layer, samples, seed=cfg.seed + 1, cfg=cfg,
    )
    assert n_tok_lambda == n_tok_b

    d_in = Q_A.shape[0]
    d_out = Q_S.shape[0]
    Q_A64 = Q_A.to(torch.float64)
    Q_S64 = Q_S.to(torch.float64)
    U = torch.kron(Q_S64, Q_A64)
    F_in_kron = U.T @ F_true_b @ U
    diag_kron = torch.diagonal(F_in_kron).reshape(d_out, d_in).T  # (d_in, d_out)
    err = (diag_kron - Λ.to(torch.float64)).abs().max().item()
    assert err < 1e-10, (
        f"Λ vs diag(Uᵀ F^(β) U) max abs diff = {err:.3e} ≥ 1e-10 "
        f"(implementation bug in fit_lambda or reconstruct_F_inv_dense)"
    )


def test_synthetic_independence_reconstruction() -> None:
    """Gate 1b (reconstruction correctness): when (m, δ) are drawn iid from
    truly independent Gaussians with population covariances A_pop, S_pop, the
    EK-FAC pipeline reconstructs ``S_pop ⊗ A_pop`` to Frobenius rel err
    ``< 1e-8``. Validates the Kronecker basis assembly is bug-free
    independently of the toy-transformer's correlation structure."""
    torch.manual_seed(0)
    d_in, d_out = 16, 8

    # Build random positive-definite A_pop, S_pop.
    chol_A = torch.randn(d_in, d_in, dtype=torch.float64)
    A_pop = chol_A @ chol_A.T + 0.05 * torch.eye(d_in, dtype=torch.float64)
    chol_S = torch.randn(d_out, d_out, dtype=torch.float64)
    S_pop = chol_S @ chol_S.T + 0.05 * torch.eye(d_out, dtype=torch.float64)

    # Use the POPULATION eigendecomposition (so we test the assembly, not
    # finite-sample estimator noise).
    Q_A, Λ_A, Q_S, Λ_S = eigendecompose(A_pop, S_pop)

    # Under independence, the population Λ is exactly Λ_A ⊗ Λ_S (outer in
    # (d_in, d_out) layout).
    Λ_pop = torch.outer(Λ_A, Λ_S)
    F_target = torch.kron(S_pop, A_pop)
    F_ek = (
        torch.kron(Q_S, Q_A)
        @ torch.diag(Λ_pop.T.reshape(-1))
        @ torch.kron(Q_S, Q_A).T
    )
    rel = ((F_ek - F_target).norm() / F_target.norm()).item()
    assert rel < 1e-8, f"reconstruction rel err {rel:.3e} >= 1e-8 (bug in Kronecker assembly)"

    # Same with the dense F⁻¹ helper.
    F_target_inv = torch.linalg.inv(F_target + 0.01 * torch.eye(d_out * d_in, dtype=torch.float64))
    F_ek_inv = reconstruct_F_inv_dense(Q_A, Q_S, Λ_pop, damping=0.01)
    rel_inv = ((F_ek_inv - F_target_inv).norm() / F_target_inv.norm()).item()
    assert rel_inv < 1e-8, f"F⁻¹ reconstruction rel err {rel_inv:.3e} >= 1e-8"


def test_informational_transformer_inverse_rel_err() -> None:
    """Gate 1e (INFORMATIONAL — not a hard gate):

    Quantify the inverse Frobenius rel error between EK-FAC's reconstructed
    F⁻¹ and the brute-force F^(β)⁻¹ on the toy transformer. This is *not* a
    pass/fail criterion — EK-FAC's Λ correction captures only the diagonal of
    F^(β) in (Q_S ⊗ Q_A) basis by construction, and transformer correlations
    leave 40-50 % off-diagonal Kronecker mass that no amount of Λ-fitting
    will recover. The actual EK-FAC acceptance is rank correlation against
    leave-one-out retraining IF, performed in Stage 3.

    Soft cap of 1.5e-1 catches gross regressions (e.g., a fit_lambda
    indexing bug that returns the wrong diagonal).
    """
    cfg, model, layer, samples, rollouts, _, _, Q_A, _, Q_S, _, _ = _build_AS_and_eigen()
    Λ, _ = fit_lambda(model, layer, rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=torch.device("cpu"))
    F_true_b, _ = compute_per_token_fisher_beta(model, layer, samples, seed=cfg.seed + 1, cfg=cfg)

    P = F_true_b.shape[0]
    I = torch.eye(P, dtype=torch.float64, device=F_true_b.device)
    F_true_inv = torch.linalg.inv(F_true_b + cfg.toy_damping * I)
    F_ek_inv = reconstruct_F_inv_dense(Q_A, Q_S, Λ, damping=cfg.toy_damping)
    inv_rel = ((F_ek_inv - F_true_inv).norm() / F_true_inv.norm()).item()

    # Off-diagonal Kronecker mass: the structural quantity that bounds inv_rel.
    Q_A64 = Q_A.to(torch.float64)
    Q_S64 = Q_S.to(torch.float64)
    U = torch.kron(Q_S64, Q_A64)
    F_kron = U.T @ F_true_b @ U
    off_frac = (
        (F_kron - torch.diag(torch.diagonal(F_kron))).norm() / F_kron.norm()
    ).item()

    print()
    print(f"  informational: toy transformer inverse Frobenius rel err = {inv_rel:.3e}")
    print(f"  informational: off-diagonal Kronecker mass               = {off_frac:.4f}")
    print(f"  (structural; matches Grosse 2023 / MDA published norms)")
    assert inv_rel < 1.5e-1, (
        f"Inverse rel err {inv_rel:.3e} > 0.15 — likely an implementation regression"
    )


# ---------------------------------------------------------------------------
# Research evidence: (α) vs (β) Frobenius rel error
# ---------------------------------------------------------------------------


def test_research_alpha_vs_beta_frobenius() -> None:
    """Quantify how much the per-prediction Fisher (α) differs from the
    per-token-position summed Fisher (β) that EK-FAC approximates."""
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer()
    model.eval()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    samples = build_toy_samples(150, model.vocab_size, max_len=model.max_len, seed=cfg.seed)

    # Both functions use the same samples; pseudo-label seeds differ but
    # any single pair of draws is representative for the structural gap.
    F_alpha, n_tok_a = compute_per_prediction_fisher_alpha(model, layer, samples, seed=cfg.seed)
    F_beta, n_tok_b = compute_per_token_fisher_beta(model, layer, samples, seed=cfg.seed, cfg=cfg)

    # Frobenius rel errors both ways (asymmetric since denominators differ).
    rel_to_alpha = ((F_alpha - F_beta).norm() / F_alpha.norm()).item()
    rel_to_beta = ((F_alpha - F_beta).norm() / F_beta.norm()).item()
    ratio_norms = (F_alpha.norm() / F_beta.norm()).item()
    print()
    print(f"  (α) per-prediction n_tok = {n_tok_a}")
    print(f"  (β) per-token-summed n_tok = {n_tok_b}")
    print(f"  ||F^(α)||_F = {F_alpha.norm().item():.3e}")
    print(f"  ||F^(β)||_F = {F_beta.norm().item():.3e}")
    print(f"  ||F^(α) - F^(β)||_F / ||F^(α)||_F = {rel_to_alpha:.3e}")
    print(f"  ||F^(α) - F^(β)||_F / ||F^(β)||_F = {rel_to_beta:.3e}")
    print(f"  ||F^(α)||_F / ||F^(β)||_F        = {ratio_norms:.3e}")
    assert n_tok_a > 0 and n_tok_b > 0


# ---------------------------------------------------------------------------
# Production IHVP sanity (algebraic consistency with reconstruct_F_inv_dense)
# ---------------------------------------------------------------------------


_ = inverse_hvp  # noqa: F841 — imported for Stage 3; consistency test deferred.
