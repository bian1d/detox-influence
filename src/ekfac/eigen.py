"""Stage 1B: eigendecomposition + corrected diagonal Λ for EK-FAC.

Conventions (consistent throughout the pipeline):

    layer.weight shape:                  (d_out, d_in)     # PyTorch nn.Linear
    A = E_t[m_t m_tᵀ]      shape:        (d_in,  d_in)
    S = E_t[δ_t δ_tᵀ]      shape:        (d_out, d_out)

    A = Q_A diag(λ_A) Q_Aᵀ,   λ_A descending
    S = Q_S diag(λ_S) Q_Sᵀ,   λ_S descending

    Λ_corrected[i, j] = E_t[(Q_Aᵀ m_t)_i² (Q_Sᵀ δ_t)_j²]   shape (d_in, d_out)

Per-token expectation for Λ (square-then-sum-over-tokens, normalise by total
token count). This differs from MDA's per-batch sum-then-square convention,
which collapses to a per-sequence statistic when batch_size = 1; the per-token
convention matches the user's Stage 2 spec and makes the toy-gate
comparison internally consistent with the per-token A, S.

EK-FAC reconstruction (matrix form on gradient g of shape (d_out, d_in)):

    F⁻¹ g  =  Q_S · (Q_Sᵀ g Q_A  ./  (Λᵀ + damping))  · Q_Aᵀ

Row-major Kronecker product gives an equivalent flat form ``F⁻¹ = U
diag(1/(Λᵀ.flatten() + damping)) Uᵀ`` with ``U = kron(Q_S, Q_A)``; the
flat form is only used in the toy gate where we need an explicit
(d_out·d_in)² matrix to compare against a brute-force inverse.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ekfac.config import EKFACConfig
from ekfac.data import Rollout
from ekfac.factors import _logits_from_model_output, sample_row_aligned_labels
from ekfac.hooks import capture_c_proj


def eigendecompose(
    A: torch.Tensor,
    S: torch.Tensor,
    *,
    orthonormal_tol: float = 1e-5,
    psd_tol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigendecompose A and S; return descending-sorted eigenpairs.

    Returns:
        Q_A:   ``(d_in,  d_in)``  orthonormal eigenvectors of A (columns).
        Λ_A:   ``(d_in,)``         descending eigenvalues, clamped ≥ 0.
        Q_S:   ``(d_out, d_out)`` orthonormal eigenvectors of S.
        Λ_S:   ``(d_out,)``        descending eigenvalues, clamped ≥ 0.

    Raises:
        RuntimeError: large negative eigenvalues (≤ -psd_tol · max|λ|) or
        Q not orthonormal to ``orthonormal_tol``.
    """
    Q_A, Λ_A = _eigh_descending(A, name="A", orthonormal_tol=orthonormal_tol, psd_tol=psd_tol)
    Q_S, Λ_S = _eigh_descending(S, name="S", orthonormal_tol=orthonormal_tol, psd_tol=psd_tol)
    return Q_A, Λ_A, Q_S, Λ_S


def _eigh_descending(
    M: torch.Tensor,
    *,
    name: str,
    orthonormal_tol: float,
    psd_tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper: eigendecompose one symmetric matrix, validate, return descending."""
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"{name}: not square, shape={tuple(M.shape)}")
    Λ, Q = torch.linalg.eigh(M)
    # eigh returns ascending; flip to descending for consistency.
    Λ = Λ.flip(0)
    Q = Q.flip(1)
    # Large negative eigenvalues are PSD violations; tiny negative roundoff is OK.
    max_abs = float(Λ.abs().max().item())
    neg_tol = psd_tol * max(max_abs, 1.0)
    min_eig = float(Λ.min().item())
    if min_eig < -neg_tol:
        raise RuntimeError(
            f"{name}: large negative eigenvalue {min_eig:.3e} "
            f"(tolerance {-neg_tol:.3e}); matrix not PSD"
        )
    # Orthonormality of Q.
    I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
    err = (Q @ Q.T - I).abs().max().item()
    if err > orthonormal_tol:
        raise RuntimeError(
            f"{name}: Q not orthonormal, max|QQᵀ - I| = {err:.3e} > {orthonormal_tol:.0e}"
        )
    # Clamp tiny negative roundoff to zero.
    Λ = Λ.clamp(min=0)
    return Q, Λ


def fit_lambda(
    model: nn.Module,
    layer: nn.Linear,
    rollouts: Iterable[Rollout],
    cfg: EKFACConfig,
    *,
    Q_A: torch.Tensor,
    Q_S: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Per-token corrected diagonal Λ in the (Q_A, Q_S) Kronecker eigenbasis.

    Second pass over the rollouts (same pseudo-label loss as Stage 1A). For
    each response token t::

        a_proj_t = Q_Aᵀ m_t        # (d_in,)
        d_proj_t = Q_Sᵀ δ_t        # (d_out,)
        Λ[i, j] += a_proj_t[i]² · d_proj_t[j]²

    Then ``Λ /= n_tok``.

    Returns:
        Λ:     (d_in, d_out) fp64.
        n_tok: total response tokens contributed.
    """
    d_in = Q_A.shape[0]
    d_out = Q_S.shape[0]
    if Q_A.shape != (d_in, d_in) or Q_S.shape != (d_out, d_out):
        raise ValueError(
            f"Q shape mismatch: Q_A {tuple(Q_A.shape)}, Q_S {tuple(Q_S.shape)}"
        )

    Λ_sum = torch.zeros(d_in, d_out, dtype=cfg.dtype_factors, device=device)
    n_tok = 0
    # Seed offset so Stage 1B's pseudo labels differ from Stage 1A's by
    # default. (For reproducibility within one stage, the offset is fixed.)
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    Q_A_d = Q_A.to(cfg.dtype_factors)
    Q_S_d = Q_S.to(cfg.dtype_factors)

    was_training = model.training
    model.eval()
    try:
        for rollout in rollouts:
            n_tok += _fit_lambda_one(
                model, layer, rollout, cfg,
                device=device, Q_A=Q_A_d, Q_S=Q_S_d, Λ_sum=Λ_sum, gen=gen,
                d_in=d_in, d_out=d_out,
            )
    finally:
        if was_training:
            model.train()
    if n_tok == 0:
        raise RuntimeError("no response tokens accumulated in fit_lambda")
    Λ = Λ_sum / n_tok
    return Λ, n_tok


def _fit_lambda_one(
    model: nn.Module,
    layer: nn.Linear,
    rollout: Rollout,
    cfg: EKFACConfig,
    *,
    device: torch.device,
    Q_A: torch.Tensor,
    Q_S: torch.Tensor,
    Λ_sum: torch.Tensor,
    gen: torch.Generator,
    d_in: int,
    d_out: int,
) -> int:
    """One rollout: forward+summed-backward; accumulate per-token Λ contribution."""
    import torch.nn.functional as F

    P = rollout.prompt_len
    R = rollout.response_len
    T = P + R
    input_ids = torch.cat([rollout.prompt_ids, rollout.response_ids]).to(device).unsqueeze(0)

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    with capture_c_proj(layer) as cache:
        out = model(input_ids)
        logits = _logits_from_model_output(out)
        # Row-aligned Fisher pseudo-labels (same single source of truth as Stage 1A).
        labels = sample_row_aligned_labels(
            logits[0],
            rollout.prompt_ids.to(device),
            rollout.response_ids.to(device),
            generator=gen,
            ignore_index=cfg.ignore_index,
        )
        shift_logits = logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32))
        shift_labels = labels[1:]
        loss = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index=cfg.ignore_index, reduction="sum",
        )
        loss.backward()

    if cache.m is None or cache.delta is None:
        raise RuntimeError("hook capture failed in fit_lambda")
    m_resp = cache.m[0, P - 1 : T - 1, :].to(cfg.dtype_factors)      # (R, d_in)
    d_resp = cache.delta[0, P - 1 : T - 1, :].to(cfg.dtype_factors)  # (R, d_out)

    a_proj = m_resp @ Q_A    # (R, d_in)
    d_proj = d_resp @ Q_S    # (R, d_out)
    # Per-token contribution: Σ_t a_proj[t,:]² ⊗ d_proj[t,:]²  ≡  a_proj².T @ d_proj²
    Λ_sum.add_(a_proj.pow(2).T @ d_proj.pow(2))
    return R


def reconstruct_F_inv_dense(
    Q_A: torch.Tensor,
    Q_S: torch.Tensor,
    Lambda: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Materialise EK-FAC's reconstructed F⁻¹ as a dense (d_out·d_in, d_out·d_in)
    matrix. Used **only** for the toy gate comparison.

    ``F_ekfac⁻¹ = U diag(1/(Λᵀ.flatten() + damping)) Uᵀ``
    where ``U = kron(Q_S, Q_A)`` (row-major: column ``j*d_in + i`` is
    ``Q_S[:, j] ⊗ Q_A[:, i]``, with eigenvalue ``Λ[i, j]``).
    """
    d_in = Q_A.shape[0]
    d_out = Q_S.shape[0]
    if Lambda.shape != (d_in, d_out):
        raise ValueError(
            f"Lambda shape {tuple(Lambda.shape)} != (d_in, d_out)=({d_in}, {d_out})"
        )
    dtype = Lambda.dtype
    device = Lambda.device
    U = torch.kron(Q_S.to(dtype).to(device), Q_A.to(dtype).to(device))
    inv_diag = 1.0 / (Lambda.T.reshape(-1) + damping)  # length d_out * d_in
    return U @ torch.diag(inv_diag) @ U.T


def inverse_hvp(
    g: torch.Tensor,
    Q_A: torch.Tensor,
    Q_S: torch.Tensor,
    Lambda: torch.Tensor,
    *,
    damping_floor: float,
    damping_alpha: float,
) -> torch.Tensor:
    """Production IHVP: apply F_ekfac⁻¹ to one gradient ``g`` of shape
    ``(d_out, d_in)``. Two-level damping per MDA:

        denom = max(Λ + damping_alpha · Λ̄, damping_floor)
    """
    if g.shape != (Q_S.shape[0], Q_A.shape[0]):
        raise ValueError(
            f"g shape {tuple(g.shape)} != (d_out, d_in)="
            f"({Q_S.shape[0]}, {Q_A.shape[0]})"
        )
    g_proj = Q_S.T @ g @ Q_A                       # (d_out, d_in)
    denom = Lambda + damping_alpha * Lambda.mean()  # (d_in, d_out)
    denom = denom.clamp(min=damping_floor)
    g_scaled = g_proj / denom.T                     # broadcast (d_out, d_in)
    return Q_S @ g_scaled @ Q_A.T


def inverse_hvp_additive(
    g: torch.Tensor,
    Q_A: torch.Tensor,
    Q_S: torch.Tensor,
    Lambda: torch.Tensor,
    *,
    damping: float,
) -> torch.Tensor:
    """Toy IHVP with uniform additive damping. Used in the toy gate so the
    EK-FAC inverse and ``inv(F + λI)`` use the *same* damping scheme:

        F_ekfac⁻¹ g = Q_S · (Q_Sᵀ g Q_A) / (Λᵀ + λ) · Q_Aᵀ

    The production ``inverse_hvp`` floor-and-add scheme would otherwise add
    a comparison-asymmetric source of error.
    """
    if g.shape != (Q_S.shape[0], Q_A.shape[0]):
        raise ValueError(
            f"g shape {tuple(g.shape)} != (d_out, d_in)="
            f"({Q_S.shape[0]}, {Q_A.shape[0]})"
        )
    g_proj = Q_S.T @ g @ Q_A
    denom = Lambda + damping                       # (d_in, d_out)
    g_scaled = g_proj / denom.T                    # (d_out, d_in)
    return Q_S @ g_scaled @ Q_A.T
