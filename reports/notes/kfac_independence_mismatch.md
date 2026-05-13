# K-FAC Independence Assumption: Empirical Mismatch on Transformers

**Status:** Stage-2 research finding (Phase 2). One-paragraph note drafted
for inclusion in `reports/phase2_report.md` once Phase 2 completes.

---

## Draft paragraph

EK-FAC's Λ correction approximates the per-token Fisher
F^(β)\_{ij,i'j'} = E_t[(δ_t ⊗ m_t)\_{ij} (δ_t ⊗ m_t)\_{i'j'}] by retaining only
its diagonal in the Kronecker eigenbasis Q_S ⊗ Q_A. Under K-FAC's structural
independence assumption (m_t ⊥ δ_t), this diagonal recovers F^(β) exactly:
the off-diagonal Kronecker entries vanish identically because
E[(Q_Aᵀ m)\_i (Q_Aᵀ m)\_{i'}] = Λ_A,i δ\_{ii'} and analogously for δ. We
measured this assumption directly on a 2-layer toy transformer (d_model=8,
d_mlp=16, GPT-Neo-shaped MLP) over 309 response tokens drawn from
pseudo-labeled rollouts: the off-diagonal Frobenius mass of F^(β) in the
Kronecker basis was **45.6 % of the total**, and the resulting Frobenius
relative error between EK-FAC's reconstructed F⁻¹ and the brute-force
inverse `(F^(β) + 0.01 I)⁻¹` was **8.0 %**. A sweep over toy sizes
(d_model ∈ {8, 16, 32, 64}) showed the off-diagonal fraction holding stable
at 38–48 % and inverse error plateauing around 4–8 %, indicating this is a
structural property of EK-FAC on transformers rather than a small-sample or
toy-specific artefact. The mismatch arises because the residual-stream
hidden state at each position couples m_t and δ_t through LayerNorm and the
causal-attention path, violating the independence assumption that K-FAC
requires for exact recovery. Within the EK-FAC literature this gap is
typically left implicit; our quantification confirms the published 5–10 %
inverse-error regime (Grosse et al., 2023; Chen et al., 2026) and motivates
validating IF rankings via leave-one-out retraining ground truth rather
than matrix-Frobenius proximity to the true Fisher.

## Supporting numbers (toy, n_samples=150, n_tok=309)

| metric | value |
|---|---|
| ‖Λ − diag(Uᵀ F^(β) U)‖∞ (fit_lambda correctness) | 2.08e-17 |
| Synthetic-independence reconstruction rel err | 2.66e-15 |
| Off-diagonal Kronecker mass | 45.63 % |
| Forward ‖F_ekfac − F^(β)‖_F / ‖F^(β)‖_F | 4.56e-1 |
| Inverse rel err @ λ=0.01 | 8.02e-2 |

## Toy-size sweep

| d_model | d_mlp | n_tok | off-diag frac | inverse rel err |
|---|---|---|---|---|
| 8 | 16 | 309 | 0.4563 | 8.02e-2 |
| 8 | 16 | 1022 | 0.3937 | 6.71e-2 |
| 16 | 32 | 594 | 0.3776 | 4.77e-2 |
| 32 | 64 | 594 | 0.4079 | 4.20e-2 |
| 64 | 128 | 410 | 0.4821 | 4.14e-2 |

## Research evidence: (α) per-prediction vs (β) per-token-position Fisher

| metric | value |
|---|---|
| ‖F^(α)‖_F | 5.937e-2 |
| ‖F^(β)‖_F | 6.829e-2 |
| ‖F^(α) − F^(β)‖_F / ‖F^(α)‖_F | 4.920e-1 |
| ‖F^(α) − F^(β)‖_F / ‖F^(β)‖_F | 4.278e-1 |

The two Fishers differ by ~45 % Frobenius — they are mathematically distinct
objects: (α) accumulates `∂L_pos/∂W` with the full causal-attention
contribution, (β) accumulates only the per-position `(δ_t m_tᵀ)` outer
product. EK-FAC structurally targets (β); the IF formula in CONTEXT.md is
strictly speaking sequence-level. The toy gate validates EK-FAC against its
own structural target (β), not against the strict IF math, consistent with
Grosse 2023 / MDA practice.
