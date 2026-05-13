# LOO Retraining vs IF Score: Investigation and Outcome

**Status:** Phase 2 research finding. Referenced by `reports/phase2_report.md`
Section 3 and inlined briefly in the thesis methods section.

---

## TL;DR

We attempted to validate the EK-FAC IF pipeline against leave-one-out
retraining ground truth on a toy transformer, under two configurations
(overparameterised and underparameterised). Both failed Pearson > 0.7. A
brute-force per-sequence Fisher inverse control (zero EK-FAC approximation
error; direct `np.linalg.inv` of the 32×32 or 128×128 empirical Fisher)
also failed. This isolates the failure to the **IF formula's finite-ε
linear approximation**, not our pipeline; the observation matches Bae et al.
(2022) §3. We adopt real-data self-influence sanity as the at-scale
validation, following Grosse et al. (2023).

---

## Experiments

### 1. Overparameterised toy (1568 params, N=20 rollouts)

* Architecture: d_model=8, d_mlp=16, n_layer=2, n_heads=2, vocab=20, T_max=4.
* Optimiser: Adam, lr=1e-2, no weight decay.
* Eval f: log π_θ(y_eval | x_eval) on a held-out 21st random rollout.

| K | final loss | EK-FAC Pearson | Brute-force F_per_seq⁻¹ Pearson | Notes |
|---|---|---|---|---|
| 10  | 2.37 | +0.638 | **+0.542** | Under-trained; ∇L(θ*) is still large. |
| 50  | 0.41 | +0.062 | +0.084 | Transitional regime. |
| 200 | 0.07 | +0.061 | **−0.203** | Interpolating; 19/20 rollouts memorised. |

Diagnostic at K=200: 19/20 training rollouts have per-token log π ≈ 0
(memorised), `‖s_m‖` collapses to 1e-3–1e-4 on training data while
`‖s_eval‖` remains 3.4 on the held-out rollout. The IF score
`I = ⟨s_m, p⟩` is therefore near-zero for all training rollouts because
the *per-sample gradients* vanish at the interpolating optimum — the IF
linear approximation predicts no first-order effect, while LOO retraining
still produces large finite changes because removing one rollout shifts
the entire interpolating manifold.

### 2. Underparameterised toy (356 params, N=100 rollouts)

* Architecture: d_model=4, d_mlp=8, n_layer=1, n_heads=1, vocab=20, T_max=4.
* Optimiser: AdamW, lr=1e-2, **weight_decay=1e-2** to prevent interpolation.
* N=100 rollouts deliberately puts the toy in a regime where the parameter
  count is *not* huge relative to data points (~200 effective response
  tokens vs 356 params, ratio ≈ 0.6).

| K | final loss | EK-FAC Pearson | Brute-force F_per_seq⁻¹ Pearson | Notes |
|---|---|---|---|---|
| 50  | 2.65 | +0.193 | (not measured) | Under-trained: still descending. |
| 200 | 1.86 | +0.023 | **−0.031** | AdamW equilibrium plateau (`∇L = −λθ*`, not zero). |

At K=200, AdamW has reached its regularised equilibrium where the gradient
of the unregularised loss is balanced by weight decay, *not* zero. The IF
stationarity assumption `∇L(θ*) = 0` is violated by the regularisation
term. But even so, the brute-force F⁻¹ control still measures the *exact
linear response*, and it too gives Pearson −0.03 — so the failure is not
"our pipeline uses the wrong Fisher" but rather the linear approximation
itself does not match finite-ε LOO in this regime.

---

## What this means

The brute-force F⁻¹ control is the decisive datum: it removes EK-FAC's
structural Kronecker + diag-Λ approximation entirely, replacing it with
direct inversion of the empirical per-sequence Fisher. At toy scale this
is feasible because the parameter dimension `d_out × d_in` is small enough
(32 or 128) to invert directly. In all configurations tested, brute-force
F⁻¹ produced approximately the same Pearson as EK-FAC, never reaching the
0.7 target. The gap is therefore *not* attributable to our pipeline.

The remaining sources of mismatch are intrinsic to the IF/LOO comparison:

1. **Finite-ε linearisation error.** IF is the derivative `df/dε`
   evaluated at ε=0; LOO retraining measures the finite-difference
   `f(ε = 1/N) − f(0)`. For non-convex losses with attention/LayerNorm
   curvature, the O(ε²) higher-order terms can dominate when ε = 1/N is
   1–5 %. Bae et al. (2022) §3 demonstrates this discrepancy quantitatively
   on shallow CNNs and proposes Proximal Bregman Response Functions (PBRF)
   as an alternative ground truth; we did not pursue PBRF given the
   thesis scope.

2. **Non-stationary AdamW optimum.** AdamW with non-zero weight decay does
   not converge to `∇L(θ*) = 0` but to `∇L(θ*) + λθ* = 0`. The classical
   IF derivation assumes the unregularised gradient vanishes at θ*. The
   correction term is small at our scale but adds noise.

3. **Toy curvature non-uniformity.** Even at small dimensions, attention
   + LayerNorm + GELU produce a loss landscape with strong cross-token
   correlations (Phase 2 reports off-diagonal Kronecker mass of 45 %). LOO
   retraining samples the *exact* curved landscape; IF linearises around
   θ*. Disagreement is expected.

---

## What we use instead

Per Grosse et al. (2023) and Wu et al. (2024), at-scale EK-FAC IF is
validated via *self-influence sanity*: under the KL-RL sign convention,
`I(z_m, z_m) = −s_mᵀ F⁻¹ s_m ≤ 0` should be large in magnitude and
dominate cross-IFs `I(z_n, z_m)` for `n ≠ m`. We computed this on
`data/rollouts/step_0500.pt` (32 PPO rollouts from GPT-Neo-125M, real
production scale) for 5 randomly-chosen self-targets:

| self idx m | I_self (signed) | rank of \|I_self\| in 32 |
|---|---|---|
| z_0 | −1.217e+4 | **1/32** |
| z_5 | −1.776e+4 | **1/32** |
| z_10 | −1.403e+4 | **1/32** |
| z_15 | −8.127e+3 | **1/32** |
| z_20 | −6.465e+3 | **1/32** |

All 5 self-IFs are negative (consistent with KL-RL convention) and all
rank #1 in the |I| distribution, with magnitudes 50–100× the next-largest
cross-IF. This is the diagnostic the published literature relies on; we
adopt it as the production-scale acceptance test.

---

## References

* Koh & Liang 2017, *Understanding Black-Box Predictions via Influence
  Functions*. Validates IF on logistic regression and shallow CNNs that
  are explicitly trained with early stopping to avoid interpolation.
* Bae, Ng, Lo, Ghassemi, Grosse 2022, *If Influence Functions Are the
  Answer, Then What Is the Question?* Documents the finite-ε / non-convex
  failure mode quantitatively; proposes PBRF.
* Grosse et al. 2023, *Studying Large Language Model Generalization with
  Influence Functions* (Anthropic EK-FAC for LLM IF). Uses self-influence
  and downstream-task agreement as validation, not LOO retraining.
* Wu et al. 2024 (DPO toxicity IF). Same validation strategy.
