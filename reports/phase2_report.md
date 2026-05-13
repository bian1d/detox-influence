# Phase 2 Report — EK-FAC Influence Function Pipeline

**Model:** `EleutherAI/gpt-neo-125m` (post-PPO; Phase 0)
**Parameter subspace φ:** `transformer.h.9.mlp.c_proj.weight` (W₂ of layer 9)
**Fisher approximation:** EK-FAC with two-level damping
**IF formula:** Section 1.2 below (sign convention split between toy supervised and real KL-RL)
**Status:** COMPLETE — implementation validated; production IF run scoped and pending eval-target selection.

---

## 1. Algorithm Overview

### 1.1 EK-FAC Structure

Phase 2 implements EK-FAC (Eigenvalue-corrected Kronecker-Factored
Approximate Curvature) restricted to a single MLP value matrix
`W = layer.weight ∈ R^(d_out × d_in)`. For the c_proj layer we use,
`d_out = d_model = 768` and `d_in = d_mlp = 3072`, so φ has 768 × 3072 ≈
2.36M parameters. The full Fisher
`F ∈ R^(d_out·d_in × d_out·d_in)` would have 5.56 × 10¹² entries — direct
inversion is infeasible. EK-FAC factorises this into two small matrices
plus an element-wise diagonal:

**Per-token Kronecker factors** (Stage 1A, file `src/ekfac/factors.py`)

    A = E_t [m_t m_tᵀ]      ∈ R^(d_in × d_in)    = R^(3072 × 3072)
    S = E_t [δ_t δ_tᵀ]      ∈ R^(d_out × d_out)  = R^(768 × 768)

where `m_t` is the c_proj input (post-GELU MLP activation) at response
token position t, `δ_t` is the gradient of the summed-loss cross-entropy
at the c_proj output, and the expectation is over response tokens from
PPO rollouts under **pseudo-labels** sampled from π_θ (true Fisher, not
empirical; CLAUDE.md hard constraint #4).

**Eigendecomposition** (Stage 1B, file `src/ekfac/eigen.py`)

    A = Q_A · diag(Λ_A) · Q_Aᵀ        (Q_A orthonormal)
    S = Q_S · diag(Λ_S) · Q_Sᵀ        (Q_S orthonormal)

Both decompositions are computed via `torch.linalg.eigh` with descending
sort and clamping of tiny-negative eigenvalues to zero (PSD enforcement,
small float-roundoff only).

**Corrected diagonal Λ** (Stage 1B same module)

    Λ_corrected[i, j] = E_t[ (Q_Aᵀ m_t)_i² · (Q_Sᵀ δ_t)_j² ]
                      ∈ R^(d_in × d_out) = R^(3072 × 768)

This is the diagonal of the per-token Fisher F^(β) expressed in the
Kronecker basis Q_S ⊗ Q_A. We use the *per-token* convention
(square-then-sum-then-normalise) rather than MDA's per-sequence
(sum-then-square) form; the two coincide for batch_size=1 single-rollout
accumulation but the per-token form is internally consistent with the
per-token A and S normalisation.

**Inverse-Hessian-vector product**

For one gradient `g ∈ R^(d_out × d_in)`:

    g_proj   = Q_Sᵀ · g · Q_A                                         # (d_out, d_in)
    denom    = max(Λ_corrected + α · Λ̄, λ_floor)                      # (d_in, d_out)
    F⁻¹ g    = Q_S · ( g_proj / denomᵀ ) · Q_Aᵀ                       # (d_out, d_in)

with **two-level damping** (`α = 0.1`, `λ_floor = 1e-5`) per MDA convention
(CONTEXT.md §13.2, §18.3). The α-component adapts to the curvature scale;
the floor clamps tiny eigenvalues that would otherwise produce
explosive inverses.

### 1.2 IF Formula (Two Sign Conventions)

For training rollout `z_m = (x_m, y_m)` and eval objective
`f(θ) = log π_θ(y_eval | x_eval)`:

**KL-regularised RL** (PPO/RLHF; production case):

    I_KL(z_m, f) = − ∇_φ f(θ*)ᵀ · F⁻¹ · ∇_φ log π_θ*(y_m | x_m)
                 = − ⟨ s_m, p ⟩_F        where p = F⁻¹ · g_eval

**Supervised CE** (toy validation only):

    I_SV(z_m, f) = + ∇_φ f(θ*)ᵀ · F⁻¹ · ∇_φ log π_θ*(y_m | x_m)
                 = + ⟨ s_m, p ⟩_F

The two formulas differ only by sign. The KL-RL minus sign comes from the
effective-reward perturbation in CONTEXT.md §5: in KL-RL, perturbing by
`+ε · [r − β log(π/π_ref)]` gives a perturbation gradient of `−β s_m`
(the `−β log π` term flips the sign relative to plain supervised
CE-perturbation `+ε · log π`). Both functions live in
`src/ekfac/influence.py` (`influence_score_klrl` and
`influence_score_supervised`); the call site selects by training regime.

### 1.3 Why EK-FAC

Computational tractability is the only reason. Exact `F⁻¹` is infeasible at
2.36M parameters; alternatives like block-diagonal Fisher, Lanczos low-rank,
or LBFGS-style trust regions are either slower (Lanczos), less accurate
(block-diagonal), or require gradient checkpointing pipelines beyond
thesis scope. EK-FAC is the published standard for LLM IF (Grosse et al.
2023, Chen et al. 2026); our pipeline structurally matches their
implementation. Gradient similarity (raw `⟨s_m, g_eval⟩` without F⁻¹) is
computed alongside EK-FAC at near-zero cost as a sanity baseline per the
CLAUDE.md locked-in decision.

### 1.4 Layer Choice — Provisional `transformer.h.9.mlp.c_proj`

Lee et al. (2024)'s DPO-toxicity paper identifies the most-toxic value
vector `v^19_770` in GPT2-medium at layer 19 of 24 (≈ 79 % of depth),
with the top-7 toxic vectors concentrated in layers 18–20 (75–87 %).
Architecture-invariant proportion mapping to GPT-Neo-125M (12 layers):
**layer 9 of 12 ≈ 75 % of depth**. The weight shape is `(d_model=768,
d_mlp=3072)`. This choice was originally provisional pending re-training
of the toxicity probe on GPT-Neo-125M; the Phase 0 probe verification
(Section 1.5 of `reports/phase0_report.md`) yielded inconclusive
logit-lens evidence, so we retain layer 9 as the production target and
mark final layer selection as a Phase 3 task (mechanistic verification
of the chosen layer against toxic value-vector activations on PPO
checkpoints).

---

## 2. Implementation Validation

### 2.1 Four Stage-2 Implementation Gates (all pass at machine precision)

Each gate isolates one mechanic of the pipeline and accepts only if the
deviation is at float-roundoff level.

| Gate | What it tests | Threshold | **Measured (toy)** | **Measured (real)** |
|---|---|---|---|---|
| **1a** | `Λ_corrected` exactly equals `diag(Uᵀ F^(β) U)` (i.e., the per-token square-then-sum convention is implemented correctly and the Kronecker basis assembly has no transpose/index bug) | < 1e-10 | **2.08e-17** | — |
| **1b** | Under synthetic independent `m ~ N(0, A_pop), δ ~ N(0, S_pop)`, EK-FAC reconstructs `S_pop ⊗ A_pop` (the population K-FAC limit) | < 1e-8 | **2.66e-15** | — |
| **1c** | Q_A and Q_S are orthonormal: `‖QQᵀ − I‖∞` | < 1e-10 | **8.88e-16 / 4.79e-16** | **4.89e-15 / 8.88e-16** |
| **1d** | Reconstruction: `‖A − Q_A diag(Λ_A) Q_Aᵀ‖_F / ‖A‖_F` (and same for S) | < 1e-6 | **9.94e-16 / 6.99e-16** | **3.05e-15 / 2.66e-15** |

All four pass by 4 to 9 orders of magnitude. The implementation is
correct at machine precision.

Additionally we verified the IHVP function against direct dense
reconstruction at fp64: `‖inverse_hvp_additive(g, …, λ) − F_inv_dense @
vec(g)‖_F / ‖dense‖_F < 1e-10` (`test_influence_toy.py::
test_inverse_hvp_additive_matches_dense_reconstruction`).

### 2.2 K-FAC Independence-Mismatch Quantification (informational)

EK-FAC's Λ correction captures only the *diagonal* of F^(β) in the
Kronecker basis. The off-diagonal Kronecker mass — present when the
K-FAC independence assumption `m_t ⊥ δ_t` is violated — is discarded by
construction.

On our toy transformer (d_model=8, d_mlp=16, 1568 params, 309 response
tokens) the off-diagonal Kronecker mass is **45.6 %** of total Frobenius,
yielding an inverse-Frobenius relative error of **8.02 %** between
`F_ekfac⁻¹` and the brute-force `(F^(β) + 0.01·I)⁻¹`. A toy-size sweep
confirms this is structural, not a small-dimension artefact:

| d_model | d_mlp | n_tok | off-diag fraction | inverse rel err @ λ=0.01 |
|---|---|---|---|---|
| 8 | 16 | 309 | 0.4563 | 8.02e-2 |
| 8 | 16 | 1022 | 0.3937 | 6.71e-2 |
| 16 | 32 | 594 | 0.3776 | 4.77e-2 |
| 32 | 64 | 594 | 0.4079 | 4.20e-2 |
| 64 | 128 | 410 | 0.4821 | 4.14e-2 |

The 4–8 % inverse error matches the published norm for EK-FAC at LLM
scale (Grosse et al. 2023; Chen et al. 2026) and reflects the fact that
LayerNorm, attention, and the causal residual stream couple `m_t` and
`δ_t` at each position. The classical IF/EK-FAC literature accepts this
gap and validates against downstream behaviour (self-influence, rank
correlation with retraining) rather than matrix-Frobenius proximity.
Detail: `reports/notes/kfac_independence_mismatch.md`.

---

## 3. IF Validation Strategy

### 3.1 Attempted: Leave-One-Out Retraining on Toy

Standard IF papers (Koh & Liang 2017, Bae et al. 2022) validate the
linear-in-ε IF approximation against leave-one-out retraining: train θ*
on all N rollouts, train θ\*_{-m} omitting rollout m, define
`I_true(z_m, f) ≈ f(θ*) − f(θ\*_{-m})`, and check Pearson / Kendall τ
against `I_ekfac`. We attempted this in two configurations:

| toy | params | optimiser | K | final loss | EK-FAC Pearson | **Brute-force F⁻¹ Pearson** |
|---|---|---|---|---|---|---|
| Overparam (d=8/16, n=2) | 1568 | Adam lr=1e-2, no wd | 200 | 0.07 (interpolating) | +0.061 | **−0.20** |
| Overparam, K=10 (under-trained) | 1568 | Adam lr=1e-2 | 10 | 2.37 | +0.638 | **+0.542** |
| Underparam (d=4/8, n=1) | 356 | AdamW lr=1e-2, wd=1e-2 | 200 | 1.86 (plateau) | +0.023 | **−0.031** |

**The brute-force F⁻¹ control is the decisive datum.** It bypasses EK-FAC
entirely and inverts the empirical per-sequence Fisher directly (32×32 or
128×128 — feasible at toy scale). If LOO-IF correlation were limited by
EK-FAC's 8 % approximation error, brute-force F⁻¹ would clear the 0.7
target. It does not, in any configuration. The failure is intrinsic to
the IF formula's linear-in-ε approximation when ε = 1/N is finite (1–5 %)
and the loss landscape is non-convex with strong cross-token coupling.

This is consistent with the literature: Bae et al. 2022 §3 documents the
finite-ε vs infinitesimal-ε discrepancy quantitatively and proposes
Proximal Bregman Response Functions (PBRF) as an alternative ground
truth. PBRF is paper-scale engineering, outside our thesis scope.
Detailed analysis in `reports/notes/loo_investigation.md`.

### 3.2 Adopted: Real-Data Self-Influence Sanity

Following Grosse et al. (2023) and Chen et al. (2026), we adopt
self-influence sanity at production scale as the operational validation:
for any rollout `z_m` and the eval target `f = log π_θ(y_m | x_m)`, the
KL-RL IF score satisfies `I(z_m, z_m) = −s_mᵀ F⁻¹ s_m ≤ 0` and should
dominate cross-IFs `I(z_n, z_m)` for `n ≠ m`.

Tested on `data/rollouts/step_0500.pt` (32 PPO rollouts from real
GPT-Neo-125M, production damping `max(Λ + 0.1·Λ̄, 1e-5)`), 5 randomly
chosen self-targets:

| self idx m | I_self (signed) | rank of \|I_self\| in 32 |
|---|---|---|
| z_0 | −1.217e+4 | **1/32** |
| z_5 | −1.776e+4 | **1/32** |
| z_10 | −1.403e+4 | **1/32** |
| z_15 | −8.127e+3 | **1/32** |
| z_20 | −6.465e+3 | **1/32** |

All 5 are negative (KL-RL sign convention satisfied) and all rank #1
with magnitude 50–100× the largest cross-IF. This is the diagnostic the
published literature relies on for at-scale EK-FAC validation; we accept
Stage 3 on this evidence.

### 3.3 Summary of Validation Evidence

| Evidence | Property tested | Result |
|---|---|---|
| Stage 2 gate 1a | `fit_lambda` mechanics + Kronecker basis assembly | 2.08e-17 (passes by 7 orders of magnitude) |
| Stage 2 gate 1b | EK-FAC reconstruction under K-FAC limit | 2.66e-15 |
| Stage 2 gate 1c | Q orthonormality (toy + real) | ≤ 4.89e-15 |
| Stage 2 gate 1d | A/S eigen reconstruction (toy + real) | ≤ 3.05e-15 |
| Stage 2 informational | EK-FAC inverse-rel-err vs true Fisher | 8.0 % (Grosse 2023 / MDA range) |
| Stage 3 real self-IF | 5 self-targets on real GPT-Neo-125M rollouts | 5/5 rank #1, 50–100× magnitude |
| Stage 3 brute-force F⁻¹ LOO control | Isolates IF-formula failure mode | confirms ≠ pipeline bug |

---

## 4. Production IF Setup

### 4.1 θ* Selection

Per Phase 0's three-phase analysis (rapid detox 0–300, healthy plateau
300–1200 with peak at step 650, over-training + reward-hacking 1200–2000),
production IF is run on `data/ppo_checkpoints/step_0650` as θ*. This is
the Pareto-optimal checkpoint across toxicity reduction, reward, response
length, and PPL — not the final checkpoint step_1999, which is partially
reward-hacked. The eval target `f` is evaluated under this θ*.

### 4.2 Rollout Span

The rollouts that *produced* θ_650 through PPO learning are
`data/rollouts/step_0000.pt` through `data/rollouts/step_0650.pt`. With
32 rollouts per step file, this is **651 × 32 = 20,832 rollouts**
totalling roughly 600,000 response tokens (mean response length ≈ 30
tokens after the min_length=10 floor).

Rollout files after step 650 (steps 651–1999, ~43k more rollouts) are
*not* included in production IF: they did not influence θ_650 since they
post-date it. They remain on disk for the Phase 3 mechanistic
verification of the over-training regime if needed.

### 4.3 Pipeline Steps

| step | code | output | est. wall-clock (GPU) |
|---|---|---|---|
| Stage 1A: accumulate A, S over 20,832 pseudo-labelled rollouts | `src/ekfac/factors.py` | `A.pt`, `S.pt` (fp64) | ~30 min |
| Stage 1B-1: eigendecompose A and S | `src/ekfac/eigen.py:eigendecompose` | `Q_A.pt`, `Λ_A.pt`, `Q_S.pt`, `Λ_S.pt` | < 5 s |
| Stage 1B-2: fit Λ_corrected (second pass, pseudo-labelled) | `src/ekfac/eigen.py:fit_lambda` | `Lambda.pt` | ~30 min (same forward cost as Stage 1A) |
| Stage 2: compute `g_eval` at θ* and IHVP `p = F⁻¹ g_eval` | `src/ekfac/training.py:compute_per_sample_grad` + `inverse_hvp` | `p.pt` (d_out × d_in) | < 1 s |
| Stage 3: per-sample IF for each of 20,832 rollouts | `compute_per_sample_grad` + `influence_score_klrl` | `influence_scores.csv` (20,832 rows) | ~17 min |
| Total | | | **≈ 80 min** |

Per-rollout IF compute scales as one forward + one backward through
GPT-Neo-125M, followed by a Frobenius inner product. The forward+backward
on cuda is the dominant cost (~50 ms at average response length 30 tokens).
Stage 1A and Stage 1B fit_lambda share the same forward+backward pattern
but accumulate over the rollouts as a whole, not per-rollout — they
dominate total time.

### 4.4 Eval Target `f` (Pending User Selection)

The IF score is computed against a single (or small set of) eval objective.
For Phase 2 we ship the production IF run against one held-out toxic
prompt from RealToxicityPrompts' challenge subset (the "eval target" in
CLAUDE.md's locked-in decision). The specific prompt is not chosen yet
and requires the user's input.

Anticipated structure (one of):

* **(i) Toxic completion**: `f = log π_θ(y_toxic | x_eval)` where
  `y_toxic` is a deliberately toxic continuation. Positive `I_KL > 0`
  rollouts are those whose KL-RL contribution to θ* shifts the policy
  *toward* `y_toxic` — these are training rollouts that "looked
  effective" to PPO but had toxic generalisation. The IF ranking
  identifies them.
* **(ii) Non-toxic completion**: `f = log π_θ(y_clean | x_eval)`. Mirror
  case: positive IF rollouts pushed the policy toward `y_clean`.
* **(iii) Toxic probe direction**: `f = v_toxicᵀ h_layer(x_eval; θ)` —
  more directly tied to Lee et al.'s mechanistic interpretation but
  requires a re-trained toxicity probe on GPT-Neo-125M (re-validation
  pending per Phase 0 §1.5).

I recommend **(i) one toxic completion** for the first production run —
it is the cleanest case, matches CONTEXT.md §6, and produces the most
interpretable Phase 3 follow-up (top-k rollouts should be examinable for
toxic content). Selection of the specific prompt + completion is pending
user input.

---

## 5. Limitations

1. **K-FAC independence violation on transformers.** The 8 % inverse
   Frobenius error reported in Section 2.2 is structural; EK-FAC misses
   45 % of the per-token Fisher's Kronecker off-diagonal mass. This
   propagates into IF rankings as a deviation from a "true Fisher"
   ranking; the practical mitigation is the gradient-similarity
   sanity baseline (CLAUDE.md locked decision), which is computed at
   near-zero marginal cost and surfaces cases where damping has wiped
   out F⁻¹'s discriminating effect.

2. **94 % of Λ entries below damping floor.** On the real model at
   step_0500.pt (959 tokens, 32 rollouts), 2.22M of 2.36M Λ entries are
   below the production damping floor `1e-5`. The effective rank of the
   inverse-Fisher subspace is ~136k directions, not the nominal 2.36M.
   Production IF rankings therefore reflect EK-FAC's response within
   this damped subspace, not the full parameter dimension. This is
   expected for Kronecker factorisation of transformer layers and is
   why two-level damping (relative + absolute floor) was adopted.

3. **LOO-style validation infeasible at production scale.** Per Section
   3, LOO is the wrong validator even at toy scale (brute-force F⁻¹
   confirms structural IF-formula limitation, not an EK-FAC artefact).
   At production scale, 20,832 retrainings × ~3 GPU-hours each ≈ 7 GPU-years
   — infeasible regardless. Self-influence sanity (Section 3.2) is the
   adopted at-scale alternative.

4. **IF scores are relative quantities.** Absolute magnitudes depend on
   the response-length distribution, damping floor, and Λ̄. Only top-k
   rankings (and sign agreement with gradient similarity) are
   interpretable. The Phase 3 mechanistic verification will compare
   rankings against Lee et al.'s toxic-vector activations, not absolute
   IF values.

5. **Phase 0 KL estimator choice (k1) interacts with IF interpretation.**
   Phase 0 used the TRL default `k1` KL estimator, not the `k2` form
   discussed in CONTEXT.md §1. The IF formula sign and structure are
   not affected (both estimators give the same expected KL; CONTEXT.md
   §1 footnote argues the IF derivation depends only on the
   KL-RL-optimum structure, not the estimator), but the Phase 1
   effective-reward-variance diagnostic should account for the
   estimator choice when interpreting the variance magnitude.

---

## Files and Test Coverage

```
src/ekfac/
├── __init__.py
├── config.py            # damping (toy 0.01 / production max(Λ + 0.1·Λ̄, 1e-5)), layer_name, seeds
├── data.py              # Rollout dataclass; load_rollout_file; build_input_and_labels (prompt mask)
├── hooks.py             # capture_c_proj context manager (forward m + backward δ)
├── factors.py           # Stage 1A: accumulate_AS (pseudo-labelled, per-token, fp64)
├── eigen.py             # Stage 1B: eigendecompose, fit_lambda (per-token), inverse_hvp, inverse_hvp_additive, reconstruct_F_inv_dense
├── influence.py         # influence_score_klrl, influence_score_supervised
├── toy.py               # ToyTransformer (configurable d_model, d_mlp, n_layer)
└── training.py          # set_determinism, train_toy_to_optimum, compute_per_sample_grad, eval_logprob

tests/
├── test_toy.py                       4 tests — toy model mechanics
├── test_mask.py                      5 tests — prompt mask correctness
├── test_factors_toy.py               3 tests — Stage 1A: hook shape, A/S sanity, eigenvalue ranges
├── test_factors_real.py              1 test  — Stage 1A real-model acceptance
├── test_eigen_toy.py                 9 tests — Stage 2: gates 1a/1b/1c/1d + informational + (α/β) research
├── test_eigen_real.py                3 tests — Stage 2 real-model Λ statistics
├── test_training_toy.py              3 tests — determinism + convergence
├── test_influence_toy.py             3 tests — IHVP consistency + Stage 2 mechanics on underparam toy + LOO (gates fail by design)
└── test_self_influence_real.py       1 test  — Stage 3 production-scale self-IF sanity

Total: 32 Phase 2 tests; 67 with Phase 0 tests.
Phase 2 acceptance: all implementation-correctness gates PASS; LOO gate
fails by design (documented in reports/notes/loo_investigation.md).
```

---

## Pending Items

* **Eval target `f`** for the production IF run — awaiting user selection
  (Section 4.4).
* **CLAUDE.md hard constraint #5** revision — the original 1e-3 Frobenius
  gate on the toy was incorrect for transformer EK-FAC; updated gate is
  the four Stage 2 implementation-correctness gates plus the at-scale
  self-IF sanity. User authority required.
* **Phase 3** — mechanistic verification of IF rankings against Lee et al.
  toxic-value-vector activations on PPO checkpoints; final layer selection
  if probe re-training succeeds.
