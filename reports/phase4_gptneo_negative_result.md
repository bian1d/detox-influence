# Phase 4 — GPT-Neo-125M Baseline: Negative-Result Memo

**Status: archived baseline (2026-06-02).** Self-contained record of the Phase 4
error-budget on the GPT-Neo-125M detox model. The verdict is a **kill**: the
clean RL-KL influence formula is not a controlled approximation at this model's
chosen checkpoint, and the failure is estimator-independent. This memo is the
permanent comparison point for any follow-up model (e.g. OLMo-2-1B-SFT) — it
stands on its own, independent of those results.

---

## 0. TL;DR

The thesis influence formula assumes the PPO policy sits at the KL-RL optimum
(assumption **A2**). Phase 1 already showed A2 fails at the response level
(ICC = 0.892). Phase 4 asked the consequential question — *does that failure
break the formula?* — and the answer for GPT-Neo-125M is **yes**:

- **E3 (central):** the curvature correction Δ that A2-failure leaves in the
  influence operator is **large** — 4–34× the scale of the evaluation gradient
  along the direction the formula actually uses — and to first order it reorders
  the per-rollout influence ranking (Spearman 0.20–0.82, top-50 overlap 0–22%).
- **E2:** the checkpoint is genuinely non-stationary, but only **moderately**
  (~8–12% of the detox KL distance) — a bounded, secondary error.
- **E5:** the within-prompt variance driving A2-failure is **not** a k1
  estimator artifact — switching to k3 leaves the ICC essentially unchanged
  (0.892 → 0.925). The kill cannot be rescued by changing the KL estimator.

The published `I_seq/I_toxic_*` numbers are *correctly computed*; what fails is
their interpretation as the KL-RL rollout influence once A2 does not hold.

---

## 1. Setup (everything needed to read this memo)

- **Model:** GPT-Neo-125M, detoxified with PPO (TRL 0.9.6), reward model
  `facebook/roberta-hate-speech-dynabench-r4-target`.
- **θ\* (checkpoint analysed):** `step_0650` — the Pareto-optimal checkpoint from
  Phase 0 (toxicity 0.016, reward 3.84, PPL 33), **not** the KL-RL argmax.
- **φ (parameter subspace):** layer-9 MLP value matrix `W₂` =
  `transformer.h.9.mlp.c_proj.weight`, shape (768, 3072) ≈ **2.36M params**.
- **β:** 0.23652004338891985 — the adaptive KL coefficient in force at step 650
  (bit-identical to the value Phase 1 used). KL estimator **k1** throughout.
- **Influence formula (what is being tested):**
  `I_f(z_m) = −∇_φ f(θ*)ᵀ F⁻¹ ∇_φ log π_θ*(y_m | x_m)`, with the Fisher `F`
  approximated by EK-FAC, per-token-mean reduction, prompt-masked, batch=1.
- **Eval targets f:** `f_seq` (sequence-level, last-token advantage; Hu et al.
  2025) + three hand-crafted `f_toxic^{C1/C2/C3}` (racial / orientation /
  misogyny templates).

**The assumption under test (A2):** `π_θ* = π*` (the closed-form KL-RL optimum).
If A2 holds, the effective reward `R̃ = r − β·log(π_θ*/π_ref)` is constant across
responses to a prompt, the Fisher-form and Hessian-form terms cancel, and
`∂G/∂θ|_θ* = −βF`, giving the clean formula by the implicit function theorem.
If A2 fails, `∂G/∂θ|_θ* = Δ − βF` with a residual curvature term
`Δ = E[A(x,y)·(s sᵀ + ∇²_φ log π)]`, where `A(x,y) = R̃ − mean_y R̃_x` is the
same-prompt advantage. The clean formula is the special case Δ = 0.

**Two independent error sources Phase 4 measures:**
1. **Stationarity** — θ\* is a Pareto checkpoint, so `G(θ*) = ∇_φ J ≠ 0` (the IFT
   wants G = 0). → **E2.**
2. **Curvature (central)** — even granting G = 0, `∂G/∂θ ≠ −βF` because Δ ≠ 0.
   → **E3.**

---

## 2. E1 — advantage distribution, variance decomposition, ICC stability

Reused Phase 1's 100 prompts × 32 on-policy rollouts at θ\*.

- **Advantage `A(x,y)`** is non-Gaussian with a pronounced **asymmetric left
  tail**: skew −0.97, excess kurtosis 1.95, min −8.82 / max +4.21 (negative
  reach 2.1× the positive); the most-negative 5% of responses carry **38%** of
  total ΣA². (Relevance: Δ = E[A·curvature], so this heavy tail is what *could*
  drive Δ — tested in E3.)
- **Within-prompt variance decomposition** (exact identity, reconstruction error
  1.8e-15): `Var_y[R̃] = Var_y[r] + β²Var_y[KL] − 2β·Cov_y[r,KL]`. Variance-
  weighted shares: **reward 0.47 / KL 0.56 / cov −0.03** — genuinely mixed,
  leaning slightly to the KL term. A length caveat applies: corr(len, KL) = +0.45
  and ρ²(len, R̃) = 0.13, so part of the KL share is the mechanical growth of
  summed k1-KL with response length, not per-token policy deviation.
- **ICC stability:** across 10 fresh-response reseeds, ICC = **0.870 ± 0.031**,
  range [0.828, 0.924]; the Phase 1 anchor (0.892) sits inside the band. ICC=0.892
  is a stable property, not a sampling fluke.

**E1 verdict:** non-decisive (mixed signal) — by design a hint, not the call.

---

## 3. E2 — first-order non-stationarity ‖G(θ\*)‖

Sampled 256 prompts × 8 fresh on-policy rollouts at θ\*. Estimated, in the φ
subspace, `‖G‖²` and the natural-gradient step `½·GᵀF⁻¹G` (= KL the policy would
move on one Newton step toward the φ-restricted stationary point), reusing the
cached EK-FAC `F⁻¹`. **Estimator:** all-pairs U-statistic on per-prompt mean
score-vectors (the exact-unbiased limit of split-half — high-dim norms must not
be estimated by the naive `‖Ĝ‖²`, which carries a `+tr(Cov)/N` self-bias),
jackknife + without-replacement subsampling SEs.

- **‖G(θ\*)‖ = 0.0158** (‖G‖² = 2.49e-4, z ≈ 3–4) — resolvably nonzero. θ\* is
  genuinely **not** a stationary point of J (expected for a Pareto checkpoint).
- **Natural-gradient step KL ≈ 0.73** — that is ~29× a single PPO mini-update
  (approxkl ≈ 0.025) but only **~8.3% of the cumulative policy-vs-ref KL (8.84)**
  and **~12% of the controller target (6.0)**.

Two methodological notes worth carrying forward: (a) the spec's primary `(R̃−β)`
baseline is a *nearly-degenerate* U-statistic (the large between-prompt R̃ level
multiplies the score-mean noise; both jackknife and subsampling SEs then agree
with each other yet both *underestimate* the true variance) — the
**LOO-advantage** baseline is the trustworthy one and is corroborated by an
independent sample-level split-half; (b) a resample-*with-replacement* bootstrap
is invalid for a degree-2 U-statistic (duplicate units reintroduce self-pairs).

**E2 verdict:** non-stationarity is **real but bounded (~8–12% of the detox KL
journey)**. It limits the *absolute / counterfactual* interpretation of the IF
but is a secondary error. **Scope:** `GᵀF⁻¹G` is the layer-9-W₂-subspace step — a
*lower bound* on the full-model non-stationarity; the comparison to full-policy
training KL is cross-scope and indicative, not a full-model figure.

---

## 4. E3 — the Δ curvature error-budget (CENTRAL) → KILL

**Cache-semantics gate (done first).** The production per-rollout cache
`g_scaled_z[m]` is `Q_Sᵀ s_m Q_A / denom`, **not** raw `s_m`. The published
`I_seq/I_toxic_*` are nonetheless exactly `−g_fᵀ F⁻¹ s_m` (verified to 3e-9), and
`compute_per_sample_grad` uses mean reduction + prompt mask + batch=1 (no length
bias). The cache is reusable for corrected scoring via **projection-only**
eval-side vectors (`Q_Sᵀ(·)Q_A` — the denom is already in the cache; dividing
again would double-apply the damping).

**Method.** Sampled 200 prompts × 16 on-policy rollouts at θ\*; computed the
matrix-free **Δ̃-vector product** `Δ̃ p_f = (1/β)·E[A·(s(sᵀp_f) + ∇²_φlogπ·p_f)]`
(`Δ̃ = Δ/β`; per-sample HVP via double-backward, eager attention) on each target's
`p_f = F⁻¹ g_f`. **Toy correctness gates — all green before the real model:**
HVP vs dense Hessian (1e-9); matrix-free Δ̃·v vs brute-force (1e-8); the
project-but-no-double-denom chain; MINRES on symmetric-indefinite + deterministic
CG breakdown (negative control).

**Thermometer `‖Δ̃p_f‖ / ‖g_f‖`** (the relative size of the Δ perturbation vs F
on the direction the formula uses; ≲0.1 ⇒ negligible, ≳0.3 ⇒ kill):

| target | thermometer | first-order Spearman(base, corrected) | Jaccard@50 | median \|corr\|/\|base\| |
|---|---|---|---|---|
| f_seq | **34.1** | 0.815 | 0.22 | 53× |
| f_toxic C1 | **12.5** | 0.597 | 0.05 | 41× |
| f_toxic C2 | **4.2** | 0.203 | 0.00 | 11× |
| f_toxic C3 | **11.8** | 0.559 | 0.02 | 31× |

**Every target fails decisively** — thermometers 4–34 (all ≫ 0.3; each ‖Δ̃p‖² is
z ≈ 4–17 above zero), and the first-order corrected ranking is far from the clean
ranking (Spearman 0.20–0.82, top-50 overlap 0–22%).

Why so large: `p = F⁻¹g_f` lives in F's *low-curvature* directions (where F⁻¹
amplifies), and Δ̃ does **not** vanish there — so the perturbation dominates F
exactly where the formula is most sensitive.

**Two honest caveats (do not change the verdict):**
1. **First-order Spearman values are illustrative, not authoritative.** At these
   thermometers the Neumann expansion `(F−Δ̃)⁻¹ ≈ F⁻¹ + F⁻¹Δ̃F⁻¹` is well outside
   its convergence radius (median `|corr|/|base|` 11–53×, p90 up to 190×). They
   establish only that the correction is enormous and reordering; the *exact*
   corrected ranking would need a full-order MINRES solve. **The kill follows
   from the magnitude criterion alone** (thermometer ≳0.3, met 14–114× over).
   MINRES was **not** run — at thermometer 34 the operator `(F−Δ̃)` is dominated by
   −Δ̃ and very likely will not converge, and Δ̃ itself is only moderately resolved
   (two-half cosines 0.13–0.79), so a MINRES solution would be untrustworthy.
2. **Δ is broad, not driven by the E1 left tail.** The bottom-5%-by-A samples
   carry only **15%** of the Δ-contribution mass (~3× uniform, far from A²'s 38%
   concentration). The large Δ is pervasive — it cannot be removed by cleaning a
   few outlier prompts.

**E3 verdict: KILL.** Dropping Δ — the clean `−g_fᵀ F⁻¹ s_m` — is not a controlled
approximation to the curvature-corrected influence at θ\*; the A2-failure
correction is large and reorders the rollout ranking.

---

## 5. E5 — is ICC=0.892 a k1 estimator artifact?

Reproduced Phase 1 stage2's exact responses (same seeds) and recomputed R̃ / A /
ICC under both k1 (signed; the paper's) and **k3** = `Σ[(ρ−1) − log ρ]`,
`ρ = π_ref/π_θ*` (unbiased, non-negative). Diagnostic only (no retrain; F does
not depend on the KL estimator).

| | k1 | k3 |
|---|---|---|
| ICC (within/total) | **0.8917** (= Phase 1, exact) | **0.9250** |
| mean within-prompt Var[R̃] | 2.39 | 40.27 |
| responses with KL < 0 | 7.2% | 0 |

**ICC does not drop under k3 — it slightly rises.** The hypothesis that k1's
sign-flips (7.2% negative-KL responses, where k1 turns a penalty into a reward)
inflate the within-prompt variance is rejected: removing that artifact leaves the
within/total ratio essentially unchanged. The within-prompt R̃ heterogeneity is a
real structural feature (consistent with E1: the reward term alone is ~47% of it,
which is estimator-independent).

**E5 verdict:** the kill is **estimator-independent** — k3 cannot rescue it.

---

## 6. Overall verdict and scope

**The clean RL-KL influence formula, applied to GPT-Neo-125M at step_0650 in the
layer-9 W₂ subspace, does not survive the A2-failure it is built to ignore.** Of
the two error sources, stationarity (E2) is bounded (~8–12%), but the central
curvature correction Δ (E3) is large (thermometers 4–34) and reorders the
ranking; E5 shows this is not an artifact of the k1 estimator.

**Scope / what is NOT claimed:**
- The published influence *values* are correctly computed (Q3) — the failure is
  of *interpretation*, not arithmetic.
- All measurements are in the **layer-9 W₂ subspace**; other layers (Grosse-style
  layer-summed IF) are untested here (Phase 4 E6, deferred).
- The exact corrected ranking is not pinned (first-order non-convergent; MINRES
  not run). The kill rests on the magnitude criterion, which is unambiguous.

---

## 7. What this baseline is for

This result poses a question the next model is meant to answer: **is A2-failure
specific to a sub-optimal PPO run, or a general failure mode of the influence
method?** GPT-Neo-125M is a raw-pretrain base detoxified by PPO, so it may simply
sit far from any KL-RL optimum. A genuinely SFT-aligned model (e.g.
OLMo-2-1B-SFT) tests the alternative:

- **Δ small on the aligned model** → A2-failure is PPO-suboptimality-specific; the
  method works when the policy is near-optimal. GPT-Neo is then the cautionary
  baseline.
- **Δ still large** → it is a general death of the method in this RLHF setting; a
  *universal* negative result, with GPT-Neo as the first of two confirming cases.

Either outcome is publishable, and this memo is the fixed GPT-Neo comparison point
for both.

---

## 8. Reproducibility

- **Code:** `src/phase4/{config,e1_advantage,e1_icc_ci,e1_plots,models,
  grad_phi,sampling,diagnostics,e2_stationarity,hvp_logpi,e3_delta}.py`;
  drivers `src/run_phase4_e{1,2,3,5}.py`; cache check
  `tests/diag_e3_gscaled_check.py`. Tests `tests/test_phase4_{variance_decomp,
  e2,e3}.py` (22, all green). EK-FAC `F⁻¹` reused unchanged from Phase 2
  (`src/ekfac/`).
- **Data:** `data/phase4/e1_advantage_stats.json`, `e1_icc_resample.json`,
  `e2_stationarity.json`, `e3_delta_budget.json`, `e3_onpolicy_pool.jsonl`,
  `e5_estimator.json`; figures `reports/phase4/e1_*.png`. Full per-experiment
  detail in `reports/phase4_error_budget.md`.
- **Seeds:** on-policy main seed 42; split/subsample seeds 1000+; ICC reseeds
  2000+. β = 0.23652004338891985, k1, θ\* = step_0650 throughout.
- **Git:** commits `09af662` (E1), `1bfb041` (E2), `b7c082c` (E3), `2119fed` (E5).
