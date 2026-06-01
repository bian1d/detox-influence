# Phase 4 — Theoretical Error-Budget & Robustness of the RL-KL Influence Function

Snapshot under analysis: **θ\* = step_0650**, frozen, with the step-650 adaptive
KL coefficient **β = 0.23652004338891985** (bit-identical to the value Phase 1
used to obtain ICC = 0.892). KL estimator **k1** throughout, consistent with
Phases 0/1/2. Parameter subspace φ = layer-9 MLP `W_2`
(`transformer.h.9.mlp.c_proj`, ≈2.36 M params).

Execution order E1 → E2 → E3, each with a stop-and-report gate. This file
accumulates one section per experiment.

---

## E1 — Advantage A(x,y) distribution + within-prompt variance decomposition + ICC CI

**What was done.** Two parts, both reusing the already-paid Phase 1 stage2
sampling (100 prompts × 32 on-policy rollouts at θ\*):

1. **Advantage / decomposition (no model).** Computed the same-prompt-centred
   advantage `A(x,y) = R̃(x,y) − mean_y R̃(x,·)` pooled over all 3200 samples;
   characterised its moments and tails; and decomposed each prompt's within
   variance via the exact identity
   `Var_y[R̃] = Var_y[r] + β²·Var_y[KL] − 2β·Cov_y[r,KL]`.
2. **ICC estimation band (models).** Redrew 32 fresh on-policy responses for a
   20-prompt subset, 10 independent reseeds, recomputing the subset ICC with
   Phase 1's identical estimator each time, to bound the run-to-run sampling
   error of the ICC = 0.892 statistic. (The original "fix-response noise floor"
   was dropped as conceptually empty: with (x, y, θ\*) fixed, R̃ is deterministic.)

**Output files.**
- `data/phase4/e1_advantage_stats.json` (75 KB) — moments, tail concentration,
  Q-Q grid, per-prompt decomposition, length lens.
- `data/phase4/e1_icc_resample.json` (3 KB) — subset ids, per-reseed ICCs, CI.
- `reports/phase4/e1_advantage_hist.png`, `e1_advantage_qq.png`,
  `e1_variance_decomposition.png`.

**Acceptance criteria & measured values.**

| Quantity | Value |
|---|---|
| Decomposition reconstruction max abs err | **1.78e-15** (machine precision) |
| Advantage A: mean | −1.5e-16 (≈0 by construction ✓) |
| A: std | 1.523 |
| A: skewness | **−0.974** (heavy left tail) |
| A: excess kurtosis | **1.945** (leptokurtic) |
| A: min / max | **−8.82 / +4.21** |
| Tail asymmetry (neg/pos reach ratio) | **2.10** |
| Lower-5% share of Σ A² | **0.377** vs upper-5% **0.141** |
| Weighted share: reward `Var[r]` | **0.468** |
| Weighted share: KL `β²·Var[KL]` | **0.561** |
| Weighted share: cov `−2β·Cov[r,KL]` | **−0.029** |
| Per-prompt median share: reward / KL / cov | 0.387 / 0.621 / −0.007 |
| Length lens: corr(len, r) / corr(len, KL) / corr(len, R̃) | −0.029 / **+0.454** / −0.355 |
| Length lens: ρ²(len, R̃) | **0.126** (≈ spec's 0.11) |
| ICC full-100 (Phase 1 anchor) | 0.8917 |
| ICC subset, Phase 1 data | 0.8992 |
| ICC resample: mean ± std | **0.8698 ± 0.0306** |
| ICC resample: range | [0.8281, 0.9240] |
| ICC resample: 95% CI of mean | [0.8523, 0.8872] |

**Reading (kept separate from the numbers).**

- *Advantage is non-Gaussian with a pronounced asymmetric left tail.* Skew
  −0.97, excess kurtosis 1.95, and a most-negative reach (−8.8) ≈2.1× the
  most-positive (+4.2). The lower 5 % of responses carry 38 % of total ΣA²
  versus 14 % for the upper 5 %. **Relevance to E3:** Δ = E[A·curvature], so the
  small set of large-|A| (mostly very-low-R̃, i.e. occasionally much more toxic /
  high-KL) responses is exactly what can drive Δ and reorder the IF ranking.
  E1 flags this concentration; E3 measures whether it actually moves rankings.

- *The within-prompt variance is split, leaning slightly to the KL term.*
  Weighted: reward 0.47, KL 0.56, cov −0.03. Per-prompt median leans more to KL
  (0.62 vs 0.39). This is **not** the clean "reward-surface-dominated" picture
  that would be a transition hint, nor a clean KL-dominated kill picture — it is
  genuinely mixed. Caveat from the length lens: corr(len, KL) = +0.45 and
  ρ²(len, R̃) = 0.13, so a meaningful slice of the KL term is the mechanical
  growth of *summed* k1-KL with response length (sampling draws 10–30 tokens),
  not per-token policy deviation from the optimum. So the effective "genuine
  KL-deviation" share is somewhat below 0.56.

- *ICC = 0.892 is a stable property, not a sampling artifact.* Across 10 response
  redraws the subset ICC is 0.870 ± 0.031, range [0.828, 0.924]; the Phase 1
  anchor (subset 0.899, full-100 0.892) sits inside this band. A2 failure at the
  y-level is real and reproducible; its estimation wobble is ≈±0.03.

**Transition / kill verdict for E1:** *non-decisive, mixed-leaning-KL* — exactly
as the spec scopes E1 ("线索，但不是决定性；决定性看 E3"). E1 does not by itself move
the decision either way; it (a) confirms A is real, stable, and heavy-/asym-tailed,
and (b) localises the would-be danger to a small large-|A| tail, which E3 must test.

**Deviations from plan.** None of substance. As agreed (Q-supplement), the
"fix-response R̃ noise floor" was replaced by the ICC estimation band. OOD prompts
deferred (Q1). β confirmed bit-identical to Phase 1 (supplement 2).

**Skipped / not finished.** Nothing in E1 scope. E2/E3 sections follow after review.

**Seeds.** Advantage/decomposition: deterministic (reuses Phase 1 stage2,
seed 42). ICC resample: reseed s, prompt pos p → seed 2000 + 100000·s + p;
bootstrap CI seed 1000.

---

## E2 — ‖G(θ\*)‖: first-order non-stationarity + natural-gradient step

**What was done.** Sampled **256 prompts × 8 fresh on-policy rollouts** at θ\*
(2048 samples, `y ~ π_θ*`, not stored training rollouts; seed 42). Computed each
per-sample φ-score `s = ∇_φ log π_θ*(y|x)` (batch=1, prompt-masked, per-token
mean) and the scalar `(R̃−β)`. Then estimated, in the 2.36M-dim φ subspace:
- **‖G‖²** where `G = ∇_φ J = E[s·(R̃−β)] = E_x[Cov_{y|x}(s, R̃)]`, and
- **GᵀF⁻¹G** (the squared natural-gradient norm; `½·GᵀF⁻¹G` = KL of one unit
  Newton step toward the φ-restricted stationary point), reusing the cached
  EK-FAC `F⁻¹` (`inverse_hvp`, production two-level damping).

**Estimator (CLAUDE.md Hard Constraint 8).** Naive `‖Ĝ‖²` carries a
`+tr(Cov)/N` self-product bias that swamps the signal in 2.36M dims. We use the
**all-pairs U-statistic** (the limit of averaging over all disjoint-half splits)
on per-prompt mean score-vectors, with a **jackknife-over-prompts** SE and a
**without-replacement subsampling** SE cross-check. (A resample-*with*-replacement
bootstrap was implemented, found **invalid** for a degree-2 U-statistic —
duplicated prompts create p==q self-pairs that bias it back toward the naive
plug-in — and replaced by subsampling. Noted as a methodological finding.)

**Two baselines for the weight, both unbiased for the same G:**
- `primary = (R̃−β)` (exactly per spec);
- `loo_advantage = R̃_i − mean_{j≠i, same prompt} R̃_j` (removes the large
  between-prompt R̃ level; far lower variance).

**Output files.** `data/phase4/e2_stationarity.json` (2.5 KB),
`data/phase4/e2_onpolicy_pool.jsonl` (669 KB, token ids + scalars).

**Acceptance criteria & measured values.**

| Quantity | primary `(R̃−β)` | **loo_advantage (trusted)** |
|---|---|---|
| ‖G‖² (U-stat) | 2.098e-2 | **2.489e-4** |
| ‖G‖² jackknife SE / subsample SE | 2.55e-3 / 1.88e-3 | 7.74e-5 / 5.63e-5 |
| ‖G‖² z (jack / subs) | 8.2 / 11.2 | 3.2 / 4.4 |
| ‖G‖² naive (biased) | 2.196e-2 | 6.104e-4 |
| **‖G‖** | 0.1448 | **0.0158** |
| GᵀF⁻¹G (U-stat) | 47.19 | **1.459** |
| GᵀF⁻¹G jack SE / subs SE | 7.32 / 5.32 | 0.647 / 0.476 |
| GᵀF⁻¹G z (jack / subs) | 6.4 / 8.9 | 2.3 / 3.1 |
| **natural-grad step KL** `½GᵀF⁻¹G` | 23.6 | **0.730** |
| Training comparators | — | per-step approxkl 0.025; policy-vs-ref k1 KL 8.84; target 6.0 |

**Which baseline to trust (and why the two disagree 84×).** Both are unbiased
for the same ‖G‖², yet the primary's point sits 84× above LOO with
non-overlapping SEs. The primary's per-prompt vector is `ḡ_p^{LOO} + (R̄_x−β)·s̄_p`;
the extra term is a large (~1.9× the score-mean noise), mean-zero, between-prompt
noise. It makes the primary U-statistic **nearly degenerate** (kernel variance
ζ₂ ≫ conditional variance ζ₁), so *both* jackknife and subsampling SEs — which
estimate the `(4/N)ζ₁` part — agree with each other yet **both underestimate**
the true variance (they miss the `(2/N²)ζ₂` degenerate term). Tell-tale: the
all-pairs debiasing removed only ~4% for the primary (2.196e-2 → 2.098e-2) but
59% for LOO (6.10e-4 → 2.49e-4). The **LOO estimate is trusted**: well-behaved,
and independently corroborated by a sample-level split-half (2.44e-4 ≈ 2.49e-4).
The primary is reported only as a cautionary diagnostic and is **not** used for
the conclusion.

**Reading (separated from numbers).**

- *θ\* is genuinely NOT a stationary point of J in the φ-subspace.* ‖G‖ ≈ 0.0158
  is resolvably nonzero (z ≈ 3–4 by two SE methods). This is expected — θ\* =
  step_0650 is a Pareto checkpoint, not argmax J — and confirms the IFT premise
  `G=0` does not hold exactly.

- *The implied one-step correction is large vs a single PPO step but a modest
  fraction of the detox journey.* The (damped, regularized — same F⁻¹ the IF
  uses) one-Newton-step KL is ≈ 0.73. That is ~29× a single PPO mini-update
  (approxkl ≈ 0.025) yet only ~8.3% of the cumulative policy-vs-ref KL (8.84) and
  ~12% of the controller target (6.0). The spec's two comparators thus point
  opposite ways: against a single training step the residual is large; against
  the journey/target it is moderate. A full Newton step is meant to cover the
  *whole* remaining distance, so the journey/target comparison is the more
  meaningful one — by it, θ\* sits within ~10% (KL) of the φ-restricted
  stationary point relative to how far it travelled from the base model.

- *Scope caveat.* GᵀF⁻¹G is the **layer-9 W₂ subspace** natural-grad step — a
  lower bound on the full-model one; the comparison to full-policy training KL is
  cross-scope and indicative, not exact. Per the KL-length confirmation, note
  the policy-vs-ref KL (8.84) itself grows partly with response length; this
  comparison inherits that.

**Transition / kill verdict for E2:** *non-stationarity is real but bounded.*
θ\* is not stationary (`G≠0`, robustly), so the **counterfactual / absolute-value**
interpretation of the IF carries a non-negligible stationarity error. But the
magnitude is moderate (~8–12% of the detox KL distance), and — per the spec —
E2 non-stationarity does **not** by itself kill the project: it limits absolute
IF values while leaving the **ranking** question to E3 (the central experiment).
E2 leans "bounded error, ranking-TBD", consistent with E1's non-decisive result.

**Deviations from plan.** (1) Headline estimator upgraded from plain split-half
(spec/Hard Constraint 8) to the **all-pairs U-statistic** — its exact-unbiased
limit — with jackknife + subsampling SEs; the plain sample-level split-half was
used as a corroborating cross-check (LOO 2.44e-4). (2) The with-replacement
bootstrap was found invalid for U-statistics and dropped (documented above).
(3) Added the LOO-advantage baseline alongside the spec's `(R̃−β)`; it proved
necessary, as the primary is nearly-degenerate and unreliable.

**Skipped / not finished.** Nothing in E2 scope. The φ-subspace-vs-full-model
scope gap is acknowledged, not closed (out of E2 scope).

**Seeds.** On-policy pool: prompt-selection seed 42, generation seed 42+i per
prompt. Jackknife: deterministic. Subsampling SE: seed 1000.
