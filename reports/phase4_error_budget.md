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

---

## E3 — Δ error-budget: does the curvature correction reorder the IF ranking? (CENTRAL)

**Q3 cache gate (done first, per instruction).** The production cache
`g_scaled_z[m]` is **not** raw `s_m`: it is `Q_Sᵀ s_m Q_A / denom_T`
(eigenbasis-projected + denom-divided). The literal check `−⟨g_scaled_z[0],
p_seq⟩ = −7416` ≠ `I_seq[0] = −3.2008` (different bases). But the production
numbers are **correct**: `−⟨Q_Sᵀ g_seq Q_A, g_scaled_z[0]⟩ = −3.2008160608`
reproduces `I_seq[0] = −3.2008160639` to 3e-9, i.e. `I_seq = −g_seqᵀ F⁻¹ s_m`.
`compute_per_sample_grad` uses mean reduction + prompt mask + batch=1 (no length
bias). The cache is reusable for E3: both the baseline and the Δ-correction are
inner products against `g_scaled_z` with **projection-only** eval-side vectors
(`Q_Sᵀ(·)Q_A`, NO second denom divide — gated by `test_no_double_denom`).
Diagnostic: `tests/diag_e3_gscaled_check.py`.

**What was done.** Sampled **200 prompts × 16 on-policy rollouts** at θ\* (3200
samples, seed 42, in-distribution; OOD deferred per Q1), computed each sample's
advantage `A = R̃ − mean_y R̃_x`. For each of the four eval targets `f ∈
{f_seq, f_toxic^{C1}, C2, C3}`, formed the matrix-free **Δ̃-vector product**
`Δ̃ p_f = (1/β)·E[A·(s(sᵀp_f) + ∇²_φlogπ·p_f)]` (per-sample HVP via double
backward, eager attention) on `p_f = F⁻¹g_f` (cached). Reported the thermometer
`‖Δ̃p_f‖/‖g_f‖` (U-statistic-debiased `‖Δ̃p_f‖²` over per-prompt vectors,
jackknife SE, two-half cosine), then the first-order corrected ranking
`I_corr = −pᵀs_m − qᵀs_m` (`q = F⁻¹Δ̃p_f`) over all 20,832 rollouts via the cache.

**Toy gates (all green before the real model; `tests/test_phase4_e3.py`).**
HVP vs dense Hessian (1e-9); matrix-free Δ̃·v vs brute-force dense Δ̃ (1e-8);
per-prompt == flat total; **project-but-no-double-denom** chain == brute
`−g_fᵀF⁻¹s_m` and double-divide gives a *different* answer; MINRES converges on
symmetric-indefinite, CG breaks down (deterministic negative control).

**Output files.** `data/phase4/e3_delta_budget.json`,
`data/phase4/e3_onpolicy_pool.jsonl` (3200 rows: A, R̃, response text).

**Acceptance criteria & measured values.**

| target | thermometer `‖Δ̃p‖/‖g_f‖` | `‖Δ̃p‖²` U-stat ± jackSE | two-half cos | Spearman (1st-order) | Jaccard@50 | median \|corr\|/\|base\| |
|---|---|---|---|---|---|---|
| f_seq | **34.1** | 3.67 ± 0.92 | 0.79 | 0.815 | 0.22 | 53× |
| f_toxic C1 | **12.5** | 62.7 ± 9.3 | 0.46 | 0.597 | 0.05 | 41× |
| f_toxic C2 | **4.2** | 20.9 ± 1.2 | 0.13 | 0.203 | 0.00 | 11× |
| f_toxic C3 | **11.8** | 67.1 ± 10.5 | 0.51 | 0.559 | 0.02 | 31× |

(`‖g_f‖` = 0.056 / 0.472 / 0.521 / 0.565. Baseline reproduces cached `I_*` to
≤2e-6 — corrected-scoring path verified.)

**Decision thresholds:** transition needs all Spearman>0.95 & thermometer≲0.1;
**kill** needs any Spearman<0.9 **or** thermometer≳0.3. **Every target fails on
both counts**, by a wide margin: thermometers 4–34 (≫0.3), first-order Spearman
0.20–0.82 (<0.9).

### Verdict: KILL.

At θ\* = step_0650, in the layer-9 W₂ subspace, with k1, the curvature
correction Δ that A2-failure leaves in `∂G/∂θ = Δ − βF` is **not negligible** —
it is 4–34× the scale of `g_f` along the very direction the influence function
uses (`p = F⁻¹g_f`, which lives in F's low-curvature directions, exactly where
Δ̃ is not small). Dropping Δ — i.e. the paper's clean formula `−g_fᵀF⁻¹s_m` — is
therefore **not a controlled approximation** to the curvature-corrected
influence `−g_fᵀ(F−Δ̃)⁻¹s_m`. The published `I_seq/I_C1/C2/C3` are still
*correctly computed* `−g_fᵀF⁻¹s_m` (Q3); what fails is their interpretation as
the KL-RL rollout influence once A2 does not hold.

### Honest caveats (kept separate from the verdict)

1. **The first-order Spearman values are illustrative, not authoritative.** With
   thermometers ≫0.3 the Neumann expansion `(F−Δ̃)⁻¹ ≈ F⁻¹ + F⁻¹Δ̃F⁻¹` is far
   outside its convergence radius — confirmed by `|corr|/|base|` medians of
   11–53× (p90 78–190×): the "correction" dwarfs the baseline. So 0.20–0.82 only
   establishes that the correction is enormous and reordering; the *exact*
   corrected ranking is unresolved without a **full-order MINRES** solve of
   `(F−Δ̃)p_corr = g_f`. The **kill itself does not depend on this** — it follows
   directly from the spec's magnitude criterion (thermometer ≳0.3), met 14–114×
   over on all four targets.

2. **Δ is broadly distributed, NOT driven by the E1 left tail.** The bottom-5%-by-A
   samples carry **15.2%** of the Δ-contribution mass (vs 5% uniform — elevated
   ~3×, but far from the 38% concentration A² showed in E1); top-5%-by-|A| carry
   16.8%. So the large Δ is **not** attributable to a few removable outlier
   prompts — one cannot "clean" the heavy tail and rescue the formula. (The most
   extreme-A samples are partly degenerate generations, e.g. `A=−9.30` →
   repetitive "Myth of1000076…"; listed in the JSON, but they are a minority of Δ.)

3. **Convergence.** Two-half cosines 0.13–0.79 show the Δ̃p *direction* is only
   moderately converged at 200×16 (C2 worst, 0.13). The thermometer *magnitudes*
   are robust (each `‖Δ̃p‖²` is z≈4–17 above zero); more samples would sharpen the
   exact ratios but cannot bring any near 0.3.

4. **Scope (standing caveat).** This is the layer-9 W₂ subspace, θ\*=step_0650,
   k1. Whether other layers (E6) or the k3 estimator (E5) change the picture is
   untested and deferred.

### §4 decision-logic synthesis (E1+E2+E3)

| error source | by | reading |
|---|---|---|
| within-var composition (E1) | E1 | non-decisive (reward 0.47 / KL 0.56, mixed) |
| stationarity ‖G(θ\*)‖ (E2) | E2 | real but bounded (~8–12% of detox KL) |
| **curvature Δ (E3, central)** | **E3** | **KILL — Δ large (4–34× g_f), ranking not robust** |

The central experiment lands on kill: **ICC=0.892 does damage the formula.** The
clean `−∇f F⁻¹ ∇logπ` ranking is *not* a first-order-precise local attribution at
θ\* — the A2-failure curvature correction is large and reorders it. Per the
project's stop-criterion framing, this is the "锤死" outcome.

**E4 NOT run** (gated on transition; skipped after kill confirmed). **MINRES
full-order not run** — flagged as the only remaining step that could refine *how*
the ranking changes, but it is the expensive/possibly-non-convergent step the
plan reserved for human sign-off, and the kill verdict does not require it.

**Seeds.** Pool: prompt seed 42, generation seed 42+i. Toy gates: fixed seeds.

---

## E5 — k1 vs k3 KL estimator: is ICC=0.892 a k1 artifact?

**What was done.** Reproduced Phase 1 stage2's exact responses (100 prompts ×
32, same seeds) and recomputed effective reward / advantage / ICC under **both**
the k1 estimator (signed; the paper's) and **k3** = `Σ[(ρ−1) − log ρ]`,
`ρ=π_ref/π_θ*` (unbiased, non-negative). Diagnostic only — no retrain; F is a
pure score outer product and does not depend on the KL estimator, so only R̃/A/ICC
change. Output: `data/phase4/e5_estimator.json`.

**Measured values.**

| quantity | k1 | k3 |
|---|---|---|
| **ICC (within/total)** | **0.8917** (= Phase 1 ref, exact) | **0.9250** |
| mean within-prompt Var[R̃] | 2.39 | 40.27 (16.8×) |
| KL mean / std | 6.82 / 5.11 | 16.31 / 27.62 |
| fraction of responses with KL < 0 | 0.072 | 0 (k3 ≥ 0 by construction) |

**Reading.** The hypothesis was that k1's sign-flips (7.2% of responses have
negative k1 KL — on suppressed-toxic tokens k1 turns into a reward) might inflate
the within-prompt variance, so that k3 would show a markedly *lower* ICC and part
of "A2 failure" would be a k1 artifact (a transition-leaning lead). **The result
rejects that hypothesis: ICC does not drop — it slightly rises (0.892 → 0.925).**
k3's within-prompt variance is in fact 16.8× larger in absolute terms (k3's
`(ρ−1)` explodes on rare large-ratio tokens), but the between-prompt variance
scales with it, so the within/total *ratio* is stable across two very different
estimators. The within-prompt R̃ heterogeneity is therefore a **real structural
feature, not a k1 estimator artifact** (consistent with E1: the reward term alone
contributes ~47% of within-variance, and that is estimator-independent).

**Implication for the kill.** **The kill is estimator-independent — switching
k1→k3 on this model does not rescue it.** A2 fails at the y-level under both
estimators; E3's Δ is built from F (estimator-independent) weighted by A
(within/total ratio stable), so a k3 redo would not shrink Δ. The lever that
could change the verdict is a *different model* (e.g. the deferred OLMo-2-1B-SFT,
genuinely SFT-aligned, where A2 might hold better) — not a different KL estimator.

**Seeds.** Reproduces Phase 1 stage2: prompt seed 42, per-prompt generation seed
42+key.
