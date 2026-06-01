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
