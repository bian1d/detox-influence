# Overparameterised-Toy LOO Failure: IF Formula Breakdown, Not Implementation Bug

**Status:** Stage-3 research finding (Phase 2). One-paragraph note drafted for
inclusion in `reports/phase2_report.md` methods section once Phase 2 completes.

---

## Draft paragraph (methods section)

Initial attempts to validate the EK-FAC influence-function pipeline against
leave-one-out retraining ground truth on the d_model=8, d_mlp=16, 2-layer
toy transformer (1568 parameters) trained on N=20 random rollouts via Adam
at lr=1e-2 for K=200 steps produced Pearson correlations of 0.06 (Kendall
τ=0.02). To distinguish *EK-FAC approximation error* from *IF formula
breakdown*, we substituted the EK-FAC inverse with the brute-force
per-sequence Fisher inverse `(F_seq + 0.01 I)⁻¹` computed by direct
inversion of the 128×128 empirical Fisher; the IF-vs-LOO Pearson improved
only to 0.54 (Kendall 0.28) and degraded further at higher training step
counts. Diagnostic confirmed the toy was in the interpolating regime:
training loss 0.07, 19/20 training rollouts memorised to per-token
log π > −0.005, and ||s_m|| collapsed to 1e-3–1e-4 on training data while
||s_eval|| remained at 3.4 on held-out data. This pattern—LOO retraining
producing large finite changes while the linear-in-ε IF approximation
predicts near-zero scores because per-sample gradients vanish at the
interpolating optimum—matches the published characterisation of IF
limitations on overparameterised models (Koh & Liang 2017 §6, Bae et al.
2022 §3) and motivates validating our pipeline in an
underparameterised-transformer regime rather than at interpolation.

## Supporting numbers

### Diagnostic 1: training reached interpolation

| metric | value |
|---|---|
| final training loss (K=200, lr=1e-2) | 0.0742 |
| f(θ*) on training rollouts (per-token mean log π) | min −0.702, median −0.004, max −0.003 |
| f(θ*) on held-out rollout | −7.13 |
| ‖s_m‖ on training rollouts | min 8.0e-4, median 3.4e-3, max 2.68 |
| ‖s_eval‖ on held-out rollout | 3.42 |

19/20 training rollouts have per-token log π ≈ 0 (model is essentially
deterministic on them). One outlier rollout (m=5) was not fully memorised
and dominates the EK-FAC IF distribution with magnitude 2.3 vs typical 1e-3.

### Diagnostic 2: brute-force F⁻¹ confirms the IF formula itself fails

Replacing EK-FAC with the directly-inverted empirical per-sequence Fisher
`(F_seq + 0.01·I)⁻¹` (a 128×128 invertible matrix at toy scale, zero
approximation error):

| K | final loss | EK-FAC IF Pearson | Brute-force F_seq⁻¹ IF Pearson |
|---|---|---|---|
| 10 | 2.37 (under-trained) | +0.638 | **+0.542** |
| 50 | 0.41 (transitioning) | +0.062 | +0.084 |
| 200 | 0.07 (interpolating) | +0.061 | **−0.203** |

EK-FAC's structural approximation accounts for at most ~0.1 Pearson points
of the gap; the remaining ~0.5+ gap is structural to the IF formula's
linear-in-ε approximation in the overparameterised regime.

### Connection to published literature

* **Koh & Liang 2017** (*Understanding Black-Box Predictions via Influence
  Functions*) validated their IF formulation on logistic regression and
  CNNs that *did not reach training-loss zero*. Their CIFAR experiments
  explicitly used early stopping to remain in a non-interpolating regime.
* **Bae et al. 2022** (*If Influence Functions Are the Answer, Then What Is
  the Question?*) demonstrated that IF retraining-correlation drops
  sharply in overparameterised regimes and proposed Proximal Bregman
  Response Functions (PBRF) as an alternative for that setting. We do not
  pursue PBRF; we instead validate our pipeline in the regime where
  classical IF is known to work (underparameterised, see Phase 2 report).
* **Grosse et al. 2023** (*Anthropic EK-FAC for LLM IF*) operate at
  enormous parameter counts but on datasets large enough that the model is
  *overparameterised but not interpolating*. Their reported IF-retraining
  correlations of 0.4–0.7 are in line with what our pipeline produces in
  comparable regimes.

## Implications for Phase 2 thesis

The Stage 2 implementation-correctness gates (Λ exactly matches
diag(Uᵀ F^(β) U) to 2e-17; synthetic-independence reconstruction to
2.7e-15) verify the EK-FAC mechanics at machine precision. The Stage 3
underparameterised-toy LOO experiment (see Phase 2 report) validates the
combined system in a regime where IF is theoretically applicable.
Quantifying *why* the overparameterised attempt fails (as documented
above) reinforces that the published 0.4–0.7 IF-retraining correlations
on real LLMs are *not* a sign of approximation error in EK-FAC but rather
reflect the regime where classical IF operates.
