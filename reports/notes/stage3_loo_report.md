# Stage 3 toy LOO acceptance report (underparameterised regime)

**Configuration:** d_model=4, d_mlp=8, n_layer=1, n_heads=1, vocab=20, max_len=4 — 356 parameters; N=100 rollouts; K=200 AdamW steps; lr=0.01; weight_decay=0.01.

Determinism check: PASS (verified in tests/test_training_toy.py)

## Training regime sanity
- Initial loss: 3.1793
- Step 25 loss: 2.8732
- Final loss (step 200): **1.8652**
- f(θ*) on held-out eval: -6.0312
- Gate (final loss > 0.5, not interpolating): PASS

## Stage 2 mechanics on this toy (rerun)
- 1c Q orthonormality, 1d A/S reconstruction, Λ > 0: see `test_stage2_mechanics_still_pass_on_underparam_toy` (all gates pass at machine precision).
- IHVP-vs-dense consistency: see `test_inverse_hvp_additive_matches_dense_reconstruction` (rel err < 1e-10).

## LOO acceptance (eval = held-out rollout 101)
- Pearson(I_ekfac, I_true):  **+0.0858**
- Kendall τ:                  **-0.0218**
- Sign agreement:             **48/100**

### Brute-force F⁻¹ control (zero EK-FAC approximation error)
- Pearson(I_brute_force, I_true): **-0.0309**
- Kendall τ:                       **-0.0913**
- Sign agreement:                  **48/100**
- *Interpretation: an upper bound on what any inverse-Fisher approximation can achieve in this regime; gap between brute-force and EK-FAC isolates structural approximation error.*

Per-rollout sample (first 20 of 100):
```
    m   I_ekfac        I_true
    m=  0 +3.985e-02    -1.012e+00
    m=  1 -2.455e+00    -7.969e-01
    m=  2 +7.856e-01    -2.690e-01
    m=  3 -9.091e-02    +6.152e-02
    m=  4 -2.763e-01    +9.556e-02
    m=  5 +4.684e-03    -9.316e-01
    m=  6 +5.065e-01    -9.380e-01
    m=  7 +1.111e+00    -1.131e+00
    m=  8 +1.785e-01    -8.932e-01
    m=  9 +6.460e-01    -2.032e+00
    m= 10 -6.900e-02    +3.593e-01
    m= 11 -6.583e-02    -8.939e-01
    m= 12 -1.004e-02    -1.070e+00
    m= 13 +3.722e-01    -1.287e+00
    m= 14 -1.319e-01    +1.200e+00
    m= 15 +2.773e-01    +6.971e-01
    m= 16 -5.939e-03    -9.411e-01
    m= 17 -5.283e+00    -7.638e-01
    m= 18 -6.717e-01    -1.706e+00
    m= 19 +2.313e-01    -1.799e+00
```

## Gates
- Final loss > 0.5 (non-interpolating): **PASS** (1.8652)
- LOO Pearson > 0.7: **FAIL** (+0.0858)
- LOO Kendall τ > 0.6: **FAIL** (-0.0218)
- Sign agreement > 80/100: **FAIL** (48/100)

## Wall-clock
- Train θ*: 29.14 s
- 100 LOO retrainings: 1495.98 s
- Stage 1 (A, S, Λ): 0.237 s
- g_eval + IHVP: 0.0007 s
- 100 per-sample IFs: 0.047 s (0.47 ms/IF)
- **Total**: 1525.45 s

## Production extrapolation
- Toy per-IF: 0.47 ms (CPU)
- Real-model per-IF (GPT-Neo-125M, d=(768,3072), R≈30, GPU): roughly **5198 ms** (rough scaling: ratio of d_in·d_out × tokens / GPU-vs-CPU)
- For 1000 production rollouts: ≈ **5198 s** at the IF-loop stage. Stage 1 (one-shot) ~30 s (see test_eigen_real).
