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
- Pearson(I_ekfac, I_true):  **+0.0229**
- Kendall τ:                  **-0.0149**
- Sign agreement:             **54/100**

### Brute-force F⁻¹ control (zero EK-FAC approximation error)
- Pearson(I_brute_force, I_true): **-0.0309**
- Kendall τ:                       **-0.0913**
- Sign agreement:                  **48/100**
- *Interpretation: an upper bound on what any inverse-Fisher approximation can achieve in this regime; gap between brute-force and EK-FAC isolates structural approximation error.*

Per-rollout sample (first 20 of 100):
```
    m   I_ekfac        I_true
    m=  0 +6.421e-03    -1.012e+00
    m=  1 -6.631e-01    -7.969e-01
    m=  2 -1.154e-02    -2.690e-01
    m=  3 -2.273e-02    +6.152e-02
    m=  4 -3.121e-02    +9.556e-02
    m=  5 -1.684e-02    -9.316e-01
    m=  6 +4.258e-02    -9.380e-01
    m=  7 +3.670e-01    -1.131e+00
    m=  8 +6.413e-02    -8.932e-01
    m=  9 +1.071e-01    -2.032e+00
    m= 10 -3.160e-02    +3.593e-01
    m= 11 -5.917e-03    -8.939e-01
    m= 12 -6.845e-04    -1.070e+00
    m= 13 +9.465e-02    -1.287e+00
    m= 14 -1.315e-02    +1.200e+00
    m= 15 +9.997e-02    +6.971e-01
    m= 16 +4.002e-03    -9.411e-01
    m= 17 -1.942e+00    -7.638e-01
    m= 18 -9.210e-02    -1.706e+00
    m= 19 -3.168e-02    -1.799e+00
```

## Gates
- Final loss > 0.5 (non-interpolating): **PASS** (1.8652)
- LOO Pearson > 0.7: **FAIL** (+0.0229)
- LOO Kendall τ > 0.6: **FAIL** (-0.0149)
- Sign agreement > 80/100: **FAIL** (54/100)

## Wall-clock
- Train θ*: 16.61 s
- 100 LOO retrainings: 1432.40 s
- Stage 1 (A, S, Λ): 0.239 s
- g_eval + IHVP: 0.0008 s
- 100 per-sample IFs: 0.048 s (0.48 ms/IF)
- **Total**: 1449.35 s

## Production extrapolation
- Toy per-IF: 0.48 ms (CPU)
- Real-model per-IF (GPT-Neo-125M, d=(768,3072), R≈30, GPU): roughly **5278 ms** (rough scaling: ratio of d_in·d_out × tokens / GPU-vs-CPU)
- For 1000 production rollouts: ≈ **5278 s** at the IF-loop stage. Stage 1 (one-shot) ~30 s (see test_eigen_real).
