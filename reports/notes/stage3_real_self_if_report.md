# Stage 3 real-data self-influence sanity report

**Setup:** GPT-Neo-125M (post-PPO), layer `transformer.h.9.mlp.c_proj`, rollouts from `data/rollouts/step_0500.pt` (n=32). Production damping `max(Λ + 0.1·Λ̄, 1e-05)`. IF sign convention: KL-RL (`influence_score_klrl`).

Self-influence is theoretically ``I_self = −s_mᵀ F⁻¹ s_m ≤ 0`` under the KL-RL convention (F⁻¹ is PSD with damping).

## Per-self-target results

| self idx m | I_self (signed) | |I_self|  | rank of |I_self| / 32 |
|---|---|---|---|
| z_0 | -1.2174e+04 | 1.2174e+04 | **1/32** |
| z_5 | -1.7765e+04 | 1.7765e+04 | **1/32** |
| z_10 | -1.4035e+04 | 1.4035e+04 | **1/32** |
| z_15 | -8.1267e+03 | 8.1267e+03 | **1/32** |
| z_20 | -6.4646e+03 | 6.4646e+03 | **1/32** |

## Distribution of |I| for each self-target (showing top-5)
- self z_0: |I(z_0|z_0)|=1.217e+04, |I(z_12|z_0)|=2.302e+02, |I(z_4|z_0)|=1.212e+02, |I(z_22|z_0)|=9.786e+01, |I(z_8|z_0)|=9.719e+01
- self z_5: |I(z_5|z_5)|=1.776e+04, |I(z_18|z_5)|=2.144e+02, |I(z_27|z_5)|=2.130e+02, |I(z_28|z_5)|=1.718e+02, |I(z_9|z_5)|=1.181e+02
- self z_10: |I(z_10|z_10)|=1.403e+04, |I(z_28|z_10)|=1.190e+02, |I(z_21|z_10)|=8.675e+01, |I(z_18|z_10)|=6.622e+01, |I(z_11|z_10)|=6.093e+01
- self z_15: |I(z_15|z_15)|=8.127e+03, |I(z_18|z_15)|=1.264e+02, |I(z_22|z_15)|=1.223e+02, |I(z_28|z_15)|=1.192e+02, |I(z_24|z_15)|=1.075e+02
- self z_20: |I(z_20|z_20)|=6.465e+03, |I(z_27|z_20)|=1.355e+02, |I(z_18|z_20)|=1.254e+02, |I(z_24|z_20)|=1.225e+02, |I(z_22|z_20)|=1.207e+02

## Gates
- All 5 self-IFs are NEGATIVE: **PASS** (5/5)
- All 5 self-IFs rank in top-5 of |I|: **PASS** (ranks: [1, 1, 1, 1, 1])

## Wall-clock
- Stage 1A+1B (A, S, Λ): 1.66 s
- 32 per-sample gradient computations: 0.38 s
- 5 IHVPs + 5×32 dot products: 0.160 s
