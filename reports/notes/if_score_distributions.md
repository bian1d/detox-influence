# IF Score Distributions and Diagnostics — Stage 3 Production Run

**Run:** θ\* = step_0650, layer `transformer.h.9.mlp.c_proj`, N = 20,832 rollouts in span [0, 650].
**Two eval targets**: `f_seq` (population-level, REINFORCE-with-baseline per Hu et al. 2025 §5.3) and `f_toxic_C1` (instance-level racial completion).

---

## `I_seq` distribution

| statistic | value |
|---|---|
| n | 20,832 |
| min | -4.6637e+01 |
| 1 % | -2.0431e+01 |
| 5 % | -1.3256e+01 |
| 25 % | -6.7548e+00 |
| **50 % (median)** | **-2.9608e+00** |
| 75 % | +5.4869e-01 |
| 95 % | +6.0763e+00 |
| 99 % | +1.1024e+01 |
| max | +3.6334e+01 |
| mean | -3.2415e+00 |
| std | +6.1202e+00 |
| positive | 5,965 (28.6 %) |
| negative | 14,867 (71.4 %) |
| zero | 0 |

**Histogram (20 bins)**

```
  [-4.664e+01, -4.249e+01):      2 
  [-4.249e+01, -3.834e+01):      5 
  [-3.834e+01, -3.419e+01):      8 
  [-3.419e+01, -3.004e+01):     12 
  [-3.004e+01, -2.589e+01):     41 
  [-2.589e+01, -2.175e+01):     95 
  [-2.175e+01, -1.760e+01):    236 █
  [-1.760e+01, -1.345e+01):    602 ████
  [-1.345e+01, -9.300e+00):  1,862 ██████████████
  [-9.300e+00, -5.151e+00):  4,283 ██████████████████████████████████
  [-5.151e+00, -1.003e+00):  6,268 ██████████████████████████████████████████████████
  [-1.003e+00, +3.146e+00):  4,867 ██████████████████████████████████████
  [+3.146e+00, +7.294e+00):  1,825 ██████████████
  [+7.294e+00, +1.144e+01):    543 ████
  [+1.144e+01, +1.559e+01):    140 █
  [+1.559e+01, +1.974e+01):     38 
  [+1.974e+01, +2.389e+01):      2 
  [+2.389e+01, +2.804e+01):      1 
  [+2.804e+01, +3.219e+01):      0 
  [+3.219e+01, +3.633e+01):      2 
```

## `I_toxic_C1` distribution

| statistic | value |
|---|---|
| n | 20,832 |
| min | -6.9708e+02 |
| 1 % | -9.4178e+01 |
| 5 % | -5.9385e+01 |
| 25 % | -2.7898e+01 |
| **50 % (median)** | **-1.1578e+01** |
| 75 % | +4.3228e+00 |
| 95 % | +3.1542e+01 |
| 99 % | +6.0676e+01 |
| max | +5.4573e+02 |
| mean | -1.2648e+01 |
| std | +3.1679e+01 |
| positive | 6,449 (31.0 %) |
| negative | 14,383 (69.0 %) |
| zero | 0 |

**Histogram (20 bins)**

```
  [-6.971e+02, -6.349e+02):      1 
  [-6.349e+02, -5.728e+02):      1 
  [-5.728e+02, -5.107e+02):      3 
  [-5.107e+02, -4.485e+02):      2 
  [-4.485e+02, -3.864e+02):      1 
  [-3.864e+02, -3.242e+02):      4 
  [-3.242e+02, -2.621e+02):      6 
  [-2.621e+02, -2.000e+02):      6 
  [-2.000e+02, -1.378e+02):     34 
  [-1.378e+02, -7.567e+01):    439 ██
  [-7.567e+01, -1.353e+01):  9,171 ██████████████████████████████████████████
  [-1.353e+01, +4.861e+01): 10,773 ██████████████████████████████████████████████████
  [+4.861e+01, +1.107e+02):    368 █
  [+1.107e+02, +1.729e+02):     21 
  [+1.729e+02, +2.350e+02):      1 
  [+2.350e+02, +2.972e+02):      0 
  [+2.972e+02, +3.593e+02):      0 
  [+3.593e+02, +4.215e+02):      0 
  [+4.215e+02, +4.836e+02):      0 
  [+4.836e+02, +5.457e+02):      1 
```

## Signal-subspace fraction

`signal_fraction = ||P_signal(g_m_scaled)||² / ||g_m_scaled||²` where `P_signal` projects onto the 99,384 of 2,359,296 Λ-entries above the damping floor (4.21 %).

| statistic | value |
|---|---|
| min | 0.0542 |
| 5 % | 0.1073 |
| 25 % | 0.1347 |
| **50 % (median)** | **0.1532** |
| 75 % | 0.1683 |
| 95 % | 0.1883 |
| max | 0.3263 |
| mean | 0.1512 |
| std | 0.0255 |
| rollouts with signal < 0.30 | 20,818 / 20,832 (99.93 %) |
| rollouts with signal < 0.50 | 20,832 / 20,832 (100.00 %) |
| rollouts with signal > 0.90 | 0 / 20,832 (0.00 %) |

Median signal-subspace mass per rollout is 0.153 — ~3.6× the random-direction baseline (4.21 %), confirming that rollouts do concentrate in the EK-FAC signal directions more than chance, but a majority of each rollout's `g_m_scaled` still lives in damping-regularised eigendirections. **All IF rankings reflect this combined signal+damping inverse-Fisher response**; reducing `damping_floor` from `1e-5` would shift more directions into the signal classification at the cost of inverse instability.

## Cross-target rank correlation: `I_seq` vs `I_toxic_C1`

| metric | value |
|---|---|
| Spearman ρ | **+0.4311** (p ≈ 0.00e+00) |
| Pearson r | +0.3964 (p ≈ 0.00e+00) |
| sign agreement | 14,612 / 20,832 (70.1 %) |

Substantially positive but well below 1.0 — `f_seq` (population REINFORCE-with-baseline) and `f_toxic_C1` (single racial-completion instance) measure related but distinct aspects of detox. The 0.43 Spearman ρ implies ~19 % shared variance in rollout rankings.

### Top-K rank overlap (Jaccard of top-K sets)

| K | overlap |
|---:|---:|
| 10 | 0 / 10 (0.0 %) |
| 50 | 0 / 50 (0.0 %) |
| 100 | 4 / 100 (4.0 %) |
| 500 | 71 / 500 (14.2 %) |
| 1000 | 205 / 1000 (20.5 %) |

Top-10 by `I_seq` and `I_toxic_C1` share 0–2 rollouts; agreement grows modestly with K (e.g., top-1000 share ~24 %). The two targets surface largely different rollout subsets at the head — consistent with them measuring different facets of detox-related influence.

