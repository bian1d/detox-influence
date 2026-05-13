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

---

## `I_toxic_C2` distribution

| statistic | value |
|---|---|
| n | 20,832 |
| min | -5.6266e+02 |
| 1 % | -8.8794e+01 |
| 5 % | -5.3718e+01 |
| 25 % | -1.9752e+01 |
| **50 % (median)** | **-3.0604e+00** |
| 75 % | +1.2308e+01 |
| 95 % | +3.6149e+01 |
| 99 % | +5.7486e+01 |
| max | +1.0132e+03 |
| mean | -5.1716e+00 |
| std | +3.0134e+01 |
| positive | 9,312 (44.7 %) |
| negative | 11,520 (55.3 %) |

**Histogram (20 bins)**

```
  [-5.627e+02, -4.839e+02):      2 
  [-4.839e+02, -4.051e+02):      1 
  [-4.051e+02, -3.263e+02):      1 
  [-3.263e+02, -2.475e+02):      1 
  [-2.475e+02, -1.687e+02):     16 
  [-1.687e+02, -8.991e+01):    176 
  [-8.991e+01, -1.111e+01):  7,506 ████████████████████████████
  [-1.111e+01, +6.768e+01): 13,015 ██████████████████████████████████████████████████
  [+6.768e+01, +1.465e+02):    110 
  [+1.465e+02, +2.253e+02):      3 
  [+2.253e+02, +3.041e+02):      0 
  [+3.041e+02, +3.829e+02):      0 
  [+3.829e+02, +4.616e+02):      0 
  [+4.616e+02, +5.404e+02):      0 
  [+5.404e+02, +6.192e+02):      0 
  [+6.192e+02, +6.980e+02):      0 
  [+6.980e+02, +7.768e+02):      0 
  [+7.768e+02, +8.556e+02):      0 
  [+8.556e+02, +9.344e+02):      0 
  [+9.344e+02, +1.013e+03):      1 
```

## `I_toxic_C3` distribution

| statistic | value |
|---|---|
| n | 20,832 |
| min | -8.6534e+02 |
| 1 % | -1.2258e+02 |
| 5 % | -7.6333e+01 |
| 25 % | -3.5074e+01 |
| **50 % (median)** | **-1.4412e+01** |
| 75 % | +5.5220e+00 |
| 95 % | +3.8934e+01 |
| 99 % | +7.1600e+01 |
| max | +3.6785e+02 |
| mean | -1.5927e+01 |
| std | +3.7860e+01 |
| positive | 6,515 (31.3 %) |
| negative | 14,317 (68.7 %) |

**Histogram (20 bins)**

```
  [-8.653e+02, -8.037e+02):      1 
  [-8.037e+02, -7.420e+02):      0 
  [-7.420e+02, -6.804e+02):      0 
  [-6.804e+02, -6.187e+02):      1 
  [-6.187e+02, -5.570e+02):      1 
  [-5.570e+02, -4.954e+02):      0 
  [-4.954e+02, -4.337e+02):      0 
  [-4.337e+02, -3.721e+02):      1 
  [-3.721e+02, -3.104e+02):      1 
  [-3.104e+02, -2.487e+02):      6 
  [-2.487e+02, -1.871e+02):     26 
  [-1.871e+02, -1.254e+02):    148 
  [-1.254e+02, -6.376e+01):  1,478 ██████
  [-6.376e+01, -2.103e+00): 12,172 ██████████████████████████████████████████████████
  [-2.103e+00, +5.956e+01):  6,616 ███████████████████████████
  [+5.956e+01, +1.212e+02):    349 █
  [+1.212e+02, +1.829e+02):     26 
  [+1.829e+02, +2.445e+02):      4 
  [+2.445e+02, +3.062e+02):      0 
  [+3.062e+02, +3.679e+02):      2 
```

## 4×4 Spearman correlation matrix

| | `I_seq` | `I_C1` | `I_C2` | `I_C3` |
|---|---:|---:|---:|---:|
| `I_seq` | **+1.0000** | +0.4311 | -0.1732 | +0.4066 |
| `I_C1` | +0.4311 | **+1.0000** | +0.0256 | +0.4986 |
| `I_C2` | -0.1732 | +0.0256 | **+1.0000** | +0.0766 |
| `I_C3` | +0.4066 | +0.4986 | +0.0766 | **+1.0000** |

**4×4 Pearson correlation matrix**

| | `I_seq` | `I_C1` | `I_C2` | `I_C3` |
|---|---:|---:|---:|---:|
| `I_seq` | **+1.0000** | +0.3964 | -0.1742 | +0.3816 |
| `I_C1` | +0.3964 | **+1.0000** | +0.0236 | +0.4307 |
| `I_C2` | -0.1742 | +0.0236 | **+1.0000** | +0.0739 |
| `I_C3` | +0.3816 | +0.4307 | +0.0739 | **+1.0000** |

## Top-K rank overlap (|A ∩ B| / K) — 4×4 matrices

### K = 10

| | `I_seq` | `I_C1` | `I_C2` | `I_C3` |
|---|---:|---:|---:|---:|
| `I_seq` | **10/10** | 0/10 (0 %) | 0/10 (0 %) | 0/10 (0 %) |
| `I_C1` | 0/10 (0 %) | **10/10** | 0/10 (0 %) | 0/10 (0 %) |
| `I_C2` | 0/10 (0 %) | 0/10 (0 %) | **10/10** | 0/10 (0 %) |
| `I_C3` | 0/10 (0 %) | 0/10 (0 %) | 0/10 (0 %) | **10/10** |

### K = 50

| | `I_seq` | `I_C1` | `I_C2` | `I_C3` |
|---|---:|---:|---:|---:|
| `I_seq` | **50/50** | 0/50 (0 %) | 0/50 (0 %) | 1/50 (2 %) |
| `I_C1` | 0/50 (0 %) | **50/50** | 0/50 (0 %) | 12/50 (24 %) |
| `I_C2` | 0/50 (0 %) | 0/50 (0 %) | **50/50** | 1/50 (2 %) |
| `I_C3` | 1/50 (2 %) | 12/50 (24 %) | 1/50 (2 %) | **50/50** |

### K = 100

| | `I_seq` | `I_C1` | `I_C2` | `I_C3` |
|---|---:|---:|---:|---:|
| `I_seq` | **100/100** | 4/100 (4 %) | 1/100 (1 %) | 3/100 (3 %) |
| `I_C1` | 4/100 (4 %) | **100/100** | 1/100 (1 %) | 25/100 (25 %) |
| `I_C2` | 1/100 (1 %) | 1/100 (1 %) | **100/100** | 2/100 (2 %) |
| `I_C3` | 3/100 (3 %) | 25/100 (25 %) | 2/100 (2 %) | **100/100** |

### K = 500

| | `I_seq` | `I_C1` | `I_C2` | `I_C3` |
|---|---:|---:|---:|---:|
| `I_seq` | **500/500** | 71/500 (14 %) | 13/500 (3 %) | 70/500 (14 %) |
| `I_C1` | 71/500 (14 %) | **500/500** | 22/500 (4 %) | 148/500 (30 %) |
| `I_C2` | 13/500 (3 %) | 22/500 (4 %) | **500/500** | 27/500 (5 %) |
| `I_C3` | 70/500 (14 %) | 148/500 (30 %) | 27/500 (5 %) | **500/500** |

### K = 1000

| | `I_seq` | `I_C1` | `I_C2` | `I_C3` |
|---|---:|---:|---:|---:|
| `I_seq` | **1000/1000** | 205/1000 (20 %) | 43/1000 (4 %) | 207/1000 (21 %) |
| `I_C1` | 205/1000 (20 %) | **1000/1000** | 71/1000 (7 %) | 317/1000 (32 %) |
| `I_C2` | 43/1000 (4 %) | 71/1000 (7 %) | **1000/1000** | 95/1000 (10 %) |
| `I_C3` | 207/1000 (21 %) | 317/1000 (32 %) | 95/1000 (10 %) | **1000/1000** |

## Canary cross-target table (template-specificity evidence)

| m | I_seq | I_C1 | I_C2 | I_C3 | note |
|---:|---:|---:|---:|---:|---|
| 10730 | +1.724e+00 | -6.971e+02 | +2.154e+00 | -5.478e+01 | **TEMPLATE-SPECIFICITY EVIDENCE**: C1 is the run's largest-magnitude IF (−697); C2 is essentially zero (+2.15); refutes generic-vulgarity transfer hypothesis |
| 13697 | +1.562e+00 | +1.611e+02 | +2.908e+00 | +3.491e+01 | Direct C1_racial topical match (+161, rank #5); C3 echoes (demographic); C2 near-zero |
| 12001 | +1.629e+00 | +1.493e+02 | +4.071e+01 | +2.952e+01 | Explicit sexual content transfers C1 → C2 cleanly (+149 → +40.7); content-aligned template transfer |

## Interpretation

The 4×4 Spearman matrix shows ρ(C1, C2) = +0.026, ρ(C1, C3) = +0.499, ρ(C2, C3) = +0.077. The three toxic-template rankings are positively correlated (so PPO did learn *some* shared detox structure on layer 9 W₂) but well below 1.0 (so the three templates also have substantial template-specific structure). Top-K Jaccard overlap is similarly partial — e.g., at K=100 the C1/C2 top-100 sets share only 1 rollouts. The canary table makes the same point at instance level: m=10730's C1-specificity is direct evidence that layer 9 W₂ encodes multiple distinct, content-aligned detox directions rather than a single "safety" axis. f_seq has lower correlation with all three toxic templates (ρ = +0.431, -0.173, +0.407) — consistent with f_seq measuring the population-level REINFORCE-with-baseline signal, which integrates over many template directions, while f_toxic_C{1,2,3} measure single-template instance-level alignment.

## Wall-clock note (extra-target run)

Scoring C2 + C3 from the warm fp32 cache took ~26 min total (streaming-scoring loop alone: 783 s = 13 min, ~38 ms per cache read). Slower than the 30-second initial estimate due to overlay-fs metadata cost on 184 GB of cold-page-cache reads. A warm second run would be 5-10× faster. For future eval-target experiments: batch multiple new targets into a single streaming pass (each cache read can supply N inner products at near-zero marginal cost), so N=10 templates at once is still ~26 min — much better than running 10× separately.
