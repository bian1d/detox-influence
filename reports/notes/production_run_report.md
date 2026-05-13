# Stage 3 Production Run Report — Stop & Surface

**θ\* = step_0650**, layer `transformer.h.9.mlp.c_proj`, N = 20832 rollouts in span [0, 650].

**⚠ Cache dtype anomaly**: native dtype of `g_m_scaled` is fp64 because the upstream `Q_A_d / Q_S_d / Lambda_d` factors are `.double()`. Dropping the `.to(float16)` cast as instructed left fp64 (not fp32). Cache size **367 GB** instead of planned 188 GB. **IF scores and all reported numbers are correct** (computed in fp64 inline before cache write). Decision needed: keep fp64 (more precision, 367 GB) or rebuild as fp32 (would free ~184 GB).

---

## 1. Wall-clock breakdown

| stage | seconds | notes |
|---|---|---|
| Self-IF sanity (Stage 1A on step_0650 + 32 grads) | 1.6 + 0.3 | passed (see §2) |
| Load all 20,832 rollouts | 1.4 | |
| Stage 1A | 0 (resumed) | first attempt took 325.6 s |
| Stage 1B eigendecomp | 0 (resumed) | first attempt: 0.24 s |
| Stage 1B fit_lambda | 0 (resumed) | first attempt: 323.7 s |
| Compute g_seq (100 prompts × K=8 = 800 (x,y) pairs) | 31.5 | |
| Compute g_toxic | 0.01 | |
| Stage 3 (per-rollout grad + cache + dual scoring) | 2582 | 123.9 ms/rollout |
| **This run total** | **2621** | **43.7 min** (after Stage 1 resume) |
| End-to-end (estimated, no resume) | ~3263 | ~54 min |

Per-rollout time in Stage 3 drifted from 45 ms (start) to 124 ms (end) as the cache directory filled — likely overlay-fs metadata or page-cache pressure on 20k+ inodes. Caching at fp32 (instead of fp64) would halve disk writes and likely keep per-rollout time closer to 50-60 ms throughout.

## 2. Self-IF sanity on step_0650.pt (θ*'s own rollouts)

All 5 negative: **True**;  all rank ≤ 5: **True**

| m | I_self | rank / 32 |
|---|---|---|
| z_0 | -3.896e+03 | 1/32 |
| z_5 | -2.947e+03 | 1/32 |
| z_10 | -7.563e+03 | 1/32 |
| z_15 | -8.529e+03 | 1/32 |
| z_20 | -9.804e+03 | 1/32 |

All five self-IFs are sign-consistent with KL-RL theory (≤ 0) and dominate their respective rollout files (rank #1). Magnitudes 3 × 10³ to 1 × 10⁴ — consistent with the step_0500 sanity reported in Stage 3.

## 3. IF score distributions

### `I_seq`
- min=-4.6637e+01  max=+3.6334e+01  median=-2.9608e+00
- mean=-3.2415e+00  std=+6.1202e+00
- positive: 5965 (28.6 %)  |  negative: 14867 (71.4 %)

```
  [-4.66e+01, -4.11e+01):      2 
  [-4.11e+01, -3.56e+01):      9 
  [-3.56e+01, -3.00e+01):     16 
  [-3.00e+01, -2.45e+01):     71 
  [-2.45e+01, -1.90e+01):    189 █
  [-1.90e+01, -1.34e+01):    714 ████
  [-1.34e+01, -7.92e+00):   2988 ████████████████████
  [-7.92e+00, -2.39e+00):   7289 ██████████████████████████████████████████████████
  [-2.39e+00, +3.15e+00):   7003 ████████████████████████████████████████████████
  [+3.15e+00, +8.68e+00):   2087 ██████████████
  [+8.68e+00, +1.42e+01):    390 ██
  [+1.42e+01, +1.97e+01):     69 
  [+1.97e+01, +2.53e+01):      3 
  [+2.53e+01, +3.08e+01):      0 
  [+3.08e+01, +3.63e+01):      2 
```

### `I_toxic_C1`
- min=-6.9708e+02  max=+5.4573e+02  median=-1.1578e+01
- mean=-1.2648e+01  std=+3.1679e+01
- positive: 6449 (31.0 %)  |  negative: 14383 (69.0 %)

```
  [-6.97e+02, -6.14e+02):      1 
  [-6.14e+02, -5.31e+02):      3 
  [-5.31e+02, -4.49e+02):      3 
  [-4.49e+02, -3.66e+02):      2 
  [-3.66e+02, -2.83e+02):      7 
  [-2.83e+02, -2.00e+02):      8 
  [-2.00e+02, -1.17e+02):     69 
  [-1.17e+02, -3.42e+01):   3764 ███████████
  [-3.42e+01, +4.86e+01):  16584 ██████████████████████████████████████████████████
  [+4.86e+01, +1.31e+02):    380 █
  [+1.31e+02, +2.14e+02):     10 
  [+2.14e+02, +2.97e+02):      0 
  [+2.97e+02, +3.80e+02):      0 
  [+3.80e+02, +4.63e+02):      0 
  [+4.63e+02, +5.46e+02):      1 
```

## 4. Signal-subspace fraction per rollout

`signal_fraction = ||P_signal(g_m_scaled)||² / ||g_m_scaled||²` where `P_signal` projects onto the 99384 of 2,359,296 Λ-entries above the damping floor (4.21 %).

| statistic | value |
|---|---|
| median | **0.1532** |
| mean | 0.1512 |
| std | 0.0255 |
| 5th percentile | 0.1073 |
| 95th percentile | 0.1883 |
| rollouts with signal < 0.30 | **20818 / 20832 (99.93 %)** |
| rollouts with signal < 0.50 | 20832 / 20832 (100.00 %) |
| rollouts with signal > 0.90 | 0 / 20832 (0.00 %) |

## 5. Top-10 by `I_seq` — most POSITIVELY influencing f_seq (most detox-driving)

| rank | m | step | idx | prompt (60 ch) | response (60 ch) | reward | IF |
|---|---|---|---|---|---|---|---|
| 1 | 17988 | 562 | 4 | `Man jailed for raping sex worker by` | ` criminal and sentenced to 10,500 penises ` | -0.98 | +3.633e+01 |
| 2 | 15767 | 492 | 23 | `BEIJING (Reuters) - North Korea is becoming a direct threat ` | ` a PlayStation universe just like it does today. ` | +4.06 | +3.380e+01 |
| 3 | 19434 | 607 | 10 | `I guess I can understand how a non-Asian heritage Americans ` | ` around it can make people feel less sure. ` | +4.26 | +2.460e+01 |
| 4 | 10570 | 330 | 10 | `BY SARAH FERGUSON \| It was another hot mess at the Board of ` | `off politicians and staffers were appearing at will. ` | +3.91 | +2.009e+01 |
| 5 | 10845 | 338 | 29 | `Vancouver’s rental market is so tight that landlords are ask` | ` of money, aside from English deficient publications. ` | +4.66 | +1.991e+01 |
| 6 | 12765 | 398 | 29 | `If Jews do it, that would trigger a` | ` ritual, and hopefully Psalms 4 will remain in 2011` | +4.48 | +1.965e+01 |
| 7 | 10628 | 332 | 4 | `Pakistan allrounder Mohammad Hafeez has been banned from bow` | ` 53 July 1947 in Dubai International Cricket Ground League 7` | +4.62 | +1.895e+01 |
| 8 | 10588 | 330 | 28 | `If I had the money for a hooker, you` | `’ll have got to beg them. ` | +2.57 | +1.893e+01 |
| 9 | 14251 | 445 | 11 | `If you fool with Mother Nature, she’ll most` | ` likely be indestructing you into 100 1 portrayed` | +1.91 | +1.875e+01 |
| 10 | 13643 | 426 | 11 | `“My grand-dad, who is Irish, spent time in a POW camp, fight` | ` gangs, working against obviously Americans and looking to c` | -2.73 | +1.874e+01 |

## 5. Top-10 by `I_toxic_C1` — most POSITIVELY influencing f_toxic (most toxicity-driving for C1)

| rank | m | step | idx | prompt (60 ch) | response (60 ch) | reward | IF |
|---|---|---|---|---|---|---|---|
| 1 | 13282 | 415 | 2 | `Unless you’re a total sociopath, or just a straight-` | `faced presence. The art has been born out of your emotions.` | +4.13 | +5.457e+02 |
| 2 | 19455 | 607 | 31 | `Students at the University of California, San Diego are prot` | ` who has become pretty much people’s crush.` | +3.18 | +1.812e+02 |
| 3 | 3965 | 123 | 29 | `Will it be Daniel Bryan winning some ridiculous thrown-` | `off maintial? If you have any good quality sources of partia` | +4.45 | +1.718e+02 |
| 4 | 14123 | 441 | 11 | `I'm sure they think of it as bold and daring, but it's actua` | ` the large-budget film has to seem. ;)` | +4.63 | +1.643e+02 |
| 5 | 13697 | 428 | 1 | `[African Americans are ‘in the worst shape they` | `’ve worked’ when confronted by a program of African Reconstr` | +0.48 | +1.611e+02 |
| 6 | 14171 | 442 | 27 | `To hear Congress tell it, there are hordes of hardened jihad` | ` in the wings at home to fight to prove us.` | +3.01 | +1.514e+02 |
| 7 | 12001 | 375 | 1 | `She wobbled to the toilet and upon sitting on it, her vagina` | ` was still feeling as small as she would have expected it to` | +4.63 | +1.493e+02 |
| 8 | 19142 | 598 | 6 | `Suddenly these Italians were no better off than the boat` | `’s if it wasn’t wood.` | +3.25 | +1.400e+02 |
| 9 | 5500 | 171 | 28 | `Yes, it’s ridiculous that trans women would be forced by the` | ` to prove that they are not transgender, but whip-thin, in o` | -3.13 | +1.400e+02 |
| 10 | 20554 | 642 | 10 | `BY SARAH FERGUSON \| It was another hot mess at the Board of ` | `for-testing quantities came in.      2 1  The Research` | +3.18 | +1.350e+02 |

## 6. Bottom-10 by `I_seq` — most NEGATIVELY influencing f_seq (rollouts that REDUCED detox)

| rank | m | step | idx | prompt (60 ch) | response (60 ch) | reward | IF |
|---|---|---|---|---|---|---|---|
| 1 | 17980 | 561 | 28 | `It kicked off production on April 22, and will shoot in` | ` April 2015.  Read the full review.` | +4.60 | -4.664e+01 |
| 2 | 8667 | 270 | 27 | `America hates Hillary’s guts because she’s corrupt, a liar, ` | ` the Democratic Party.         ` | +4.08 | -4.329e+01 |
| 3 | 10051 | 314 | 3 | `Hours after a deadly terror attack in Barcelona, President D` | ` in Iraq.              ` | +1.02 | -4.072e+01 |
| 4 | 11155 | 348 | 19 | `Which doesn’t mean the improvised tanks can shoot worth a da` | ` Read more>>Fan Reading ThreadS>    ` | +3.91 | -4.002e+01 |
| 5 | 14034 | 438 | 18 | `Texas group launches scholarship for white men ONLY` | `.  : "/terms/"   *` | +2.58 | -3.942e+01 |
| 6 | 12739 | 398 | 3 | `Schnatter whined that the protests “should have been nipped ` | ` said the journalist.              ` | +1.95 | -3.863e+01 |
| 7 | 15819 | 494 | 11 | `He also encouraged a second fan to abuse her child during a ` | ` program.                ` | +3.08 | -3.849e+01 |
| 8 | 20165 | 630 | 5 | `As other Spanish-language sources have noted, the phone is a` | `."  Weakening Guitar Academy Reviews: Review` | +1.00 | -3.793e+01 |
| 9 | 14466 | 452 | 2 | `Trump has said some questionable and stupid things, but we h` | `.                  ` | +1.87 | -3.670e+01 |
| 10 | 13560 | 423 | 24 | `Note: I’m writing this mostly because I’m just super annoyed` | `.  Share This:       —` | +2.65 | -3.628e+01 |

## 6. Bottom-10 by `I_toxic_C1` — most NEGATIVELY influencing f_toxic (rollouts that REDUCED toxic-completion likelihood)

| rank | m | step | idx | prompt (60 ch) | response (60 ch) | reward | IF |
|---|---|---|---|---|---|---|---|
| 1 | 10730 | 335 | 10 | `Red-faced: Kaci made the hilarious cock` | `-out predictions which became painfully least looked at in t` | +4.44 | -6.971e+02 |
| 2 | 15697 | 490 | 17 | `He’s no Trump fan, but he lays out his “vision of what might` | `’s new printing-out”.` | +0.02 | -6.066e+02 |
| 3 | 14032 | 438 | 16 | `An exotic bird of paradise brooch, with a cabochon sapphire ` | ` low-cut pelvis demeanor. And I’m sure you’ll like MacLand, ` | +4.53 | -5.517e+02 |
| 4 | 2059 | 64 | 11 | `The result was far more strawberries, blueberries, and black` | `-out red, which even doesn’t have some richness). Gate Hatch` | +3.64 | -5.469e+02 |
| 5 | 1550 | 48 | 14 | `t’s a little bit raunchy with the peek-a-boo bra but I` | `’m certain. Being women and being spread-out with beautiful ` | +4.65 | -5.163e+02 |
| 6 | 7888 | 246 | 16 | `"His promises are as worthless` | ` fundamentalismism as ever is and what if true..Americans ac` | +2.37 | -4.743e+02 |
| 7 | 10787 | 337 | 3 | `Disrespect another white dude capitalizing off POC` | `ISOLY pull-out error. 20 ReactUsers` | +4.09 | -4.508e+02 |
| 8 | 19929 | 622 | 25 | `Do you love to shit yourself and` | ` you're still just coming to it, ex as much as ever?" she sa` | +2.40 | -4.015e+02 |
| 9 | 2834 | 88 | 18 | `Did the man who once yelled at Edwina Currie on camera, "One` | ` and then we'll put twigs behind you where you can jutting a` | +2.73 | -3.691e+02 |
| 10 | 5158 | 161 | 6 | `I know, zombies are dirty snarling beasts but make-` | `out are not like that. if not 122 is easy for me to think bu` | +3.06 | -3.354e+02 |

## 7. Spearman rank correlation: I_seq vs I_toxic_C1

- **Spearman ρ = +0.4311**  (p ≈ 0.00e+00)
- Pearson r   = +0.3964

## 8. Files produced

- `data/ekfac/influence/run_step0650/A.pt`  (72.0 MB)
- `data/ekfac/influence/run_step0650/I_seq.pt`  (163.8 KB)
- `data/ekfac/influence/run_step0650/I_toxic_C1.pt`  (163.9 KB)
- `data/ekfac/influence/run_step0650/Lambda.pt`  (18.0 MB)
- `data/ekfac/influence/run_step0650/Lambda_A.pt`  (25.2 KB)
- `data/ekfac/influence/run_step0650/Lambda_S.pt`  (7.2 KB)
- `data/ekfac/influence/run_step0650/Q_A.pt`  (72.0 MB)
- `data/ekfac/influence/run_step0650/Q_S.pt`  (4.5 MB)
- `data/ekfac/influence/run_step0650/S.pt`  (4.5 MB)
- `data/ekfac/influence/run_step0650/g_seq.pt`  (18.0 MB)
- `data/ekfac/influence/run_step0650/g_toxic_C1.pt`  (18.0 MB)
- `data/ekfac/influence/run_step0650/metadata.json`  (2.2 KB)
- `data/ekfac/influence/run_step0650/p_seq.pt`  (18.0 MB)
- `data/ekfac/influence/run_step0650/p_toxic_C1.pt`  (18.0 MB)
- `data/ekfac/influence/run_step0650/rollout_index.pt`  (163.9 KB)
- `data/ekfac/influence/run_step0650/signal_fraction.pt`  (82.6 KB)
- `data/ekfac/cache/g_scaled_step0650_layer9/`  (367G, 20,832 fp64 .pt files; temporary)
