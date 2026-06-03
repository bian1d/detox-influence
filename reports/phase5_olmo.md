# Phase 5 — OLMo-2-1B: A2-failure, PPO's fault or the method's?

Companion baseline: `reports/phase4_gptneo_negative_result.md` (GPT-Neo kill).
Plain-language throughout (no math rendering).

---

## External dependencies LANDED (2026-06-02, second pass)

Both Section-0 blockers are resolved. This section supersedes the
"STOPPED at two dependency blockers" reconnaissance below it.

### 1. Isolated `olmo` conda environment (GPT-Neo env untouched)

- New env `olmo` (Python 3.11.15): **torch 2.5.1+cu121, transformers 4.49.0,
  numpy 2.4.6, scipy 1.17.1**, plus datasets / accelerate / safetensors /
  gdown / matplotlib / pytest. CUDA visible (RTX 4090). transformers 4.49.0
  chosen as a stable release that loads OLMo-2 (model card requires >= 4.48);
  not blindly-latest.
- The GPT-Neo baseline env is **confirmed unchanged**: torch 2.1.2+cu121,
  transformers 4.44.2, TRL 0.9.6. Phase 4 / GPT-Neo reproducibility intact.
- Binaries: `/root/miniconda3/envs/olmo/bin/{python,pip,gdown}`.

### 2. OLMo-2 load verification (tests/diag_olmo_load.py)

- **SFT (theta-star) = `allenai/OLMo-2-0425-1B-SFT`: fully verified.** Loads as
  `Olmo2ForCausalLM` (1.485B params, fp32), tokenizer round-trips, forward OK
  (logits vocab 100352), **peak VRAM 5.95 GB** (fp32 load + fwd; ample headroom
  on a 48 GB card). MLP at layer 12 = `['gate_proj','up_proj','down_proj',
  'act_fn']`; **down_proj.weight = (2048, 8192) = 16.78M params**, depth
  12/16 = 0.750. `get_submodule('model.layers.12.mlp.down_proj')` works. This
  confirms Section-1 phi exactly.
- **Base (reference) = `allenai/OLMo-2-0425-1B`: fully verified.** Both
  safetensors shards (4.98 GB + 0.96 GB, **5.95 GB fp32 total** — note: the base
  pretrained is stored fp32, so it is ~6 GB by itself, not part of a "4.8 GB
  total") are sha256-verified in the HF cache; `diag_olmo_load.py` now loads
  **both** models by id (`HF_HUB_OFFLINE=1`), forward OK, tokenizer round-trip
  OK, peak VRAM 5.95 GB → **"ALL OLMO-2 LOAD CHECKS PASSED"**. The shards were
  pulled by a parallel-chunk curl fetcher (tests/diag_olmo_fetch_base.py) with
  per-shard sha256 gating, then placed in the HF cache so `from_pretrained` loads
  by id. See "proxy note" for why the standard downloader could not be used and
  the one fetcher bug that the sha256 gate caught.

### 3. Lee toxicity_pairwise data — downloadable file (NOT regeneration)

- Fetched `toxicity_pairwise.zip` (ID `1BmBkhNS4R...` ) by file-id from the
  paper's Drive folder via gdown (the folder also holds `dpo.pt`, `probe.pt`,
  `intervene_data.zip` — skipped; only the pairwise set is needed). Zip
  integrity OK.
- Extracted to `data/lee_pairwise/toxicity_pairwise/`: **6 splits x 4096 lines
  = 24,576 pairwise records.** Each line has `prompt_text` (Wikitext-2 prompt),
  `unpert_gen_text` (non-toxic GPT-2 continuation), `pert_gen_text`
  (PPLM-perturbed **toxic** continuation), plus GPT-2 BPE `*_input_ids`. The
  `*_input_ids` are GPT-2 tokenization — for OLMo we re-tokenize from the
  **text** fields (the universal anchor), not the stored ids. This is a
  downloadable artifact; no PPLM regeneration needed.

### 4. numpy-2.x compatibility of the Phase 4 estimators (reuse check)

- Ran `tests/test_phase4_{variance_decomp,e2,e3}.py` under the olmo env. **Every
  numerical test passes under numpy 2.4.6 / torch 2.5.1 / transformers 4.49**
  (score-vs-finite-difference on the fp64 toy, split-half unbiasedness, IHVP
  orientation, Delta-VP, HVP, MINRES with its expected CG-breakdown warning).
  An isolated re-check (`tests/diag_olmo_estimator_compat.py`) confirms the
  headline U-statistic matches brute force at ~1e-16 and the jackknife point at
  ~1e-12 under numpy 2.x.
- **One coupling to flag (NOT a numpy problem):** the estimator modules
  transitively import `phase4.sampling -> phase1.score -> train_ppo -> trl`, and
  TRL is deliberately absent from the olmo env (Stage 1 SFT uses no PPO). So a
  bare `pytest` collection raises `ModuleNotFoundError: trl` on the E1/U-stat
  paths. This is a packaging coupling, fully decouplable by making the
  train_ppo/trl import lazy (or splitting the pure-numpy estimators out of the
  model-coupled module). **Left for Stage 1 itself; not patched now, not worked
  around silently** — surfaced here per the stop-and-report rule.

### Proxy note (operational, why the workarounds exist)

The machine has no direct internet — all egress goes through a proxy. Two are
available and were benchmarked: the default `127.0.0.1:7890` and AutoDL's
academic-acceleration proxy `10.37.1.23:12798` (from `source /etc/network_turbo`).
**They are empirically equal** (~0.45-0.5 MB/s aggregate over 4 connections; turbo
single-stream was actually slower at ~47 KB/s and timed out) — the upstream HF
bandwidth is just ~0.5-1 MB/s this session, so "turbo" is not a speed-up here.

Two failure modes hit during the base-model fetch, both routed around:
1. **HF python downloader hangs at 0 bytes** on large streamed shards through
   either proxy (no read-timeout fires). So shards were pulled with a
   parallel-chunk curl fetcher (8 byte-range connections, resume, sha256 gate).
2. **Silent range-corruption.** The proxy intermittently answers a Range request
   with a **200 (whole file from byte 0)** instead of a **206 (partial)**. The
   first fetcher version only checked accumulated *size*, so it appended those
   wrong bytes and poisoned the *head* of several chunks while still hitting the
   exact expected size → **right size, wrong sha256**. The sha256 gate caught it
   (it did its job — nothing corrupt entered the cache). Fixed by requiring curl
   to report **http 206** before appending an attempt; a 200/error body is
   discarded. Re-download then verified clean.

**Implication for Stage 2:** any large download here must be sha256/etag-verified,
not size-checked, and must reject non-206 range responses. The standard
`from_pretrained`/`hf download` path is unreliable through this proxy; use the
verified chunk fetcher (or hf_transfer if it proves to respect the proxy — not
installed/tested yet, and unlikely to beat the ~0.5-1 MB/s upstream cap anyway).

---

## Stage 1 — SFT quick test, broad toxicity direction (2026-06-04): VERDICT = DEATH-direction

**Positioning (read this first).** OLMo-2-1B-SFT is supervised — no reward, no
KL penalty — so it is NOT in the KL-RL framework and this stage is NOT a strict
A2 test. It is the heuristic probe the spec defines: does a closer-to-converged,
supervised policy have naturally smaller effective-reward variance than the PPO
model, under a *nominal* beta? The conclusion below is a directional clue, not a
final answer; the strictly comparable test is Stage 2 (OLMo-PPO).

Setup: theta* = OLMo-2-0425-1B-SFT, reference = OLMo-2-0425-1B (pretrained),
reward = RoBERTa nothate logit (unchanged), phi = layer-12 down_proj (16.78M
params), beta nominal 0.2365 (swept 0.1/0.2/0.5 for ICC), raw-completion
sampling on the identical seed-42 RTP prompt subset GPT-Neo Phase 1 used.

### The three measurements (always together, per the reading rules)

**(a) OLMo-SFT's actual toxicity — the confound is ABSENT.**
Mean nothate logit **+3.36** (higher = less toxic; quartiles −4.31 / 2.64 /
3.82 / 4.47 / 4.72), reward variance 2.04. The model is safety-tuned and
mostly non-toxic with a small toxic tail. So a high ICC here cannot be blamed
on "the model is just toxic" — the reward-driven confound the Section-3 note
worried about does not materialise. What IS large: the k1 KL to the reference
(mean **14.9**, variance **58.4**) — SFT drifted far from the pretrained
reference, and that drift varies a lot across responses.

**(b) Within-prompt variance decomposition — KL-dominated (the A2-relevant part).**

| beta | share reward | share KL | share cov | ICC |
|---|---|---|---|---|
| 0.1 | 0.719 | 0.228 | 0.053 | 0.805 |
| 0.2 | 0.414 | 0.524 | 0.062 | 0.828 |
| **0.2365 (nominal)** | **0.339** | **0.601** | **0.060** | **0.835** |
| 0.5 | 0.108 | 0.852 | 0.040 | 0.860 |

GPT-Neo baseline at its trained beta: shares 0.47 reward / 0.56 KL, ICC 0.892.
At the nominal beta OLMo-SFT's within-prompt variance is MORE KL-dominated than
GPT-Neo's. Per the reading rules, the KL part is the closer proxy for the A2
signal; it is the dominant part here.

**(c1) ICC — high, beta-insensitive, NOT meaningfully below GPT-Neo.**
ICC = 0.835 at nominal beta, range 0.805–0.860 across a 5x beta sweep. The
"life" condition (ICC clearly below 0.89, e.g. < 0.5) is NOT met.

**(c2) Thermometer — enormous, at the GPT-Neo kill's own evidentiary standard.**

| pool | g_f version | thermometer | two-half cos | U-stat z | naive/ustat |
|---|---|---|---|---|---|
| 400x32 (primary) | subtracted | **1031** | 0.464 | 4.30 | — |
| 400x32 (primary) | toxic-only | **1096** | 0.457 | 4.77 | — |
| 200x16 (consistency) | subtracted | 751 | 0.064 | 1.72 | 4.65 |
| 200x16 (consistency) | toxic-only | 738 | 0.057 | 1.82 | 5.26 |

GPT-Neo E3 (200x16): thermometers 34.1 / 12.5 / 4.2 / 11.8 with two-half cos
0.13–0.79 and z 3.7–4.2. The OLMo 200x16 pool was too noisy by that standard
(z ~1.7, cos 0.06), so the pool was scaled 4x; at 400x32 the estimate meets the
GPT-Neo bar (z 4.3–4.8, cos ~0.46) and the two pools agree within their noise.
g_f: 100 Lee pairs, ||g_f|| = 0.97 (subtracted) / 1.14 (toxic-only),
cos(mean-toxic, mean-nontoxic) = 0.556 — the baseline subtraction removes a
substantial shared "topic" component, and both versions give the same verdict.

### Verdict (per the agreed decision rules)

- Thermometer ~1000 >> 0.3 on both g_f versions, AND
- the within-prompt variance is KL-dominated (0.60 at nominal beta), AND
- the toxicity confound is absent (the model is non-toxic), AND
- ICC stays high (0.835 ~ GPT-Neo's 0.892).

=> **DEATH-direction: A2-failure is not PPO-specific.** Switching to a
supervised, better-converged 12x-larger model does not produce the clean-formula
regime — the effective-reward variance does not collapse and the curvature
contamination Delta dwarfs the eval direction, exactly as on GPT-Neo-PPO.
The special-case flag (thermometer large but all in the reward part) does NOT
apply — the source check points at KL, i.e. at the A2-relevant signal.

**Magnitude caveat (do not over-read ~1000 vs GPT-Neo's 4–34).** The OLMo
thermometer is 30–250x GPT-Neo's, but the two sit in different frameworks: the
PPO model actually optimised its KL-RL objective (it is "near" its own optimum
in a way SFT never tried to be), while OLMo-SFT's huge, variable reference-KL
(mean 14.9) makes it *structurally* far from the nominal-beta KL-RL optimum.
The honest cross-model statement is directional only: both are far above 0.3;
neither is anywhere near the clean regime. Per the spec, if a universality
conclusion is wanted at thesis level, Stage 2 (OLMo-PPO, same paradigm as
GPT-Neo) remains the clean test — that decision is the user's.

### Validation chain backing these numbers

1. Gated-SwiGLU toy: HVP / Delta-VP / per-prompt gates at 1e-9/1e-8 (plus the
   model-independent no-double-denom + MINRES gates) — 8/8.
2. Real-OLMo HVP finite-difference: vector-HVP rel err fp32 ~3e-3, fp64
   3.25e-4 (forward-only 1st-derivative check grounds s at 1.8e-4).
3. EK-FAC factors at the GPT-Neo token budget: 20,540 on-policy rollouts,
   601,418 response tokens (GPT-Neo: 20,832 / ~600k).
4. p_f-scale check: OLMo's F^-1 amplification (~8e3) is *milder* than
   GPT-Neo's (4–6e4 with the same damping and valid thermometers) — damping
   left identical for comparability.
5. The 26.8-GiB jackknife OOM on the 7x-larger phi was fixed by a streaming
   implementation gated bit-exact against the Phase-4 reference
   (tests/diag_streaming_stats_check.py), and the off-GPU accumulator path is
   gated bit-exact on the gated toy.

### Outputs

- `data/phase5/stage1_icc.json` (ICC + decomposition + toxicity level + examples)
- `data/phase5/stage1_thermometer_400x32.json` (primary), `stage1_thermometer.json` (200x16)
- `data/phase5/factors/` (A/S/Q_A/Q_S/Lambda + meta, ~1.3 GB)
- drivers: `src/run_phase5_stage1_{icc,factors,thermometer}.py`, `src/phase5/*`

### Deviations / not done

- Thermometer pool scaled 200x16 -> 400x32 beyond the spec's E3 default, because
  the OLMo 200x16 estimate (z ~1.7, two-half cos 0.06) was below the GPT-Neo
  kill's own evidentiary standard; 400x32 meets it. Both reported.
- beta sweep applied to ICC (as specified); the thermometer uses nominal beta
  only.
- No left-tail/contrib tracking in the Stage-1 thermometer (not in the Stage-1
  spec; available via track_contribs if wanted).
- No MINRES / corrected ranking (Stage-1 is a magnitude probe; ranking analyses
  belong to a hypothetical Stage-2/E3-style run).
- Stage 2 (OLMo-PPO) NOT started, per instruction.

---

## Section 0 + 1 reconnaissance (2026-06-02) — STOPPED at two dependency blockers
## (superseded by "External dependencies LANDED" above; kept for the record)

### Dependency A — OLMo models: available, structure known, but env too old to load

- **Downloadable (network OK, configs fetched):** `allenai/OLMo-2-0425-1B`
  (pretrained → reference policy) and `allenai/OLMo-2-0425-1B-SFT`
  (SFT → Stage-1 theta-star). Both real.
- **Architecture:** `Olmo2ForCausalLM`, model_type `olmo2`. 16 layers, hidden
  size 2048, intermediate size 8192, 16 heads, vocab 100352, untied embeddings,
  activation **silu** → confirms a **gated SwiGLU MLP** (gate_proj / up_proj /
  down_proj), NOT GPT-Neo's plain two-layer MLP.
- **BLOCKER:** the installed transformers is **4.44.2**; `Olmo2Config` /
  `Olmo2ForCausalLM` need **transformers >= 4.48**. The model cannot be
  instantiated in the current environment, so the spec's "load once + run a
  forward + check VRAM/tokenizer" sub-check is **not yet confirmed**. Upgrading
  transformers risks (a) TRL 0.9.6 used for Stage-2 PPO, and (b) the GPT-Neo
  baseline's exact reproducibility. This needs a human environment decision
  (isolated env for Stage 1 — which needs no TRL — vs. in-place upgrade +
  re-verify the GPT-Neo tests).

### Dependency B — Lee toxicity_pairwise data: format known, NOT local, source TBD

- **Format** (from `references/DPO-detoxify/.../pplm_dataset.py`): a directory of
  `*.jsonl`, each line a JSON object with `prompt_text`, `unpert_gen_text`
  (the non-toxic / unperturbed continuation) and a perturbed (toxic) continuation
  — PPLM-generated pairs over Wikitext-2 prompts. Plain text, so it will tokenize
  with any tokenizer (OLMo's included).
- **NOT present locally:** `references/DPO-detoxify/data/` holds only a
  `placeholder`; the `toxicity_pairwise/` directory the loader expects is absent.
- **Not on HuggingFace as such:** a dataset search returns `unalignment/toxic-dpo`
  variants, which are a *different* dataset, not Lee's PPLM pairwise set. It must
  be fetched from the spec's named GitHub source (`ajyl/dpo_toxic` /
  `Yushi-Y/dpo-toxic-neurons`) — **not yet confirmed** to host the data as a file
  (may be git-LFS, or may require re-running PPLM generation).

Per the spec ("either dependency unconfirmed → stop and find an alternative
together"), both are open. Stopped here.

---

## Section 1 — phi for OLMo-2-1B (doable from config; no model load needed)

OLMo-2's MLP is gated SwiGLU: `gate_proj` (2048→8192), `up_proj` (2048→8192),
`down_proj` (8192→2048), with `down_proj( silu(gate_proj(x)) * up_proj(x) )`.

**Proposed phi = `down_proj`**, matching the spec's recommendation: it is the
"back-to-hidden" final linear, the structural analog of GPT-Neo's `c_proj`, and a
plain linear layer — so EK-FAC's Kronecker factorization is clean on it (its input
is the gated activation vector; the gating nonlinearity sits *upstream* of
down_proj and does not break down_proj's own Kronecker structure). gate_proj /
up_proj are left out (they feed the gating multiply; messier for EK-FAC).

**Layer = index 12 of 16 (0-indexed)** = 0.75 relative depth, matching GPT-Neo's
layer 9 of 12 (0.75). Module path: `model.layers.12.mlp.down_proj`.

| | GPT-Neo-125M (baseline) | OLMo-2-1B (proposed) |
|---|---|---|
| total layers | 12 | 16 |
| hidden size | 768 | 2048 |
| MLP type | plain (c_fc → c_proj) | gated SwiGLU (gate/up/down, silu) |
| intermediate size | 3072 | 8192 |
| phi module | layer 9 `c_proj` | layer 12 `down_proj` |
| relative depth | 9/12 = 0.75 | 12/16 = 0.75 |
| phi weight shape (out, in) | (768, 3072) | (2048, 8192) |
| **phi params** | **2.36M** | **16.78M** (7.1×) |
| EK-FAC A block (in × in) | 3072² ≈ 75 MB | 8192² ≈ 536 MB |
| EK-FAC S block (out × out) | 768² ≈ 4.7 MB | 2048² ≈ 33.5 MB |

Relative depth matches; magnitude is ~7× larger but well within single-GPU for
the Stage-1 IF analysis (factors ~1.3 GB; model + reference + reward + a
double-backward graph fit comfortably in 48 GB). **Flag for Stage 2:** a naive
per-rollout score cache like GPT-Neo's would be ~1.4 TB here (20.8k rollouts ×
67 MB) — Stage 2 will need on-the-fly scoring or a compressed cache, not the
184 GB-style dump.

A code note (not "just model loading"): EK-FAC's `layer_name` and the c_proj
forward/backward hooks (`src/ekfac/hooks.py`) are named for `c_proj`; they need
generalizing to `down_proj` / the OLMo module path. Small, but real.

---

## Section 3 — conceptual concern (raised per the spec's invitation)

The spec already concedes Stage 1 is not a strict A2 test, reframing it as "does a
supervised, closer-to-optimal policy have naturally smaller effective-reward
variance than PPO." I agree with that framing and add one specific confound:

**GPT-Neo's theta-star is a detox-trained model; OLMo-2-1B-SFT is not.** The
effective reward is (RoBERTa non-toxic score) minus beta times (KL from the
reference). Its within-prompt variance therefore has a reward-driven part and a
KL-driven part. GPT-Neo step_0650 was trained to be non-toxic, so its responses
are consistently non-toxic and the reward part is comparatively tame (E1 still put
it at ~47% of within-variance). OLMo-2-1B-SFT was trained for general instruction
following, not detox; on toxic prompts its responses may range widely in toxicity,
inflating the reward part for a reason that has nothing to do with A2 or the
method. So a high ICC (or large thermometer) on OLMo-SFT could reflect "it is not a
detox model" rather than "A2-failure is universal"; a low one could reflect "it
happens to be safety-tuned" rather than "A2 holds."

**This does not make Stage 1 meaningless — it makes ICC alone insufficient.** The
fix is cheap and already built (Phase 4 E1): alongside ICC, always report (a)
OLMo-SFT's actual toxicity level (mean reward and its variance) and (b) the
reward/KL/covariance decomposition of within-prompt variance. Read the KL-term
share and the thermometer as the more method-relevant signals; treat the
reward-term share as the detox-ness confound. And keep Stage 2 (PPO on OLMo, same
paradigm as GPT-Neo) as the actually-clean test of "PPO's fault vs the method's."
Stage 1 is a cheap directional probe whose verdict must be read through the
decomposition, not the single ICC number.

I do **not** think the premise is meaningless, so I did not hard-stop on it — but
flagging it so the decomposition is in Stage 1's plan from the start.
