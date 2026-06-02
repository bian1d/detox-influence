# Phase 5 — OLMo-2-1B: A2-failure, PPO's fault or the method's?

Companion baseline: `reports/phase4_gptneo_negative_result.md` (GPT-Neo kill).
Plain-language throughout (no math rendering).

---

## Section 0 + 1 reconnaissance (2026-06-02) — STOPPED at two dependency blockers

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
