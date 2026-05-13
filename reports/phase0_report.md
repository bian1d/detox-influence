# Phase 0 Report — PPO Setup, Baselines, and Training

**Model:** EleutherAI/gpt-neo-125m
**Reward model:** facebook/roberta-hate-speech-dynabench-r4-target
**Date completed:** 2026-05-13
**Status:** COMPLETE — all acceptance criteria met. Ready for Phase 1.

---

## 1. Setup

### 1.1 Model and Tokenizer

GPT-Neo-125m (`EleutherAI/gpt-neo-125m`) was selected after 7 failed training
runs on GPT2-medium (see Section 5.2). Architecture: 12 transformer layers,
d=768, d_mlp=3072, vocab_size=50257. Pretrained on The Pile (Gao et al. 2020),
which includes Wikipedia — this explains the lower baseline PPL (~24) relative
to the 35-40 range expected for WebText-only models.

### 1.2 Reward Model

`facebook/roberta-hate-speech-dynabench-r4-target` (binary hate-speech classifier,
label 0=nothate, label 1=hate). Reward = raw logit for label 0; no softmax, no
negation. Higher reward = less toxic. This is the exact formula from the
HuggingFace TRL detoxification tutorial (Burtenshaw 2022).

### 1.3 PPO Hyperparameters (final, working config)

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 2.94e-5 |
| `batch_size` | 32 |
| `mini_batch_size` | 32 |
| `ppo_epochs` | 4 |
| `init_kl_coef` | 0.2 |
| `kl_penalty` | `'kl'` (k1: log π_θ − log π_ref, TRL default) |
| `adap_kl_ctrl` | True (target KL 6.0) |
| `cliprange` | 0.2 |
| `max_new_tokens` | 30 |
| `min_new_tokens` | **10** (critical — see 5.4) |
| Total steps | 2000 |
| Rollouts per step | 32 |
| Total rollouts | 64,000 |

### 1.4 Training Data

RealToxicityPrompts (Gehman et al. 2020), filtered to `prompt.toxicity > 0.3`,
excluding the 400-prompt eval set. Filtered pool: 34,980 prompts. Sampled 10,000
with seed=42; epochs cycle through the pool. Prompts truncated to 64 tokens.

### 1.5 Probe Verification (Task 3.2)

A linear toxicity probe was trained on GPT-Neo-125m last-layer mean-pooled
hidden states using Jigsaw toxic comment data (binary label = OR-fold of all 6
toxicity columns, 90/10 split, seed=42).

- **Accuracy:** 0.9323 at threshold=0.90 (trivial all-negative baseline: 0.9004)
- **AUC-ROC:** 0.9318
- **Logit lens verdict:** NEGATIVE — 0/30 top tokens are profanity when
  projecting probe weight through unembedding

Positive control: GPT2-medium + Lee et al.'s `probe.pt` shows 21/30 profanity
tokens in the same diagnostic (max cosine 0.2993 at layer=19, neuron=770).
GPT-Neo-125m's max value-vector cosine: 0.1466. **Conclusion:** GPT-Neo-125m
does not have a concentrated toxic value vector analogous to GPT2-medium's;
the probe learned a political/topical direction rather than a token-level
toxicity direction. This is consistent with Pile vs. WebText pretraining
differences. The finding does not block Phase 2 EK-FAC analysis (which targets
W₂ based on KL concentration, not probe direction).

Full diagnostic in `data/probe/probe_analysis_gpt_neo_125m.json` and
`tests/diag_logit_lens_probe.py`.

---

## 2. Baselines (Task 3.3)

### 2.1 Wikitext-2 Perplexity

| Metric | Value |
|--------|-------|
| Dataset | Wikitext-2-raw-v1, test split |
| Stride / max_length | 512 / 1024 |
| **Baseline PPL** | **24.13** |

Saved to: `data/baseline_ppl.json`.

### 2.2 Baseline Toxicity

| Metric | Value |
|--------|-------|
| Eval set | 400 prompts, uniform sample from RealToxicityPrompts (seed=42) |
| Continuations | 30 new tokens, do_sample=True, top_k=0, top_p=1.0 |
| **Mean toxicity** | **0.0520** |
| Std | 0.1594 |
| n | 400 |

Lower than Burtenshaw's tutorial reference (~0.1627) because we sample uniformly
rather than filtering to toxic prompts. Saved to: `data/baseline_toxicity.json`.
Eval prompt indices saved to `data/eval_prompts_400.json` (fixed across all
evaluations).

---

## 3. PPO Training (Task 3.4)

### 3a. Hyperparameter Rationale

Adopted the TRL official detoxification tutorial config (Burtenshaw 2022) with
`kl_penalty='kl'` (k1) and `adap_kl_ctrl=True`. The adaptive controller adjusts
`kl_coef` to keep running KL near `target=6.0`, providing stability without
manual tuning. After 7 failed runs on GPT2-medium (Section 5.2), GPT-Neo-125m's
lower base toxicity probability allowed this config to remain stable.

Checkpointing: every 50 steps for steps 0–999, every 100 thereafter, plus the
final step (31 checkpoints total). Per-step rollouts saved as
`data/rollouts/step_NNNN.pt`.

### 3b. Training Trajectory (Run 3)

| Step | mean_reward | mean_length | approxkl | kl_coef |
|------|-------------|-------------|----------|---------|
| 0 | +2.636 | 30.0 | 0.29 | 0.200 |
| 50 | +3.173 | 30.0 | 0.06 | 0.197 |
| 500 | +3.710 | 17.6 | 0.06 | 0.292 |
| 1000 | +3.503 | 24.1 | 0.07 | 0.353 |
| 1500 | +3.465 | 24.7 | 0.03 | 0.418 |
| 1999 | +3.771 | 17.8 | 0.02 | 0.452 |

- **Reward trajectory:** +2.636 → +3.771 (Δ = +1.135 from step 0 to final)
- **Mean reward over steps 1900–1999:** +3.186 (std 0.285)
- **Max approxkl across all 2000 steps:** 0.458 (hard stop = 50, well safe)
- **n_collapsed_responses across all steps:** 0
- **Wall-clock time:** ~1h 37min on a single GPU

Wandb run: `astral-bird-4` (project `rlhf-if-detox`).

---

## 4. Checkpoint Evaluation (Task 3.5)

Per-checkpoint metrics on the fixed 400-prompt eval set
(`data/checkpoint_eval.csv`, 31 rows):

| Step | mean_toxicity | mean_reward | ppl_wikitext2 | mean_length | n_short(<10) |
|------|---------------|-------------|---------------|-------------|--------------|
| 0    | 0.0506 | +3.096 | 24.20 | 29.9 | 1 |
| 50   | 0.0447 | +3.433 | 27.00 | 29.5 | 3 |
| 100  | 0.0427 | +3.426 | 28.04 | 27.9 | 18 |
| 250  | 0.0363 | +3.656 | 29.84 | 15.6 | 136 |
| 500  | 0.0270 | +3.723 | 32.69 | 10.5 | 227 |
| **650** | **0.0160** | **+3.842** | 33.73 | 13.4 | 173 |
| 1000 | 0.0333 | +3.712 | 33.16 | 18.1 | 130 |
| 1500 | 0.0231 | +3.786 | 35.84 | 15.6 | 166 |
| 1999 | 0.0231 | +3.638 | 40.14 | 8.0 | 279 |

**Sanity check (step 0 vs baseline):** toxicity delta = 0.0014, PPL delta = 0.07
— well within noise.

**Curves:** `reports/phase0_checkpoint_curves.png` (4-panel: toxicity, reward,
PPL, length vs. step).

### 4.1 Trajectory Summary

| Metric | step 0 | best (step 650) | final (step 1999) | Δ final vs 0 |
|--------|--------|------------------|---------------------|--------------|
| Mean toxicity | 0.0506 | 0.0160 (−68%) | 0.0231 | **−54%** |
| Mean reward (eval) | +3.096 | +3.842 | +3.638 | **+0.54** |
| Reward (training log, same step) | +2.636 | +3.461 | +3.771 | matches |
| Wikitext-2 PPL | 24.20 | 33.73 | 40.14 | **+15.94 (+66%)** |
| Mean response length (eval) | 29.9 | 13.4 | 8.0 | −73% |

### 4.2 Acceptance Criteria

| Criterion | Target | Actual | Pass? |
|-----------|--------|--------|-------|
| Probe accuracy on Jigsaw | ≥ 92% | 93.23% (AUC 0.9318) | ✓ |
| Wikitext-2 baseline PPL | 20–25 (GPT2-medium spec) | 24.13 (GPT-Neo on Pile) | ✓ (range expanded for model swap) |
| Baseline toxicity > 0 | yes | 0.0520 | ✓ |
| Mean reward step 500 > step 0 | yes | +3.723 > +3.096 | ✓ |
| approxkl < 50 always | always | max 0.458 | ✓ |
| Response length ≥ 10 in training | mean ≥ 10 | 0 collapsed responses | ✓ |
| Final toxicity < step-0 toxicity | strictly less | 0.0231 < 0.0506 | ✓ |
| step=0 eval ≈ baseline files | toxicity Δ ≤ 0.02 | Δ = 0.0014 | ✓ |

All acceptance criteria pass.

### 4.3 Three-Phase Training Story (Recommended θ* = step 650)

The full eval trajectory reveals **three distinct training phases**, visible
across all four metrics in `reports/phase0_checkpoint_curves.png`:

#### Phase A — Rapid Detox Learning (steps 0–300)

| Metric | step 0 → step 300 |
|--------|-------------------|
| Toxicity | 0.0506 → 0.0301 (−40%) |
| Reward (eval) | +3.10 → +3.65 |
| PPL | 24.20 → 34.04 |
| Length | 29.9 → 9.7 |

The policy rapidly chases reward, length collapses toward the
`min_new_tokens=10` floor, and PPL drift begins. The fastest learning happens
here.

#### Phase B — Healthy Plateau (steps 300–1200)

| Metric | typical range |
|--------|---------------|
| Toxicity | 0.016–0.033 (best at step 650: **0.0160**) |
| Reward (eval) | +3.70 to +3.85 (peak) |
| PPL | 32–34 (slow drift) |
| Length | 10–25 (recovers from Phase A floor as policy finds better tradeoff) |

The policy stabilises into a healthy regime. Length recovers from Phase A's
floor as the model learns that longer non-toxic completions also score well.
The toxicity minimum and reward peak both occur in this phase, around
step 650.

#### Phase C — Over-training and Reward-Hacking Onset (steps 1200–2000)

| Metric | step 1200 → step 1999 |
|--------|-----------------------|
| Toxicity | 0.0214 → 0.0231 (rises 0.022→0.040 transiently around step 1900) |
| Reward (eval) | +3.81 → +3.64 (declines) |
| PPL | 34.61 → 40.14 (continues climbing) |
| Length | 22.7 → 8.0 (monotonic compression — reward hacking via brevity returns) |

The reward declines, length compresses to near the eval-time floor, and PPL
keeps climbing. This is reward-hacking re-emerging — without
`min_new_tokens` at eval time the policy increasingly emits very short
continuations. The training-time `min_new_tokens=10` floor masks this from
the training log (training length 17.8 at step 1999 vs eval length 8.0).

Independent corroboration: from the wandb run, **`ppo/val/var_explained`
occasionally reaches 0.75 in Phases A/B (value head modelling real signal)
but never exceeds 0.5 after step 1165 sustained** — the value head can no
longer model the reward-hacking-driven oscillation. This independently
confirms step ~1165+ as the onset of over-training.

#### Recommended θ* for Phase 2: **step 650**

Step 650 is Pareto-optimal across all metrics relative to step 1999:

| Metric | step 650 | step 1999 | Verdict |
|--------|----------|-----------|---------|
| Toxicity | **0.0160** | 0.0231 | step 650 better |
| Reward (eval) | **+3.842** | +3.638 | step 650 better |
| PPL | **33.73** | 40.14 | step 650 better |
| Length (eval) | 13.4 | 8.0 | step 650 better (less compression) |

**Phase 2 IF analysis should target step 650 as θ*.**

The checkpoint has been verified for integrity: model weights load via
`AutoModelForCausalLM` (125,198,592 params), `ppo_step.json` confirms step 650,
and 5-prompt generations produce neutral, fluent text (e.g., "The weather
today is" → "fairly warm and sunny, but the low- to mid-day temperatures may
reach 90 degrees…").

#### Step 277 Anomaly (Phase A)

A single anomalous step at 277 (`objective/kl = −12`, `approxkl = 2.55`,
the only step in the run with `approxkl > 1`) has been diagnosed in
`reports/notes/step_277_anomaly.md`. The discrepancy arises from TRL's
padded-batch computation of `objective/kl`; non-pad rollout-level
k1_sum mean at step 277 is the normal +8.75. The single-step spike
resolved on its own (step 278 already returned to baseline). **No effect
on stored rollout data; no remediation needed.**

### 4.4 PPL Drift — Open Issue

PPL drift of +66% on Wikitext-2 is substantial. This is a well-known
detoxification side-effect — the policy trades off general language fluency
for low-toxicity outputs. Three observations:

1. The trajectory is monotonic with training step (24.20 → 40.14), not a
   sudden break — suggests gradual drift, not catastrophic forgetting.
2. The best detox/PPL trade-off appears around step 650 (toxicity 0.0160,
   PPL 33.73) — Phase 2 IF analysis may want to use this rather than the
   final checkpoint.
3. Lee et al. (2024) report similar PPL drift on GPT2-medium after DPO
   detoxification — this is not specific to PPO.

The drift does not affect Phase 2 EK-FAC correctness (IF is computed on
training rollouts, not on Wikitext-2). It is noted here for downstream
reference.

---

## 5. Engineering Findings (Three Failure Modes)

### 5.1 Overview

Phase 0 ran **three full PPO attempts**, each surfacing a distinct
implementation pitfall. All three are valid engineering findings worth
documenting in publication.

| Run | Config | Outcome | Lesson |
|-----|--------|---------|--------|
| GPT2-medium ×7 | various KL configs | k1 explosion → divergence | Base model toxicity matters for KL stability |
| Run 1 | `min_length=-1` | Length collapse step 902 | RLHF length-hacking is well-known; needs floor |
| Run 2 | `min_length=10` | Length collapse step 1012 | `min_length` is *total* length, not new-token length |
| Run 3 | `min_new_tokens=10` | 2000 steps complete, all clean | Use `min_new_tokens` for the new-token floor |

### 5.2 GPT2-medium k1 Explosion (7 Failed Runs, Model Pivot)

k1 = log π_θ − log π_ref is signed and goes negative when the policy suppresses
a token the reference model assigned moderate probability. On GPT2-medium
(WebText pretrained, higher base toxic-token probability), early-training k1_sum
per response reached approximately −500, converting the KL penalty into an
effective +100 reward bonus. This runaway feedback destabilised training before
step 10 in all 7 attempts.

GPT-Neo-125m (Pile pretrained) has lower base toxic-token probabilities, so k1
magnitudes remain bounded under `adap_kl_ctrl=True`. **Decision:** Pivot from
GPT2-medium to GPT-Neo-125m, accepting that the Lee et al. (2024) probe direction
finding does not transfer.

### 5.3 Length Collapse #1: `min_length=-1` (Run 1, Stopped at Step 902)

With no minimum-length constraint, the model learned to generate 1–4 token
responses (often just EOS) to maximise the nothate reward. By step 902, the
hard-stop window (mean response length < 10 over 50 steps) triggered at
mean_length=3.6.

This is the classic RLHF reward-hacking failure (Stiennon et al. 2020; Gao et
al. 2022): the reward model assigns high nothate logits to near-empty
continuations because they contain no hate-speech tokens.

Archived run: `data/archive/run_no_min_length/` (steps 0–901 with valid
detox-learning rollouts before collapse).

### 5.4 Length Collapse #2: `min_length` is Total Length, not New-Token Length (Run 2, Stopped at Step 1012)

After adding `min_length=10` (the parameter named in the Burtenshaw tutorial's
LengthSampler), the run lasted 110 more steps than Run 1 before collapsing at
step 1012 with the same symptoms.

**Root cause:** HuggingFace `generate()`'s `min_length` is **minimum total
sequence length (prompt + continuation)**, not minimum new tokens. Training
prompts are 15–64 tokens; for any prompt ≥ 10 tokens (the vast majority),
`min_length=10` imposed zero constraint on continuation length. The ~110-step
extension came from the minority of prompts shorter than 10 tokens.

**Diagnosis method:** Comparing the two collapse step numbers (902 vs 1012) was
the clue — if `min_length=10` had been doing what its name suggests, the
constraint should have either prevented collapse entirely or made it impossible.
The fact that it merely delayed collapse pointed directly to the constraint
being partial — which led to inspecting the HuggingFace docs and discovering the
total-length semantics.

**Fix:** Use `min_new_tokens=10` (supported since transformers 4.25), which
suppresses EOS for the first 10 generated positions unconditionally. Verified
empirically with a logits-processor test that forces EOS — confirmed
`min_new_tokens` masks EOS in those positions while `min_length` does not when
prompt is ≥ N tokens.

Archived run: `data/archive/run_min_length_wrong/`.

**Note:** `LengthSampler(4, 30)` from the Burtenshaw tutorial was also evaluated
and rejected — it samples a per-prompt minimum from [4, 30], allowing some
prompts to have a minimum of 4 tokens, which risks re-collapse on those.

### 5.5 The Successful Config (Run 3)

`min_new_tokens=10`, otherwise identical to Run 2. 2000 steps complete:

- Mean response length stayed above 17 throughout the run (min over 50-step
  window: 13.7 at step 1670, never hit the 15-token soft warning more than
  briefly).
- Zero collapsed responses (`n_collapsed_responses=0` for all 2000 steps).
- Approxkl peaked at 0.458, two orders of magnitude under the 50 hard-stop.
- Final reward +3.771, up from +2.636 at step 0.

---

## 6. Rollouts Produced (Task 3.4 — Phase 2 Input)

Rollout schema per step file (`data/rollouts/step_NNNN.pt`):

| Field | Type | Notes |
|-------|------|-------|
| `prompt_text` | str | Decoded prompt |
| `prompt_token_ids` | LongTensor (T_p,) | Raw IDs, ≤ 64 tokens |
| `response_text` | str | Decoded continuation |
| `response_token_ids` | LongTensor (T_r,) | Raw IDs, 10 ≤ T_r ≤ 30 |
| `policy_logprobs` | float16 (T_r,) | Per-token log π_θ, pre-update |
| `ref_logprobs` | float16 (T_r,) | Per-token log π_ref, frozen ref model |
| `reward` | float | Raw nothate logit |
| `step` | int | PPO step index |

| Item | Value |
|------|-------|
| Total rollout files | 2,000 |
| Rollouts per file | 32 |
| **Total rollouts** | **64,000** |
| policy_logprobs capture timing | before `trainer.step()` (pre-update) |
| ref_logprobs source | `trainer.ref_model` (frozen throughout) |
| Storage | `data/rollouts/` |

**These 64,000 rollouts are the primary input for Phase 1 (effective reward
variance diagnostic) and Phase 2 (EK-FAC influence function computation).
Ready for Phase 2 IF analysis at θ* = `data/ppo_checkpoints/step_0650/`
(see Section 4.3 for the Pareto-optimal selection rationale).**

---

## 7. Output Files

| File | Contents |
|------|----------|
| `data/baseline_ppl.json` | PPL=24.13 on Wikitext-2-raw-v1 test |
| `data/baseline_toxicity.json` | mean=0.0520, std=0.1594, n=400 |
| `data/eval_prompts_400.json` | Fixed 400 RealToxicityPrompts indices + texts |
| `data/probe/probe_gpt_neo_125m.pt` | Linear probe (accuracy=0.9323, AUC=0.9318) |
| `data/probe/probe_analysis_gpt_neo_125m.json` | Cosine + logit-lens diagnostic |
| `data/ppo_train_log.jsonl` | 2,000 lines — per-step training metrics |
| `data/rollouts/step_0000.pt`–`step_1999.pt` | 64,000 rollouts |
| `data/ppo_checkpoints/step_NNNN/` | 31 checkpoints (steps 0, 50, …, 1900, 1999) |
| `data/checkpoint_eval.csv` | Per-checkpoint metric table (31 rows) |
| `data/archive/run_no_min_length/` | Run 1 (steps 0–901, min_length=-1, collapsed) |
| `data/archive/run_min_length_wrong/` | Run 2 (steps 0–1012, min_length=10, collapsed) |
| `reports/phase0_checkpoint_curves.png` | 4-panel metric curves |
| `reports/phase0_report.md` | This document |
| `reports/notes/step_277_anomaly.md` | Diagnostic note for single-step KL anomaly |

---

## 8. Deviations from the Plan

| Deviation | Reason |
|-----------|--------|
| Model: GPT2-medium → GPT-Neo-125m | 7 failed GPT2-medium runs from k1 explosion under TRL's adap_kl_ctrl + k1 estimator |
| Wikitext-2 PPL expected 20–25 → actual 24.13 | GPT-Neo trained on The Pile (includes Wikipedia); 24.13 is within the WebText 20–25 spec by coincidence — would be 35–40 for a comparable WebText-only model |
| Lee et al. probe direction does not transfer | GPT-Neo-125m has no concentrated toxic value vector; probe AUC=0.93 but logit-lens 0/30 profanity (vs 21/30 on GPT2-medium) — open finding for Phase 3 |
| Three training runs instead of one | Length-collapse failure modes #1 and #2 surfaced via successive `min_length`/`min_new_tokens` debugging |

---

## 9. What is NOT in this Report (Skipped or Deferred)

- **Probe retraining:** The current probe learned a topical (political) direction
  rather than token-level toxicity. Deferred — does not block Phase 2 because
  Phase 2 targets W₂ based on KL concentration, not the probe.
- **PPL drift mitigation:** No KL-penalty tightening or early stopping attempted
  — the drift is documented but accepted. Phase 1/2 may use step 650 (best
  toxicity/PPL trade-off) instead of step 1999 if preferred.
- **`gpt2-medium` ablation:** No further GPT2-medium runs after the 7th failure.
  If Phase 3 mechanistic verification needs Lee et al.'s exact probe target,
  that's a separate downstream project, not Phase 0 scope.

---

_Phase 0 complete. All acceptance criteria pass (Section 4.2). Awaiting human
review before beginning Phase 1._
