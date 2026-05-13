# Phase 0 — Setup, Probe Verification, and PPO Detoxification

## 0. Read first

Before doing anything, read:

- `CLAUDE.md` (collaboration rules, repository layout, locked-in
  decisions, reference materials)
- `CONTEXT.md` (theoretical background; at minimum sections 1, 6, 12, 14)
- `docs/detoxify-trl-official.md` (TRL official detox tutorial)

Do not skip these. Several decisions in this phase will look arbitrary
without that context.

## 1. Goal

Produce four deliverables:

1. A verified copy of Lee et al.'s toxicity probe (`probe.pt`),
   confirmed to reproduce key results from the paper.
2. Pre-training sanity baselines for GPT2-medium: Wikitext-2 PPL and
   mean toxicity on a fixed RealToxicityPrompts subset. These are the
   "step 0" reference numbers that all later checkpoints are compared
   against.
3. A trained PPO-detoxified GPT2-medium, saved as a dense series of
   checkpoints during training.
4. A per-checkpoint evaluation log — including the untrained base
   model as step 0 — that lets us pick which checkpoint to analyze in
   later phases.

This phase produces no influence-function code. It is pure setup.

## 2. Inputs

| Item | Source | Expected at |
|------|--------|-------------|
| Lee et al. probe | already in repo (copy from `references/DPO-detoxify/checkpoints/probe.pt`) | `data/probe/probe.pt` |
| GPT2-medium | HuggingFace `gpt2-medium` | downloaded by `transformers` |
| Reward model | `facebook/roberta-hate-speech-dynabench-r4-target` | downloaded by `transformers` |
| Jigsaw toxic comment dataset | Kaggle `jigsaw-toxic-comment-classification-challenge` | `data/jigsaw/` |
| RealToxicityPrompts | HuggingFace `allenai/real-toxicity-prompts` | downloaded by `datasets` |
| TRL detox reference | `docs/detoxify-trl-official.md` | for reference only |

## 3. Tasks

### Task 3.1 — Environment check

The user already has a working base environment with **Python 3.10**
and **PyTorch 2.1.0**. Do **not** create a new conda env. Verify the
existing env, then install only the additional packages needed:

- `trl>=0.8` (PPO trainer)
- `transformers>=4.40`
- `datasets`
- `evaluate` (for toxicity measurement)
- `pytest`
- `wandb`

Pin exact versions in `requirements.txt`. Record the versions in
`reports/phase0_report.md`.

**wandb setup:**

- Entity: `benzyl`
- The user has an API key but it is **not** included in this document.
  Run `wandb login` interactively, or set the `WANDB_API_KEY`
  environment variable in your shell, before starting PPO training.
- Do **not** embed the API key in any source file, config, or commit.
- In code, set the entity via env var (most robust across TRL versions):

  ```python
  import os
  os.environ["WANDB_ENTITY"] = "benzyl"
  os.environ["WANDB_PROJECT"] = "rlhf-if-detox"
  ```

### Task 3.2 — Probe verification (do this BEFORE PPO training)

Write `src/verify_probe.py` that reproduces Lee et al.'s probe
classification result on the Jigsaw validation split.

**Before writing your own code**, examine the reference implementation
in `references/DPO-detoxify/` — particularly
`toxicity/eval_interventions/eval_utils.py` and any related files that
show how Lee et al. load `probe.pt`, forward through GPT2-medium, and
classify Jigsaw comments. **Reimplement** the logic under `src/` (do
not import from `references/`). The reference is for understanding
detail-level decisions like which hidden state index to read, how
tokenization is handled, and how the probe vector is applied. If
your accuracy falls below 92%, diff your implementation against the
reference to find where you diverged.

**What the probe is, mathematically.**

Lee et al.'s `probe.pt` stores a **single direction vector** in
GPT2-medium's residual stream space:

$$W_\text{toxic} \in \mathbb{R}^{d} = \mathbb{R}^{1024}$$

It is **not** a (2, 1024) classifier weight matrix — binary
classification only needs one direction (the toxic vs non-toxic axis).
The classification decision is:

$$P(\text{toxic} \mid \bar{x}^{L-1}) = \sigma(W_\text{toxic}^\top \bar{x}^{L-1})$$

where $\bar{x}^{L-1}$ is the **average across all token positions** of
the residual stream at the last transformer layer.

**Procedure:**

1. Copy `references/DPO-detoxify/checkpoints/probe.pt` to
   `data/probe/probe.pt`. Verify shape is `(1024,)`.
2. Load Jigsaw, take the held-out 10% validation split (~56k of the
   561,808 comments).
3. For each comment, compute the sentence-level representation:
   - Tokenize with the GPT2 tokenizer.
   - Forward through GPT2-medium with `output_hidden_states=True`.
   - In HuggingFace GPT2, `outputs.hidden_states` is a **tuple of
     length 25** (1 embedding + 24 transformer layer outputs).
     `hidden_states[-1]` is the output of the **last transformer
     block** (layer index 23 in 0-indexed terms), which is exactly
     Lee et al.'s $x^{L-1}$.
   - Take the **mean across all token positions** (NOT only the last
     token, NOT a [CLS]-style position):
     `x_bar = hidden_states[-1].mean(dim=time_dim)` → shape `(1024,)`.
4. Compute `score = W_toxic @ x_bar` (a single scalar per comment),
   threshold at 0 (or apply sigmoid + threshold at 0.5; both are
   equivalent for argmax decisions).
5. Compare predictions to ground-truth labels and report accuracy.
6. Find the top-128 MLP value vectors by cosine similarity with
   $W_\text{toxic}$ across all (layer, index) pairs. Print the top 10.
   Lee et al.'s Table 1 lists `(19, 770)`, `(12, 771)`, `(18, 2669)`,
   `(13, 668)`, `(16, 255)`, `(12, 882)`, `(19, 1438)` among them.
7. Project the top 5 vectors onto vocabulary space (`E @ v`, where
   `E` is the token embedding matrix) and print their top-10 promoted
   tokens. Should be predominantly profanity.

**Acceptance criteria:**

- Probe accuracy on Jigsaw validation: **≥ 92%** (Lee et al. report
  94%; ±2% for any train/val split or tokenization differences).
- At least **5 of the top 10** layer-index pairs match Lee et al.'s
  Table 1.
- Top tokens for `v^{19}_{770}` include at least 3 of:
  `sh*t, a**, cr*p, f*ck, c*nt, garbage, trash`.

**If accuracy is significantly below 92%, the FIRST suspect is which
hidden state index is being read.** See failure mode 1 in section 6
before assuming the probe file itself is wrong.

If any criterion fails, stop and surface the problem. The probe is the
foundation of the entire project; do not proceed.

### Task 3.3 — Pre-training sanity baselines (do this BEFORE PPO)

Two pre-training measurements that establish the "step 0" reference
that every later checkpoint comparison depends on.

#### Sub-task 3.3.1 — Wikitext-2 PPL on base GPT2-medium

Write `src/eval_ppl.py` that computes perplexity of unmodified
GPT2-medium on the Wikitext-2 test split. Use the standard
stride-based sliding-window evaluation
(HuggingFace's reference recipe, `stride=512`, `max_length=1024`).

**Acceptance criterion:** PPL falls in the range **20–25** (literature
reports ~22 for GPT2-medium on Wikitext-2 test).

If your number is wildly outside this range, **the bug is in the PPL
code, not the model**. Fix the code before continuing — every later
checkpoint's PPL is compared against this number, so it has to be
right.

Write the result to `data/baseline_ppl.json` as
`{"ppl_wikitext2_test": float}`.

#### Sub-task 3.3.2 — Baseline toxicity on a fixed RealToxicityPrompts subset

Write `src/eval_toxicity.py` that:

1. Selects a **fixed subset of 400 prompts** from RealToxicityPrompts
   (~99.4k total, single split, no filtering by `prompt.toxicity`).
   Use `seed=42` and sample uniformly. This same subset is reused for
   every checkpoint in Task 3.5. Save the indices to
   `data/eval_prompts_400.json` so the subset is exactly reproducible.
2. Generates 30 new tokens per prompt with the base GPT2-medium
   (`do_sample=True`, `top_k=0`, `top_p=1.0`, `seed=42`,
   `max_new_tokens=30`).
3. Scores **the generated continuation only — NOT prompt + continuation**
   using `evaluate.load("toxicity", module_type="measurement")`. This
   uses the same underlying RoBERTa model as the reward model and
   returns the softmax probability of the "hate" label, ∈ [0, 1].
4. Reports mean and std of toxicity over the 400 samples.

Write the result to `data/baseline_toxicity.json` as
`{"mean": float, "std": float, "n": 400}`.

**Acceptance criterion:** the file is written. There is no fixed
target — this number IS the baseline. It should be greater than zero;
typical baseline mean for GPT2-medium on uniformly-sampled
RealToxicityPrompts is roughly 0.05–0.15.

### Task 3.4 — PPO detoxification training

Follow the local TRL detox doc with the configuration below.

**Hyperparameters:**

```python
from trl import PPOConfig

config = PPOConfig(
    model_name="gpt2-medium",
    learning_rate=1.4e-5,        # TRL detox tutorial default
    batch_size=64,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    init_kl_coef=0.2,
    target_kl=6,
    cliprange=0.2,
    cliprange_value=0.2,
    seed=42,
    log_with="wandb",
)
```

**Generation kwargs:**

```python
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "max_new_tokens": 30,
}
```

**Reward function (CRITICAL — get this exactly right):**

The reward model `facebook/roberta-hate-speech-dynabench-r4-target` is
a binary classifier:
- Label index `0` → `nothate` (neutral)
- Label index `1` → `hate`

Per the TRL official tutorial, the reward is the **raw logit of the
nothate label**, used directly with **no softmax and no negation**:

```python
logits = toxicity_model(**toxicity_inputs).logits.float()
rewards = logits[:, 0].tolist()   # nothate logit, used directly
```

Higher reward = more neutral generation = less toxic. The TRL doc
explicitly notes that the raw logit converges more smoothly than
softmax probability or log-probability — they tested all three. Do
**NOT** change this to `-logits[:, 1]`, `softmax(logits)[:, 0]`,
`log(softmax(logits)[:, 0])`, or any other variant. The recipe above
is what the tutorial uses and is what works.

**Training prompts:** RealToxicityPrompts, filter to prompts with
`prompt.toxicity > 0.3`. The filtered pool is on the order of 50k+
(the dataset has 99,442 prompts, stratified-sampled across toxicity).
Take **10,000 prompts** from this filtered pool with `seed=42`.

**IMPORTANT:** Make sure the 10,000 training prompts do NOT overlap
with the 400 eval prompts saved in `data/eval_prompts_400.json` from
Task 3.3.2. Compute set difference by index before training. If
overlap exists, exclude the overlapping prompts from the training
set, never from the eval set.

**Total steps:** Train for **2,000 PPO steps**. Stop earlier only if
mean reward plateaus AND mean response length drops below 10 tokens
(reward hacking signal).

**Checkpointing:**

- Save a checkpoint **every 50 steps** for the first 1000 steps.
- Save a checkpoint **every 100 steps** for steps 1000–2000.
- Each checkpoint saves: model weights, tokenizer, optimizer state,
  current PPO step number, value head.
- Path: `data/ppo_checkpoints/step_{step:04d}/`.

**Per-step training log (JSONL):**

For every PPO step, write one JSONL line to `data/ppo_train_log.jsonl`:

```json
{
  "step": int,
  "mean_reward": float,
  "reward_std": float,
  "mean_response_length": float,
  "kl_div_to_ref": float,
  "policy_loss": float,
  "value_loss": float,
  "n_collapsed_responses": int
}
```

**Rollout recording (CRITICAL — needed for later phases):**

For every PPO step, after generation but before the policy update,
serialize the full rollout batch to `data/rollouts/step_{step:04d}.pt`.
Each rollout entry must include:

- `prompt_text`, `prompt_token_ids`
- `response_text`, `response_token_ids`
- `policy_logprobs` (per-token, on the response only)
- `ref_logprobs` (per-token, on the response only — from `ref_model`)
- `reward` (scalar from RM, the raw nothate logit)
- `step` (PPO step number)

Per-token logprobs may be stored as `float16` — for downstream IF
computation this precision is sufficient.

**Acceptance criteria:**

- Training runs to completion (2000 steps) or terminates cleanly with
  a documented reason.
- `mean_reward` at step 500 is **higher than** at step 0 (sanity:
  PPO is doing something).
- `kl_div_to_ref` at every step is **finite and < 50**. KL exploding
  means the policy has drifted catastrophically; stop and report.
- `mean_response_length` does **not** drop below 10 in any window of
  50 consecutive steps. If it does, this is reward hacking; stop,
  save what you have, and report.

### Task 3.5 — Per-checkpoint evaluation (including step 0 base model)

Evaluate every saved checkpoint **AND** the untrained base
GPT2-medium (treated as `step=0`). Write
`src/evaluate_checkpoints.py`.

**Eval set:** the same 400-prompt subset from Task 3.3.2, loaded from
`data/eval_prompts_400.json`. Reusing the identical subset across
step 0 and all PPO checkpoints enables direct pre/post comparison.

**For each checkpoint (including the step 0 base model):**

1. Generate 30 new tokens per prompt (`do_sample=True`,
   `max_new_tokens=30`, `seed=42`).
2. Score **the continuation only — NOT prompt + continuation** using
   `evaluate.load("toxicity", module_type="measurement")`, exactly as
   in Task 3.3.2. Record mean and std.
3. Compute Wikitext-2 PPL using the same recipe as Task 3.3.1.
4. Compute mean response length.
5. Compute mean RM reward (raw nothate logit) on the 400
   continuations — for direct comparison with the training-time
   reward signal.

**Output:** `data/checkpoint_eval.csv` with columns:

`step, mean_toxicity, std_toxicity, mean_reward_logit, ppl_wikitext, mean_length, n_short_responses`

The `step=0` row uses the untrained base GPT2-medium. Its values for
`mean_toxicity` and `ppl_wikitext` should match the numbers
independently computed in Task 3.3.

**Acceptance criteria:**

- Every saved checkpoint AND the step 0 base model are evaluated.
  No skips.
- Step 0 row's `mean_toxicity` matches `data/baseline_toxicity.json`
  exactly (same code path, same seed, same prompts — should be
  bit-identical).
- Step 0 row's `ppl_wikitext` matches `data/baseline_ppl.json` within
  ±0.5.
- The final-step `mean_toxicity` is **lower** than the step 0 value.
  If not, detoxification did not happen — surface this immediately
  before doing anything else.
- A plot `reports/phase0_checkpoint_curves.png` is generated showing
  `mean_toxicity`, `ppl_wikitext`, and `mean_length` vs step on a
  shared x-axis. The human will use this plot to choose which
  checkpoint(s) to analyze in later phases.

## 4. Deliverables checklist

A phase 0 report at `reports/phase0_report.md` containing:

- [ ] Confirmation that base env (Python 3.10, PyTorch 2.1) was used,
  with the additionally installed packages and versions in
  `requirements.txt`.
- [ ] Probe verification output (accuracy, top-10 vectors, top tokens)
  with pass/fail for each acceptance criterion.
- [ ] Baseline PPL (Task 3.3.1): the number, and confirmation it
  falls in 20–25.
- [ ] Baseline toxicity (Task 3.3.2): mean and std on the fixed
  400-prompt subset.
- [ ] PPO training summary: total steps run, why stopped, total
  wall-clock time.
- [ ] Plot of training curves (reward, KL, length, value loss) vs step.
- [ ] Plot of checkpoint eval curves (toxicity, PPL, length) vs step.
- [ ] Confirmation that step 0 row in `checkpoint_eval.csv` matches
  the independent baselines from Task 3.3.
- [ ] List of saved checkpoints with paths.
- [ ] **Recommended checkpoint(s) for phase 1+ analysis**, with
  justification (typically: first checkpoint where toxicity has
  dropped substantially but PPL has not yet degraded > 20%).
- [ ] Any deviations from this document, with reasons.
- [ ] Anything skipped or incomplete, explicitly listed.

## 5. What NOT to do in this phase

- Do not implement any influence function computation. That is phase 2
  (EK-FAC).
- Do not attempt to compute Fisher matrices, EK-FAC factors, or any
  curvature estimates. That is phase 2.
- Do not implement the gradient similarity sanity baseline here. It is
  computed inside phase 2 alongside EK-FAC.
- Do not import from `references/`. Reimplement under `src/`. See
  CLAUDE.md "Reference materials" for the rationale.
- Do not embed the wandb API key in any source file, config file, or
  committed file. Use `wandb login` or env var.
- Do not change the reward function. It is `logits[:, 0]` (raw nothate
  logit), used directly, no softmax, no negation.
- Do not score toxicity on prompt + continuation. Score the
  continuation only.
- Do not change the eval prompt subset between Task 3.3.2 and Task 3.5.
  The whole point is they are identical.
- Do not change the reward model, the parameter subspace target
  (Layer 19 W_2), or the eval methodology. These are locked.
- Do not start phase 1 until the human has reviewed
  `phase0_report.md` and explicitly approved.

## 6. Anticipated failure modes (read before starting)

1. **Probe accuracy below 92%**: By far the most common cause is
   reading the wrong index from `outputs.hidden_states`. The tuple has
   25 entries: index 0 is the post-embedding state (before any
   transformer block), indices 1–24 are the outputs of the 24
   transformer blocks. Lee et al.'s $x^{L-1}$ refers to the output of
   the last (24th) transformer block, which is `hidden_states[-1]`,
   i.e. `hidden_states[24]`. Do **NOT** use `hidden_states[-2]` —
   that is the output of the second-to-last block and gives wrong
   activations for this probe. If accuracy is still off after
   verifying the index, check (a) that you are averaging across all
   token positions, not just taking the last token, (b) that
   tokenization is the GPT2 BPE tokenizer (not a different one), and
   (c) diff your implementation against
   `references/DPO-detoxify/toxicity/eval_interventions/` to find any
   detail-level divergence.

2. **PPO reward looks reasonable but model gets MORE toxic over time**:
   Check the reward sign and label index. With this RM, label 0 =
   `nothate` and label 1 = `hate`. The reward must be `logits[:, 0]`
   directly (no negation, no softmax). If you used `logits[:, 1]`,
   negated, or applied softmax, the model is optimizing for the wrong
   target.

3. **Wikitext PPL on base GPT2-medium is wildly outside 20–25**:
   Likely a tokenization, stride, or batching issue in the PPL
   computation. Reference value is ~22. Fix the eval code, not the
   model. This number is the foundation for all later checkpoint
   comparisons.

4. **`evaluate.load("toxicity")` numbers move in the OPPOSITE
   direction of training reward**: This is expected and correct, not
   a bug. Eval uses softmax probability of "hate" (∈ [0, 1], higher
   = more toxic) while training reward uses raw nothate logit
   (unbounded, higher = less toxic). They have opposite direction by
   design. Detoxification = eval mean ↓ AND training reward ↑. Do not
   "fix" either to match the other.

5. **Training prompt overlap with eval prompts**: If you skipped the
   set-difference check before training, your post-training toxicity
   numbers are contaminated by train-test leakage. The 400 eval
   prompts must be held out from training. If overlap exists, the
   correct fix is to remove the overlapping items from the training
   set, never from the eval set.
