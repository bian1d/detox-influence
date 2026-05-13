# Phase 1 — Effective Reward Variance Diagnostic (Kickoff)

You are starting Phase 1 of an RLHF influence-function research project. Phase 0 (PPO training) and Phase 2 (EK-FAC pipeline) have completed in separate terminals. Your job is the theoretical-diagnostic check that connects them: verify that the post-PPO model θ* approximates a KL-RL optimum well enough for the IF formula to be valid.

This document gives you everything you need to start. Read it carefully, then complete the kickoff items at the bottom before writing any pipeline code.

---

## Project Context

**The user**: 5th-year PKU EECS undergrad doing thesis on rollout-level Influence Functions for PPO-trained RLHF detoxification.

**Setting**:
- Base model: `EleutherAI/gpt-neo-125m` (hidden_size=768, num_layers=12, intermediate_size=3072)
- RL framework: TRL 0.9.6 PPOTrainer
- PPO completed 2000 steps with min_new_tokens=10 on RealToxicityPrompts using a RoBERTa hate-speech reward model
- 2000 step checkpoints at `data/ppo_checkpoints/step_NNNN/`
- 2000 rollout files at `data/rollouts/step_NNNN.pt`
- Each rollout file is a list of 32 dicts with keys: `prompt_text, prompt_token_ids, response_text, response_token_ids, policy_logprobs, ref_logprobs, reward, step`
- Phase 0 analysis identified **step_0650 as the target θ\*** (Pareto-optimal across toxicity, reward, PPL, length)

**Your output**: a diagnostic report `reports/phase1_report.md` answering two questions:

1. **Is θ\* (step_0650 checkpoint) close enough to a KL-RL optimum that the Section 5 IF formula is valid?**
2. **For different choices of prompt-resampling protocol, what is the variance of effective reward across rollouts at θ\*?**

The KL-RL optimal policy has the property: $\tilde R(x, y, \theta^*) = \beta \log Z(x)$ — for any fixed prompt $x$, all sampled responses $y$ should have the same effective reward (only depending on $x$, not $y$). The variance across $y$-samples at fixed $x$ measures **how close policy is to the KL-RL optimum**.

**Critical: Phase 1 is independent of Phase 2's IF computation.** You do not need EK-FAC. You only need θ*, ref model, and the ability to sample rollouts from θ* and compute KL.

---

## What "KL-RL Optimum" Means Here (Important Clarification)

The KL-RL optimal policy is defined **per-prompt**, with a closed-form expression that does not depend on a training set:

$$\pi^*_x(y) = \frac{\pi_\text{ref}(y|x) \exp(r(x, y) / \beta)}{Z(x)}$$

For **any** prompt $x$, this defines a unique optimal $\pi^*_x$. The PPO trained policy $\pi_{\theta^*}$ may approximate this optimum well or poorly, **per-prompt**.

**Phase 1 evaluates this approximation on the in-training-distribution**: the same `eval_prompts_400.json` prompts used in Phase 0 evaluation, which are drawn uniformly from RealToxicityPrompts (the same distribution PPO trained on). The goal is to check whether $\pi_{\theta^*} \approx \pi^*_x$ for the prompts that Phase 2 IF analysis will care about (which are precisely the in-training-distribution prompts that generated the rollouts in Phase 2).

**Why in-training-distribution is the right framing for IF validation**: PPO trains parametrically. The closed-form KL-RL optimum $\pi^*_x$ is defined per-prompt regardless of training set. But PPO's $\theta^*$ has limited parametric capacity; it approximates $\pi^*_x$ well on prompts it has trained on, less well on held-out prompts (generalization). Phase 2 IF analysis is computed against rollouts whose prompts are in the training distribution — so the relevant variance to measure is on in-training-distribution prompts.

**Optional bonus**: also test Framing B (held-out variance) to characterize generalization. Held-out prompts can be sampled from RealToxicityPrompts NOT used during PPO training. Compare per-prompt variance between training-distribution and held-out-distribution. This is a small additional test (~10 min) that gives a generalization-related sentence for the thesis. NOT required for the main Phase 1 deliverable.

A point worth keeping in mind while writing the report:

- PPO is **local optimization** (gradient-based, finite steps). It does not guarantee convergence to the KL-RL global optimum $\pi^*_x$; it converges to a local optimum.
- If the within-prompt variance you measure is non-zero (which it will be), the cause is some mix of:
  - (a) PPO converged to a local KL-RL optimum near, but not at, $\pi^*_x$
  - (b) PPO converged well but K=32 is a finite sample (statistical noise in your variance estimate)
  - (c) The β value at step 650 from the adaptive controller does not exactly match the closed-form $\pi^*$ for any single β
- Report should distinguish these contributions where possible. (a) is fundamental and what we care about; (b) is statistical noise that more samples would reduce; (c) is an artifact of TRL's adaptive controller that should be discussed but not "fixed".

**A theoretically pleasant observation for your methods section**: KL-regularized RL training inherently satisfies a Proximal Bregman Response Function (PBRF)-analogous condition because the KL term $\beta D_\text{KL}(\pi_\theta \| \pi_\text{ref})$ is itself a Bregman divergence. This means the IF formula's stationarity assumption ($\nabla L(\theta^*) = 0$ in classical IF or PBRF stationarity in Bae et al. 2022) is satisfied by construction in our setting, unlike a supervised CE-trained model where PBRF is needed as a corrective device. Phase 1's variance measurement is the empirical instantiation of this theoretical advantage.

---

## REQUIRED READING — Do this BEFORE writing any code

1. **Read `CONTEXT.md` Sections 1, 2, 5, 7, 8 in full.** Especially:
   - Section 1: KL-regularized RL objective definition
   - Section 2: KL-RL optimal policy form $\pi^*_x(y) = \pi_\text{ref}(y) \exp(r/\beta) / Z(x)$
   - Section 2 conclusion: $\tilde R(x, y, \theta^*) = \beta \log Z(x)$ for KL-RL optimum (the property we are checking)
   - Section 5: where the optimality assumption is used in the IF derivation
   - Section 7-8: effective reward variance definition and computation steps

2. **Read `CLAUDE.md` in full** for hard constraints and reporting protocol.

3. **Inventory artifacts** from Phase 0 and Phase 2:
   - `data/ppo_checkpoints/step_0650/` should be loadable
   - `data/rollouts/` should have step_0000.pt through step_1999.pt
   - `data/baseline_toxicity.json` (Phase 0 baseline data)
   - `reports/phase0_report.md` (the official Phase 0 outcome)
   - `reports/phase2_report.md` if it exists (Phase 2 EK-FAC report)

---

## Theoretical Framework (Quick Reference)

The KL-RL training objective:

$$J_x(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \right]$$

The **closed-form KL-RL optimal policy**:

$$\pi^*_x(y) = \frac{\pi_\text{ref}(y|x) \exp(r(x, y) / \beta)}{Z(x)}$$

where $Z(x) = \sum_y \pi_\text{ref}(y|x) \exp(r(x, y)/\beta)$.

At this optimum, the **effective reward** simplifies:

$$\tilde R(x, y, \theta^*) = r(x, y) - \beta \log \frac{\pi^*_x(y)}{\pi_\text{ref}(y|x)} = \beta \log Z(x)$$

**Key property**: $\tilde R(x, y, \theta^*)$ depends only on $x$, not on $y$. So for a fixed prompt $x$, sampling multiple responses $y_1, ..., y_K$ from $\pi_{\theta^*}$ should give **identical effective reward** at the KL-RL optimum.

**Phase 1 measures the deviation from this**: compute $\widehat{\operatorname{Var}}_x[\tilde R(x, y, \theta^*)]$ averaged over a sample of prompts. Smaller variance ⟺ closer to KL-RL optimum ⟺ IF formula is more valid.

---

## KL Estimator Convention (must match Phase 0)

Phase 0 PPO used **k1 estimator** (TRL default `kl_penalty='kl'`):

$$\hat{\mathrm{KL}}_\text{k1}(y|x) = \sum_t [\log \pi_\theta(y_t|x, y_{<t}) - \log \pi_\text{ref}(y_t|x, y_{<t})]$$

This is signed (can be negative on a single sample but unbiased in expectation).

**Phase 1 effective reward computation must use the same k1 estimator** for consistency:

$$\tilde R(x, y, \theta) = r(x, y) - \beta \cdot \hat{\mathrm{KL}}_\text{k1}(y|x)$$

where $\beta$ is the **final adaptive kl_coef value at step 650** (NOT the initial 0.2 — TRL's adaptive controller adjusts this over training). Read this from `data/ppo_train_log.jsonl` at step 650.

**Do not use k2, k3, or any other estimator.** Phase 0 used k1, so Phase 1 must use k1 to maintain the cancellation identity $\tilde R = \beta \log Z$ at the optimum (Section 2 of CONTEXT.md).

---

## Hard Constraints (Phase 1 specific)

1. **Reward computation**: Same RoBERTa hate-speech model and same scoring protocol as Phase 0 (continuation-only, raw nothate logit, no negation, no softmax). Reuse `score_toxicity_eval` / `score_with_reward_model` from `src/eval_toxicity.py`.

2. **KL estimator**: k1 (sum of per-token log-ratios over response tokens only, prompt-masked).

3. **β value**: read the adaptive `kl_coef` at step 650 from `data/ppo_train_log.jsonl`. Use this exact value, not 0.2.

4. **Prompt mask**: only response tokens contribute to the KL estimator (same as Phase 2's prompt mask convention).

5. **Sampling**:
   - Use `do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=30, min_new_tokens=10`
   - This matches Phase 0's generation kwargs to ensure y samples are from the same policy distribution that training was based on.

6. **Determinism**: seed all sampling with `torch.manual_seed(42)`. Phase 1 should be reproducible.

---

## Pipeline Structure

### Stage 1: Setup
- Load `data/ppo_checkpoints/step_0650/` as θ* (use `AutoModelForCausalLM`, drop value head)
- Load `EleutherAI/gpt-neo-125m` as ref model (untrained, NOT step_0)
- Load RoBERTa reward model (same as Phase 0)
- Read β = adaptive_kl_coef at step 650 from `data/ppo_train_log.jsonl`
- Choose prompt set: a fixed set of $N_x = 100$ prompts from `data/eval_prompts_400.json` (random subset, seeded)

### Stage 2: Multi-rollout sampling at θ*
For each of the 100 prompts $x_i$:
- Sample $K = 32$ responses $y_{i,1}, ..., y_{i,K}$ from θ* (using sampling settings above)
- For each response, compute:
  - $r(x_i, y_{i,k})$ — RoBERTa nothate logit on continuation
  - $\log \pi_{\theta^*}(y_{i,k} | x_i)$ — by forwarding θ* through the (prompt + response) sequence and slicing response tokens
  - $\log \pi_\text{ref}(y_{i,k} | x_i)$ — same but with ref model
  - $\hat{\mathrm{KL}}_\text{k1}(y_{i,k}|x_i) = \sum_t [\log \pi_{\theta^*} - \log \pi_\text{ref}]$ over response tokens
  - $\tilde R(x_i, y_{i,k}, \theta^*) = r(x_i, y_{i,k}) - \beta \cdot \hat{\mathrm{KL}}_\text{k1}$

Store everything in a DataFrame / dict: $(x_i, y_{i,k}, r, \log\pi_{\theta^*}, \log\pi_\text{ref}, \hat{\mathrm{KL}}, \tilde R)$.

Compute cost estimate: 100 prompts × 32 samples × ~3 forward passes = 9,600 forward passes; ~5 sec each on GPT-Neo-125M = ~80 min wall-clock. Plan accordingly.

### Stage 3: Variance computation
For each prompt $x_i$, compute:

$$\widehat{\operatorname{Var}}_{x_i} = \frac{1}{K - 1} \sum_{k=1}^{K} \left( \tilde R_{i,k} - \bar{\tilde R}_{i \cdot} \right)^2$$

Aggregate across prompts:
- Mean per-prompt variance: $\frac{1}{N_x} \sum_i \widehat{\operatorname{Var}}_{x_i}$
- Median per-prompt variance
- Distribution: histogram and quartiles
- Total variance vs within-prompt variance ratio (intraclass correlation): how much of the total variance is between-prompt (= $\beta \log Z(x)$ differences) vs within-prompt (= deviation from KL-RL optimum)?

### Stage 4: Interpretation
The user wants concrete answers to:
- Is variance within-prompt < total variance × 0.1? (loose: <50%, strict: <10%)
- What is the within-prompt CoV (coefficient of variation) of $\tilde R$? Is it < 0.1 (close to optimum) or > 0.5 (far)?
- Specific failure modes: are there prompts where within-prompt variance is huge? What do those prompts look like?
- Does the variance scale with prompt length, prompt toxicity, or other observable features?

---

## Acceptance Criteria

This phase is diagnostic, not pass/fail. Output is a report containing:

1. **The numbers**: mean within-prompt variance, mean total variance, ratio. Distribution histograms.
2. **The interpretation**: is θ* close to KL-RL optimum? Quantified.
3. **The implication for Phase 2 IF**: if variance is large, the IF cancellation property is weakly satisfied — Phase 2 IF scores have inflated noise. If variance is small, IF is more reliable.
4. **The specific outlier prompts**: which 5-10 prompts have largest within-prompt variance? What do their responses look like? Print prompt + 5 sampled responses + their individual $\tilde R$ values.

Phase 1 deliverable is `reports/phase1_report.md` with all numerical results, plots if useful, and the interpretation paragraph for the thesis methods section.

---

## File Structure to Create

```
src/phase1/
├── __init__.py
├── config.py       # checkpoint path, beta, sampling settings
├── sample.py       # multi-rollout sampling at theta*
├── score.py        # compute task reward, k1 KL, effective reward
└── variance.py     # per-prompt and aggregate variance computation

tests/
├── test_phase1_kl.py     # verify k1 KL matches Phase 0 rollout logprobs
├── test_phase1_reward.py # verify reward scoring matches Phase 0 baseline_toxicity.json on step_0 rollouts

src/run_phase1.py         # orchestrating script
```

---

## What NOT to do

- Do not touch `data/rollouts/`, `data/ppo_checkpoints/`, `src/train_ppo.py` — Phase 0 owns these.
- Do not touch `src/ekfac/` or `data/ekfac/` — Phase 2 owns these.
- Do not modify `CONTEXT.md` — user owns the theory doc.
- Do not skip the k1 estimator and use k2 or k3 — Phase 0 used k1, Phase 1 must match.
- Do not use β = 0.2 — read the adaptive value at step 650 from the log.
- Do not silently change acceptance criteria — Phase 1 doesn't have pass/fail gates, but the user wants quantified honesty.

---

## First-Day Action Items

1. **Read** CONTEXT.md Sections 1, 2, 5, 7, 8 in full.
2. **Read** CLAUDE.md in full.
3. **Inventory** Phase 0 outputs: confirm step_0650 checkpoint loads, ppo_train_log.jsonl is readable, eval_prompts_400.json exists.
4. **Read β at step 650** from data/ppo_train_log.jsonl. Report the exact value.
5. **Skeleton creation**: src/phase1/ with empty modules; src/run_phase1.py with the main flow stubbed.
6. **One-prompt smoke**: for a single prompt $x_1$ (the first in eval_prompts_400.json), sample 4 responses from θ*. Compute task reward, k1 KL, and effective reward for each. Print the 4 values of $\tilde R$. They should all be in roughly the same magnitude (within a factor of 2-3).

**STOP after Action Item 6 and surface a report** with:
- β at step 650 (numerical value)
- The 4 sampled responses for $x_1$ (prompt + response + r + KL + $\tilde R$)
- Confirmation that the pipeline runs end-to-end

The user will then approve Stage 2 (full 100-prompt × 32-sample run).

---

Read this doc, then say **"Phase 1 kickoff received, starting required reading."** Do not write any code yet. After completing required reading (items 1-2 above), say **"Required reading complete, ready for Action Items 3-6."** Then proceed.
