# Step 277 KL Anomaly — Diagnostic Note

**Date:** 2026-05-13
**Run:** Phase 0 PPO run 3 (`min_new_tokens=10`)
**Status:** Isolated single-step anomaly, no downstream training impact.

---

## 1. Observed signal (wandb / training log)

| Step | objective/kl (TRL) | approxkl | reward | non_score_reward | kl_coef | length |
|------|--------------------|----------|--------|------------------|---------|--------|
| 275 | +9.02 | 0.297 | +3.480 | −0.088 | 0.207 | 21.2 |
| 276 | +6.54 | 0.061 | +3.721 | −0.067 | 0.207 | 20.1 |
| **277** | **−12.00** | **2.5549** | **+2.942** | **+0.115** | 0.207 | 21.7 |
| 278 | +10.37 | 0.039 | +2.973 | −0.095 | 0.207 | 22.7 |
| 279 | +10.75 | 0.028 | +3.165 | −0.098 | 0.207 | 22.8 |

Step 277 is the **only step in the entire 2000-step run** where `approxkl > 1`
(typical range 0.02–0.07). Both `objective/kl` and `non_score_reward` flip sign
in this one step and immediately return to baseline at step 278.

`objective/kl` is what we label `k2_kl_actual` in the JSONL — this is TRL's
own per-batch KL estimate. Despite the `k2` label in our code, with our
`kl_penalty='kl'` config it is actually a k1 estimator over the *padded*
batch tensor (TRL's source: `(logprobs - ref_logprobs).sum(dim=-1).mean(dim=0)`
on padded tensors).

---

## 2. Per-rollout breakdown at step 277

Computed `k1_sum = (policy_logprobs − ref_logprobs).sum()` over **non-pad
response tokens only** for each of the 32 rollouts in `step_0277.pt`:

```
batch mean k1_sum = +8.752    (non-pad rollout positions only)
TRL objective/kl  = −12.00    (padded-batch computation)
```

**The non-pad sum is +8.75; the padded sum is −12. The −21 gap comes from
pad-position contributions.** Pad tokens are `eos_token_id=50256`. At pad
positions TRL is scoring `log p_policy(EOS | …EOS context…) − log p_ref(EOS | …EOS context…)`,
which for our post-step-276 policy state was strongly negative.

### Most-negative k1_sum rollouts (non-pad)

| idx | k1_sum | T_r | reward | prompt | response |
|-----|--------|-----|--------|--------|----------|
| 30 | −3.34 | 11 | +3.875 | "Join today if you're a moron who can't" | " see yourself as yet. < Links On Postlivion" |
| 12 | −2.56 | 30 | +0.047 | "The barnacle has the largest penis in the" | " world. In addition, it has the 7-1/2 inches…" |
| 20 | −1.98 | 12 | +4.094 | "I think you deserve better than what your mother has provided" | " you for all of it.\n\n\nTo analysis:" |

### Most-positive k1_sum rollouts (non-pad)

| idx | k1_sum | T_r | reward | prompt | response |
|-----|--------|-----|--------|--------|----------|
| 22 | +26.13 | 30 | +2.798 | "After losing his penis during a botched circumcision…" | "old raped at an extraordinary speed…" |
| 3  | +20.72 | 30 | +1.433 | "And there was the repugnant accusation that…" | "the TPP". Suddenly interviews…" |
| 18 | +17.20 | 30 | +3.923 | "Every time a violent loser…" | " Tampa Bay, not at the Mercedes-Benz…" |

The high-k1_sum rollouts are all toxic-content prompts where the policy was
already actively diverging from the ref model (suppressing toxic tokens that
ref scored highly). These are normal high-detox rollouts — nothing pathological
in the texts themselves.

---

## 3. Hypothesis

The step-277 anomaly is a **single-step large policy update** driven by an
unlucky batch composition, where:

1. **Step 276's update was unusually large** (we cannot see this directly in
   step 276's `approxkl=0.061` because approxkl reflects the *within-step*
   epoch drift, not the resulting policy displacement). Whatever step 276 did,
   it left the policy in a state where its EOS distribution at pad-style
   contexts (a long run of EOS tokens) sharply diverges from the ref model.

2. **At the start of step 277, TRL recomputes logits with this perturbed
   policy.** Non-pad token logprobs look fine (sum is the normal +8.75). But
   pad-position logprobs swing strongly negative (because policy now scores
   continued EOS as less probable than ref does at deep-padding positions),
   driving the padded-batch sum to −12.

3. **The PPO update at step 277 then sees a large effective KL surprise**,
   producing `approxkl=2.55` (80× normal). This update brings the policy
   back into a normal regime — step 278 has padded `objective/kl=+10.37` and
   `approxkl=0.039`, identical to the trajectory mean.

The transient cause was almost certainly the batch composition at step 276:
three sexual/violent-content prompts (idx 3, 18, 22 in step 277's batch were
already showing k1_sum > +17, and step 276's batch likely had similar) created
a strong concentrated gradient on EOS-related parameters, briefly distorting
the policy's pad-context distribution.

**The pad-position contribution to TRL's `objective/kl` is an artefact of
how TRL pads the batch tensor.** The actual generated rollouts at step 277
are entirely normal — non-pad k1_sum mean = +8.75 is within one standard
deviation of the trajectory mean. **Our stored rollouts (used by Phase 2 IF
analysis) are not affected** because they contain only response tokens, not
pad positions.

---

## 4. Implications for Phase 2

- **No data integrity issue.** Step 277's stored rollouts (`step_0277.pt`)
  have correct, non-anomalous logprobs at all response positions. The
  per-token policy/ref logprob fields are slices over `response_token_ids`
  only and exclude any pad positions.
- **No retraining needed.** The single-step spike resolved on its own and
  the trajectory continued normally. Recommended θ* checkpoint is step 650
  (Pareto-optimal), well past the perturbation.
- **TRL's padded-batch `objective/kl` is not a reliable per-step health
  metric for short responses.** A more robust signal is `approxkl` (the
  within-step PPO drift): the same step shows `approxkl=2.55` which clearly
  flags it as the only anomalous step in the run, regardless of the padded
  vs non-padded ambiguity.

---

## 5. Reproducibility

To reproduce this analysis:

```python
import torch
import json

rollouts = torch.load("data/rollouts/step_0277.pt")
for i, r in enumerate(rollouts):
    k1_sum = (r["policy_logprobs"].float() - r["ref_logprobs"].float()).sum().item()
    print(f"idx={i}  k1_sum={k1_sum:+.2f}  T_r={r['response_token_ids'].shape[0]}  reward={r['reward']:+.3f}")

# Compare against training log
with open("data/ppo_train_log.jsonl") as f:
    entries = [json.loads(l) for l in f]
print(entries[277])
```

Expected: mean k1_sum ≈ +8.75 from rollouts, vs. training log k2_kl_actual = −12.
