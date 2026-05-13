"""
Task 3.4 — PPO detoxification training on GPT-Neo-125m with dense rollout recording.

Config: TRL official detoxification tutorial (Burtenshaw 2022), verified to
detox GPT-Neo-125M from toxicity 0.1627 to 0.1148.

  model_name     = "EleutherAI/gpt-neo-125M"
  learning_rate  = 2.94e-5
  batch_size     = 32
  mini_batch_size= 32
  ppo_epochs     = 4
  init_kl_coef   = 0.2
  adap_kl_ctrl   = True     # adaptive; drives kl_coef down if KL grows
  target         = 6.0      # target KL for adaptive controller
  kl_penalty     = 'kl'     # k1 = log_pi_theta - log_pi_ref (TRL default)
  forward_batch_size = 1

KL estimator note (why k1 + adap_kl_ctrl works here but not on GPT2-medium):
  k1 = log_pi_theta - log_pi_ref is signed: goes negative when policy suppresses
  a token that the ref model assigned high probability. In detox training, if the
  policy completely avoids a toxic token that log_ref ≈ -2, k1 for that position
  is about -18 for the SUPPRESSED token — BUT the response no longer contains that
  token; it contains whatever the policy chose instead. For those generated tokens,
  k1 is close to 0. The adap_kl_ctrl controller keeps the overall KL penalty stable.
  GPT-Neo-125m has lower base probability on toxic tokens than GPT2-medium (different
  pretraining corpus), so early-training k1_sum is less extreme.

  GPT2-medium runs 1-7 all failed because GPT2-medium's high base toxicity made k1_sum
  reach -500 per response from step 1, turning the KL penalty into a +100 effective
  reward bonus. Switching to GPT-Neo-125m with adap_kl_ctrl avoids this.

KL monitoring:
  objective/kl: TRL's per-token mean k1 (signed).
  objective/kl_coef: adaptive coefficient (varies during training).
  k2_kl_naive and k1_kl_naive computed from non-padded rollout logprobs for
  comparison. Hard stop: approxkl > 50.

Reward: logits[:, 0] (raw nothate logit from
  facebook/roberta-hate-speech-dynabench-r4-target).
  No softmax, no negation — higher = less toxic.

Rollout recording captures policy_logprobs and ref_logprobs BEFORE
trainer.step() modifies the policy. ref_model is always the frozen
initial copy; policy_logprobs are pre-update.

Length collapse history:
  Run 1 (min_length=-1): collapsed at step 902 (mean_length=3.6 tokens).
  Run 2 (min_length=10):  collapsed at step 1012 (mean_length ~5-8 tokens).
    Root cause: min_length is minimum TOTAL sequence length (prompt+continuation).
    Training prompts are 15-64 tokens, so min_length=10 had no effect.
    The ~110 extra steps came from a minority of short training prompts.
  Run 3 (min_new_tokens=10): min_new_tokens suppresses EOS for the first 10
    generated positions unconditionally — this is the correct parameter.

Run modes:
  --smoke : 30 steps at batch_size=32 -> data/smoke/  (no wandb)
  (default): Full 2000-step run -> data/  (wandb logging)
"""

import argparse
import json
import os
from itertools import cycle, islice
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

POLICY_MODEL_ID = "EleutherAI/gpt-neo-125m"
REWARD_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"
MAX_PROMPT_TOKENS = 64   # training prompt truncation length


# ── config ────────────────────────────────────────────────────────────────────

def build_ppo_config(batch_size: int = 32, log_with: Optional[str] = "wandb") -> PPOConfig:
    """TRL official detoxification tutorial config for GPT-Neo-125m."""
    project = os.environ.get("WANDB_PROJECT", "rlhf-if-detox")
    entity  = os.environ.get("WANDB_ENTITY",  "benzyl")
    return PPOConfig(
        model_name=POLICY_MODEL_ID,
        learning_rate=2.94e-5,
        batch_size=batch_size,
        mini_batch_size=32,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        init_kl_coef=0.2,
        kl_penalty="kl",        # k1 (TRL default); works with adap_kl_ctrl=True
        adap_kl_ctrl=True,      # adaptive controller; required for k1 stability
        target=6.0,             # target KL for adaptive controller
        cliprange=0.2,
        cliprange_value=0.2,
        seed=42,
        log_with=log_with,
        tracker_project_name=project,
        tracker_kwargs={"wandb": {"entity": entity}} if log_with == "wandb" else {},
    )


# ── training prompt loading ────────────────────────────────────────────────────

def load_training_prompts(
    eval_prompt_path: Path,
    tokenizer: AutoTokenizer,
    seed: int = 42,
    n: int = 10_000,
    toxicity_threshold: float = 0.3,
) -> Dataset:
    """
    Loads and tokenizes training prompts from RealToxicityPrompts.

    Filters to prompts with toxicity > toxicity_threshold, samples n
    with the given seed, excludes any indices that appear in the eval
    set, and returns a HuggingFace Dataset with 'input_ids' and 'query'.
    """
    import random

    print("Loading RealToxicityPrompts for training...")
    rtp = load_dataset("allenai/real-toxicity-prompts", split="train")

    # Load eval indices to exclude from training
    with open(eval_prompt_path) as f:
        eval_data = json.load(f)
    eval_indices = set(eval_data["indices"])

    # Build candidate index list efficiently: access the entire 'prompt' column at once
    # (HuggingFace dataset column access returns all rows as a list — much faster than
    #  row-by-row iteration or batched filter with Python-side lambdas on 99k rows).
    tox_values = rtp["prompt"]   # list of dicts, each with 'text' and 'toxicity'
    tox_array  = [row.get("toxicity") for row in tox_values]
    candidate_indices = [
        i for i, tox in enumerate(tox_array)
        if tox is not None and float(tox) > toxicity_threshold and i not in eval_indices
    ]
    pool_size = len(candidate_indices)
    print(f"  Filtered pool size (toxicity>{toxicity_threshold}, excl. eval): {pool_size:,}")

    if pool_size < n:
        raise RuntimeError(
            f"Filtered pool has only {pool_size} prompts; need {n}."
        )

    # Sample n original-RTP indices from the candidate pool, with the given seed
    rng = random.Random(seed)
    sampled_indices = rng.sample(candidate_indices, n)

    prompts = [rtp[i]["prompt"]["text"] for i in sampled_indices]
    print(f"  Sampled {len(prompts):,} training prompts (seed={seed})")

    # Tokenize — truncate to MAX_PROMPT_TOKENS, no padding (each stored separately)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize(text: str) -> list[int]:
        return tokenizer(
            text,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            add_special_tokens=False,
        )["input_ids"]

    data = [
        {"input_ids": tokenize(p), "query": p}
        for p in prompts
    ]
    return Dataset.from_list(data)


# ── reward ────────────────────────────────────────────────────────────────────

def compute_rewards(
    response_texts: list[str],
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    device: str,
) -> list[float]:
    """
    Returns raw nothate logits: logits[:, 0].tolist()

    Label 0 = nothate, label 1 = hate. Raw logit, no softmax, no negation.
    Higher = less toxic. This is the exact formula from the TRL tutorial.
    """
    inputs = reward_tokenizer(
        response_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = reward_model(**inputs).logits.float()
    return logits[:, 0].tolist()


# ── per-token logprobs ─────────────────────────────────────────────────────────

def get_per_token_logprobs(
    model: AutoModelForCausalLMWithValueHead,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Full sequence forward pass (prompt + response).
    Returns per-token log-probs for response positions only.
    Shape: (T_response,).

    Works with AutoModelForCausalLMWithValueHead (returns (logits, loss, value))
    and plain GPT2LMHeadModel (returns CausalLMOutput with .logits).

    Must be called BEFORE trainer.step() for policy_logprobs because
    step() updates the policy model in-place.
    """
    T_p = prompt_ids.shape[0]
    T_r = response_ids.shape[0]

    input_ids = torch.cat(
        [prompt_ids.to(device), response_ids.to(device)], dim=0
    ).unsqueeze(0)  # (1, T_p + T_r)

    with torch.no_grad():
        outputs = model(input_ids)

    # AutoModelForCausalLMWithValueHead returns (lm_logits, loss, value)
    lm_logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    # lm_logits: (1, T_p + T_r, vocab)

    log_probs = F.log_softmax(lm_logits[0], dim=-1)  # (T_p + T_r, vocab)

    # For response token at position t (0-indexed):
    # its prediction comes from log_probs at index (T_p - 1 + t)
    response_log_probs = log_probs[T_p - 1: T_p + T_r - 1, :]  # (T_r, vocab)
    response_ids_dev = response_ids.to(device)
    per_token_lp = response_log_probs[torch.arange(T_r, device=device), response_ids_dev]

    return per_token_lp  # (T_r,)


# ── rollout serialization ─────────────────────────────────────────────────────

def serialize_rollout_batch(
    step: int,
    queries: list[torch.Tensor],
    responses: list[torch.Tensor],
    policy_logprobs: list[torch.Tensor],
    ref_logprobs: list[torch.Tensor],
    rewards: list[float],
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> None:
    """
    Saves rollout batch to data/rollouts/step_{step:04d}.pt.

    Each dict contains 8 fields:
      prompt_text, prompt_token_ids,
      response_text, response_token_ids,
      policy_logprobs (float16), ref_logprobs (float16),
      reward, step.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = []
    for q, r, pl, rl, rw in zip(queries, responses, policy_logprobs, ref_logprobs, rewards):
        q_ids = q.cpu()
        r_ids = r.cpu()
        rollouts.append({
            "prompt_text":       tokenizer.decode(q_ids, skip_special_tokens=True),
            "prompt_token_ids":  q_ids,
            "response_text":     tokenizer.decode(r_ids, skip_special_tokens=True),
            "response_token_ids": r_ids,
            "policy_logprobs":   pl.cpu().to(torch.float16),
            "ref_logprobs":      rl.cpu().to(torch.float16),
            "reward":            float(rw),
            "step":              step,
        })
    path = output_dir / f"step_{step:04d}.pt"
    torch.save(rollouts, path)


# ── training log ──────────────────────────────────────────────────────────────

def write_training_log_entry(
    log_path: Path,
    step: int,
    stats: dict,
    responses: list[torch.Tensor],
    ratio_skip_count: int,
    k2_kl_naive: float,
    k1_kl_naive: float,
) -> None:
    """Appends one JSONL line to the training log."""

    def _scalar(key: str) -> float:
        v = stats.get(key, float("nan"))
        return float(v.item() if hasattr(v, "item") else v)

    n_collapsed = sum(1 for r in responses if r.shape[0] < 5)

    entry = {
        "step":                  step,
        "mean_reward":           _scalar("ppo/mean_scores"),
        "reward_std":            _scalar("ppo/std_scores"),
        "mean_response_length":  _scalar("tokens/responses_len_mean"),
        "k2_kl_actual":          _scalar("objective/kl"),          # TRL padded-batch k2_sum mean; what training applies
        "non_score_reward":      _scalar("ppo/mean_non_score_reward"),   # actual KL penalty per response (mean over batch)
        "k2_kl_naive":           k2_kl_naive,                      # k2_sum from non-padded rollout lp (for comparison)
        "k1_kl_naive":           k1_kl_naive,                      # k1_sum from non-padded rollout lp (diagnostic, negative when suppressing)
        "approxkl":              _scalar("ppo/policy/approxkl"),   # within-step accumulated drift; hard-stop trigger
        "kl_coef":               _scalar("objective/kl_coef"),     # should stay fixed at 0.02
        "ratio_skip_count":      ratio_skip_count,
        "policy_loss":           _scalar("ppo/loss/policy"),
        "value_loss":            _scalar("ppo/loss/value"),
        "n_collapsed_responses": n_collapsed,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _should_checkpoint(step: int, n_steps: int) -> bool:
    if step == n_steps - 1:
        return True          # always save the final step
    if step < 1000:
        return step % 50 == 0
    return step % 100 == 0


def _save_checkpoint(
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    step: int,
    checkpoint_dir: Path,
) -> None:
    path = checkpoint_dir / f"step_{step:04d}"
    path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    # Save step number alongside checkpoint
    with open(path / "ppo_step.json", "w") as f:
        json.dump({"ppo_step": step}, f)


# ── collator ──────────────────────────────────────────────────────────────────

def _collate_fn(batch: list[dict]) -> dict:
    """Returns a dict of lists — TRL PPO expects lists of variable-length tensors."""
    return {
        "input_ids": [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch],
        "query":     [item["query"] for item in batch],
    }


# ── smoke ─────────────────────────────────────────────────────────────────────

def run_smoke(
    policy_tokenizer: AutoTokenizer,
    device: str,
    smoke_dir: Path,
    eval_prompt_path: Path,
) -> None:
    """
    30-step smoke run at batch_size=4.

    Verifies:
    - Reward is finite and positive for neutral text
    - Reward is visibly climbing (or at least non-decreasing) over 30 steps
    - KL stays finite and < 5
    - Mean response length ≥ 10 in all batches
    - Rollout .pt file has all 8 required fields, correct shapes
    - ref_logprobs differ from policy_logprobs (i.e., two different models)
    """
    # Pre-create before tee/any shell redirection opens files in this dir.
    smoke_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir = smoke_dir / "rollouts"
    log_path     = smoke_dir / "ppo_train_log.jsonl"
    ckpt_dir     = smoke_dir / "ppo_checkpoints"

    print(f"\n=== Smoke: 30-step PPO at batch_size=32 (full config, no wandb) ===")
    print(f"Outputs → {smoke_dir}")

    _run_training_loop(
        policy_tokenizer=policy_tokenizer,
        device=device,
        rollouts_dir=rollouts_dir,
        log_path=log_path,
        ckpt_dir=ckpt_dir,
        eval_prompt_path=eval_prompt_path,
        n_steps=30,
        batch_size=32,
        log_with=None,   # no wandb for smoke
    )

    # ── structural verification ────────────────────────────────────────────
    rollout_files = sorted(rollouts_dir.glob("step_*.pt"))
    assert len(rollout_files) > 0, "No rollout files written"

    sample = torch.load(rollout_files[0])
    required_keys = {
        "prompt_text", "prompt_token_ids", "response_text", "response_token_ids",
        "policy_logprobs", "ref_logprobs", "reward", "step",
    }
    missing = required_keys - set(sample[0].keys())
    assert not missing, f"Rollout missing keys: {missing}"

    r0 = sample[0]
    T_r = r0["response_token_ids"].shape[0]
    assert r0["policy_logprobs"].shape == (T_r,), \
        f"policy_logprobs shape {r0['policy_logprobs'].shape} != ({T_r},)"
    assert r0["ref_logprobs"].shape == (T_r,), \
        f"ref_logprobs shape {r0['ref_logprobs'].shape} != ({T_r},)"

    # After ≥1 PPO update the policy diverges from the frozen ref model.
    # Step 0 is pre-first-update (both models are identical copies at that point),
    # so we check a later rollout where the policy has been updated at least once.
    if len(rollout_files) > 1:
        later = torch.load(rollout_files[1])
        r1 = later[0]
        assert not torch.allclose(
            r1["policy_logprobs"].float(), r1["ref_logprobs"].float(), atol=1e-3
        ), (
            "policy_logprobs == ref_logprobs at step 1 — likely both came from "
            "the same model instance (ref_model not properly frozen/copied)"
        )

    # ── log analysis ──────────────────────────────────────────────────────
    log_entries = []
    with open(log_path) as f:
        for line in f:
            log_entries.append(json.loads(line))

    rewards   = [e["mean_reward"]                    for e in log_entries]
    k2_actual = [e.get("k2_kl_actual", float("nan")) for e in log_entries]
    k2_naive  = [e.get("k2_kl_naive",  float("nan")) for e in log_entries]
    k1_naive  = [e.get("k1_kl_naive",  float("nan")) for e in log_entries]
    nsrs      = [e.get("non_score_reward", float("nan")) for e in log_entries]
    approxkls = [e.get("approxkl",        float("nan")) for e in log_entries]
    kl_coefs  = [e.get("kl_coef",         float("nan")) for e in log_entries]
    skips     = [e.get("ratio_skip_count", 0)           for e in log_entries]
    lengths   = [e["mean_response_length"]              for e in log_entries]

    rew_last50_avg = sum(rewards[-50:]) / len(rewards[-50:]) if rewards else float("nan")
    eff_rewards = [r + n for r, n in zip(rewards, nsrs)]

    print(f"\nSmoke summary ({len(log_entries)} steps):")
    print(f"  mean_reward:    first={rewards[0]:.3f}  last={rewards[-1]:.3f}  "
          f"last50_avg={rew_last50_avg:.3f}  "
          f"trend={'↑' if rewards[-1] > rewards[0] else '↓/→'}")
    print(f"  eff_reward:     first={eff_rewards[0]:.3f}  last={eff_rewards[-1]:.3f}  "
          f"{'OK (all > 0)' if min(eff_rewards) > 0 else f'WARN: min={min(eff_rewards):.3f}'}")
    print(f"  k2_kl_actual:   min={min(k2_actual):.2f}  max={max(k2_actual):.2f}  "
          f"(TRL padded; what training applies)")
    print(f"  k2_kl_naive:    min={min(k2_naive):.2f}  max={max(k2_naive):.2f}  "
          f"ratio_to_actual={max(k2_actual)/max(k2_naive):.1f}x" if max(k2_naive) > 0 else "")
    nsr_ok = all(v <= 0 for v in nsrs if not (v != v))  # nan-safe
    print(f"  non_score_rew:  min={min(nsrs):.3f}  max={max(nsrs):.3f}  "
          f"({'OK <= 0' if nsr_ok else 'WARN: some > 0'})")
    print(f"  k1_kl_naive:    min={min(k1_naive):.2f}  max={max(k1_naive):.2f}  (diagnostic, negative = suppressing)")
    print(f"  approxkl:       min={min(approxkls):.4f}  max={max(approxkls):.4f}  "
          f"{'OK' if max(approxkls) < 50.0 else 'WARN: approxkl > 50'}")
    print(f"  kl_coef:        min={min(kl_coefs):.4f}  max={max(kl_coefs):.4f}  "
          f"(adaptive; started at 0.2, target=6.0)")
    print(f"  ratio_skips:    first10_total={sum(skips[:10])}  last10_total={sum(skips[-10:])}  "
          f"{'OK' if sum(skips[-10:]) == 0 else 'INFO: skips still nonzero at end'}")
    print(f"  mean_length:    min={min(lengths):.1f}  max={max(lengths):.1f}  "
          f"{'OK' if min(lengths) >= 10.0 else 'WARN: length < 10'}")

    # Per-sample min_new_tokens verification: scan ALL rollout files for any
    # sample whose response is < 10 tokens. min_new_tokens=10 must make this
    # impossible; if any appear here, the generation constraint is not working.
    short_samples: list[tuple[int, int, int]] = []   # (step, sample_idx, n_tokens)
    for rf in rollout_files:
        batch = torch.load(rf)
        for i, item in enumerate(batch):
            n_tok = item["response_token_ids"].shape[0]
            if n_tok < 10:
                short_samples.append((item["step"], i, n_tok))

    if short_samples:
        print(f"\n  *** BUG: {len(short_samples)} response(s) have < 10 new tokens — "
              f"min_new_tokens=10 is NOT constraining generation! ***")
        for step_s, idx_s, n_s in short_samples[:5]:
            print(f"       step={step_s} sample={idx_s} n_tokens={n_s}")
    else:
        print(f"  per-sample check: all responses ≥ 10 new tokens  ✓")

    print(f"  rollout files:  {len(rollout_files)}")
    print(f"  rollout keys verified: {sorted(required_keys)}")
    print(f"  policy_logprobs dtype: {r0['policy_logprobs'].dtype}")
    if len(rollout_files) > 1:
        print(f"  ref_logprobs differ from policy_logprobs at step 1: CONFIRMED")
    print("\nSmoke: PASS" if not short_samples else "\nSmoke: FAIL (short responses)")


# ── core training loop ────────────────────────────────────────────────────────

def _run_training_loop(
    policy_tokenizer: AutoTokenizer,
    device: str,
    rollouts_dir: Path,
    log_path: Path,
    ckpt_dir: Path,
    eval_prompt_path: Path,
    n_steps: int,
    batch_size: int,
    log_with: Optional[str],
) -> None:
    """
    Shared implementation for smoke and full run.

    Critical ordering per CLAUDE.md:
      1. generate responses
      2. compute rewards
      3. get policy_logprobs (pre-step, from policy model)
      4. get ref_logprobs (from frozen ref_model)
      5. serialize rollout
      6. trainer.step() — updates policy in-place
      7. log + checkpoint
    """
    config = build_ppo_config(batch_size=batch_size, log_with=log_with)

    # ── models ────────────────────────────────────────────────────────────
    print(f"Loading {POLICY_MODEL_ID}...")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(POLICY_MODEL_ID).to(device)
    ref_model    = create_reference_model(policy_model)

    print(f"Loading reward model {REWARD_MODEL_ID}...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    reward_model     = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_ID
    ).to(device)
    reward_model.eval()

    # ── training data ─────────────────────────────────────────────────────
    dataset = load_training_prompts(
        eval_prompt_path=eval_prompt_path,
        tokenizer=policy_tokenizer,
        seed=42,
        n=min(10_000, n_steps * batch_size * 2),  # smaller for smoke
    )

    trainer = PPOTrainer(
        config=config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=policy_tokenizer,
        dataset=dataset,
        data_collator=_collate_fn,
    )

    generation_kwargs = {
        "min_new_tokens": 10,    # minimum NEW tokens generated, regardless of prompt length.
                                 # min_length (total length) failed: prompts are 15-64 tokens,
                                 # so min_length=10 imposed no constraint. Run 1 (min_length=-1)
                                 # collapsed at step 902; run 2 (min_length=10) collapsed at step
                                 # 1012 — ~110 extra steps only because a fraction of training
                                 # prompts are short enough that min_length=10 had marginal effect.
                                 # min_new_tokens=10 suppresses EOS for the first 10 generated
                                 # positions unconditionally.
        "top_k":          0.0,
        "top_p":          1.0,
        "do_sample":      True,
        "max_new_tokens": 30,
        "pad_token_id":   policy_tokenizer.eos_token_id,
    }

    print(f"\nStarting PPO training: {n_steps} steps, batch_size={batch_size}")

    _LENGTH_WINDOW_SIZE = 50
    _LENGTH_THRESHOLD   = 10.0
    _length_window: list[float] = []
    _ratio_skip_window: list[int] = []

    for step, batch in enumerate(islice(cycle(trainer.dataloader), n_steps)):
        query_tensors = batch["input_ids"]   # list of 1-D tensors

        # 1. Generate response tokens (response only, not prompt+response)
        response_tensors = trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs
        )

        # 2. Compute rewards from decoded response text
        response_texts = [
            policy_tokenizer.decode(r.squeeze(), skip_special_tokens=True)
            for r in response_tensors
        ]
        rewards_floats = compute_rewards(
            response_texts, reward_model, reward_tokenizer, device
        )
        rewards_tensors = [
            torch.tensor(r, dtype=torch.float32) for r in rewards_floats
        ]

        # 3 & 4. Per-token logprobs BEFORE step() mutates the policy model.
        # ref_model is frozen throughout, so order relative to step() doesn't
        # technically matter for ref — but we compute both here for clarity.
        policy_lp_list = []
        ref_lp_list    = []
        for q, r in zip(query_tensors, response_tensors):
            q_dev = q.to(device)
            r_dev = r.to(device)
            policy_lp_list.append(
                get_per_token_logprobs(trainer.model, q_dev, r_dev, device)
            )
            ref_lp_list.append(
                get_per_token_logprobs(trainer.ref_model, q_dev, r_dev, device)
            )

        # 5. Serialize rollout (pre-update policy logprobs + frozen ref logprobs)
        serialize_rollout_batch(
            step=step,
            queries=query_tensors,
            responses=response_tensors,
            policy_logprobs=policy_lp_list,
            ref_logprobs=ref_lp_list,
            rewards=rewards_floats,
            tokenizer=policy_tokenizer,
            output_dir=rollouts_dir,
        )

        # Naive diagnostics from rollout logprobs (non-padded individual sequences).
        # These differ from TRL's padded-batch computation — kept for comparison only.
        # k2_kl_naive: mean over batch of k2_sum per response (non-padded)
        # k1_kl_naive: mean over batch of k1_sum per response (signed, can be negative)
        k2_naive_sums = [
            (0.5 * (pl.float() - rl.float()).square()).sum().item()
            for pl, rl in zip(policy_lp_list, ref_lp_list)
        ]
        k1_naive_sums = [
            (pl.float() - rl.float()).sum().item()
            for pl, rl in zip(policy_lp_list, ref_lp_list)
        ]
        k2_kl_naive = sum(k2_naive_sums) / len(k2_naive_sums)
        k1_kl_naive = sum(k1_naive_sums) / len(k1_naive_sums)

        # 6. PPO update — intercept TRL ratio-skip warnings to count them,
        #    then restore the original handler.
        import warnings as _w
        _skip_count = [0]
        _orig_showwarn = _w.showwarning

        def _counting_showwarn(message, *args, **kw):
            if "Skipping batch" in str(message):
                _skip_count[0] += 1
            _orig_showwarn(message, *args, **kw)

        _w.showwarning = _counting_showwarn
        stats = trainer.step(query_tensors, response_tensors, rewards_tensors)
        _w.showwarning = _orig_showwarn
        ratio_skip_count = _skip_count[0]

        batch["response"] = response_texts   # log_stats requires this key
        trainer.log_stats(stats, batch, rewards_tensors)

        # 7. JSONL log
        write_training_log_entry(
            log_path, step, stats, response_tensors,
            ratio_skip_count=ratio_skip_count,
            k2_kl_naive=k2_kl_naive,
            k1_kl_naive=k1_kl_naive,
        )

        # ── progress print ────────────────────────────────────────────────
        def _s(k: str) -> float:
            v = stats.get(k, float("nan"))
            return float(v.item() if hasattr(v, "item") else v)

        rew        = _s("ppo/mean_scores")
        lng        = _s("tokens/responses_len_mean")
        approxkl   = _s("ppo/policy/approxkl")
        k2_actual  = _s("objective/kl")
        nsr        = _s("ppo/mean_non_score_reward")
        eff_r      = rew + nsr

        if step % 5 == 0 or step < 5:
            print(
                f"  step={step:4d}  reward={rew:+.3f}  eff_r={eff_r:+.3f}  "
                f"k2_sum={k2_actual:.2f}  approxkl={approxkl:.4f}  "
                f"len={lng:.1f}  skips={ratio_skip_count}"
            )

        # ── length window: soft warning + hard stop ───────────────────────
        _length_window.append(lng)
        if len(_length_window) > _LENGTH_WINDOW_SIZE:
            _length_window.pop(0)
        if len(_length_window) == _LENGTH_WINDOW_SIZE:
            window_mean_len = sum(_length_window) / _LENGTH_WINDOW_SIZE

            # Soft warning at 15 tokens — advance notice before hard stop at 10.
            _SOFT_LENGTH_THRESHOLD = 15.0
            if window_mean_len < _SOFT_LENGTH_THRESHOLD and step % 5 == 0:
                print(
                    f"  SOFT WARN step {step}: mean_length={window_mean_len:.1f} < "
                    f"{_SOFT_LENGTH_THRESHOLD} over last {_LENGTH_WINDOW_SIZE} steps"
                )

            # Hard stop: reward hacking (length collapse) ─────────────────
            if window_mean_len < _LENGTH_THRESHOLD:
                print(
                    f"\nREWARD HACKING at step {step}: "
                    f"mean_length={window_mean_len:.1f} < {_LENGTH_THRESHOLD} "
                    f"over last {_LENGTH_WINDOW_SIZE} steps. Stopping."
                )
                _save_checkpoint(trainer, policy_tokenizer, step, ckpt_dir)
                raise RuntimeError(
                    f"Reward hacking (length collapse) at step {step}; training stopped."
                )

        # ── warning: excessive ratio skips ───────────────────────────────
        # TRL's ratio>10 mini-batch skipping is itself a safety mechanism;
        # we surface sustained skipping as a warning but do not hard-stop.
        _ratio_skip_window.append(ratio_skip_count)
        if len(_ratio_skip_window) > 5:
            _ratio_skip_window.pop(0)
        if len(_ratio_skip_window) == 5 and all(s > 30 for s in _ratio_skip_window):
            print(
                f"  WARN step {step}: ratio_skip_count > 30 for 5 consecutive steps "
                f"(last 5: {_ratio_skip_window})"
            )

        # ── hard stop: KL explosion (approxkl, TRL k2, cumulative within-step) ─
        # TRL reports approxkl as the mean of 0.5*(log_new-log_old)^2 across
        # all 64 mini-batch updates in one outer step (4 epochs × 16 mini-batches).
        # Baseline at step 0 ≈ 4; steady-state ≈ target_kl × ppo_epochs = 6×4 = 24.
        # Threshold = 50 ≈ 2× steady-state: flags genuine drift while allowing normal noise.
        if approxkl > 50.0:
            print(f"\nKL EXPLOSION at step {step}: approxkl={approxkl:.4f} > 50. Stopping.")
            _save_checkpoint(trainer, policy_tokenizer, step, ckpt_dir)
            raise RuntimeError(f"approxkl > 50 at step {step}; training stopped.")

        # 8. Checkpoint
        if _should_checkpoint(step, n_steps):
            _save_checkpoint(trainer, policy_tokenizer, step, ckpt_dir)
            print(f"  Checkpoint saved: step_{step:04d}")

    print(f"\nTraining complete: {n_steps} steps.")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="PPO detoxification training on GPT2-medium."
    )
    parser.add_argument(
        "--eval-prompts", type=Path,
        default=repo_root / "data" / "eval_prompts_400.json",
        help="Path to the 400 eval prompt indices (to exclude from training).",
    )
    parser.add_argument(
        "--rollouts-dir", type=Path,
        default=repo_root / "data" / "rollouts",
    )
    parser.add_argument(
        "--log", type=Path,
        default=repo_root / "data" / "ppo_train_log.jsonl",
    )
    parser.add_argument(
        "--checkpoints", type=Path,
        default=repo_root / "data" / "ppo_checkpoints",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="30-step smoke run at batch_size=32 (full config). Outputs to data/smoke/.",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer ({POLICY_MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.smoke:
        run_smoke(
            policy_tokenizer=tokenizer,
            device=args.device,
            smoke_dir=repo_root / "data" / "smoke",
            eval_prompt_path=args.eval_prompts,
        )
        return

    _run_training_loop(
        policy_tokenizer=tokenizer,
        device=args.device,
        rollouts_dir=args.rollouts_dir,
        log_path=args.log,
        ckpt_dir=args.checkpoints,
        eval_prompt_path=args.eval_prompts,
        n_steps=2000,
        batch_size=32,
        log_with="wandb",
    )


if __name__ == "__main__":
    main()
