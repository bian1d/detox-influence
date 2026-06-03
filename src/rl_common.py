"""Reward + per-token-logprob helpers shared by PPO training and the scoring code.

These two functions are pure transformers/torch with **no trl dependency**.
Extracting them here lets the Phase 1 / Phase 4 / Phase 5-Stage-1 estimator code
reuse them without importing ``train_ppo``, whose module-level
``from trl import ...`` would otherwise drag TRL into environments (e.g. the OLMo
conda env) that intentionally do not install it. The PPO training script keeps
its own trl import and now pulls these two functions from here.

This is a pure code move — the function bodies are unchanged. The only edit is
the ``get_per_token_logprobs`` ``model`` annotation, narrowed from trl's
``AutoModelForCausalLMWithValueHead`` to ``torch.nn.Module`` (the body already
supported both the value-head model and a plain LM head at runtime).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    model: torch.nn.Module,  # was trl AutoModelForCausalLMWithValueHead; both work
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
