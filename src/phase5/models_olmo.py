"""OLMo-2-1B model loading for Phase 5 Stage 1 — parallel to phase4.models,
returning the same ``Models`` dataclass so the reused Phase 1/4 scoring code
(score_one_prompt, e1 decomposition, delta_vp) works unchanged.

theta* = OLMo-2-0425-1B-SFT, reference = OLMo-2-0425-1B (pretrained),
reward = RoBERTa hate-speech (unchanged from GPT-Neo for comparability).
"""
from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from phase4.models import Models
from phase5.config import REF_ID, REWARD_MODEL_ID, THETA_STAR_ID


def load_olmo_models(
    device: str,
    *,
    policy_requires_grad: bool = False,
    policy_attn_implementation: str | None = None,
) -> Models:
    """Load (theta*=OLMo-SFT, pi_ref=OLMo-base, reward=RoBERTa).

    ``policy_attn_implementation='eager'`` is required for the second-order
    (double-backward) HVPs in the thermometer; ICC sampling can leave it None.
    ``policy_requires_grad`` leaves the policy trainable (thermometer needs
    grads wrt phi); ICC keeps everything frozen.
    """
    tokenizer = AutoTokenizer.from_pretrained(THETA_STAR_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy_kwargs = {"torch_dtype": torch.float32}
    if policy_attn_implementation is not None:
        policy_kwargs["attn_implementation"] = policy_attn_implementation
    policy = AutoModelForCausalLM.from_pretrained(THETA_STAR_ID, **policy_kwargs).to(device).eval()
    ref = AutoModelForCausalLM.from_pretrained(REF_ID, torch_dtype=torch.float32).to(device).eval()
    if not policy_requires_grad:
        for p in policy.parameters():
            p.requires_grad_(False)
    for p in ref.parameters():
        p.requires_grad_(False)

    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    reward = (
        AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID)
        .to(device)
        .eval()
    )
    for p in reward.parameters():
        p.requires_grad_(False)

    return Models(tokenizer, policy, ref, reward_tokenizer, reward)
