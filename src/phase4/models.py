"""Shared model loading for Phase 4.

A thin, explicit loader mirroring run_phase1._load_models so phase4 code does
not import a private function from a driver script.  theta* (the policy), the
reference policy and the RoBERTa reward model are the three models every
on-policy Phase 4 experiment needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from phase4.config import CKPT_DIR, REF_MODEL_ID, REWARD_MODEL_ID


@dataclass
class Models:
    tokenizer: AutoTokenizer
    policy: AutoModelForCausalLM            # theta* = step_0650
    ref: AutoModelForCausalLM               # pi_ref = base GPT-Neo-125M
    reward_tokenizer: AutoTokenizer
    reward: AutoModelForSequenceClassification


def load_models(
    device: str,
    policy_requires_grad: bool = False,
    policy_attn_implementation: str | None = None,
) -> Models:
    """Load (theta*, pi_ref, reward).  ``policy_requires_grad`` leaves the
    policy's parameters trainable (E2/E3 need gradients w.r.t. phi); E1 keeps
    everything in inference mode.  ``policy_attn_implementation='eager'`` forces
    the eager attention path, which is required for the second-order
    (double-backward) HVPs in E3 — fused/sdpa kernels may not support it."""
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    policy_kwargs = {}
    if policy_attn_implementation is not None:
        policy_kwargs["attn_implementation"] = policy_attn_implementation
    policy = AutoModelForCausalLM.from_pretrained(CKPT_DIR, **policy_kwargs).to(device).eval()
    ref = AutoModelForCausalLM.from_pretrained(REF_MODEL_ID).to(device).eval()
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
