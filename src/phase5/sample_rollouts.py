"""Generate OLMo-SFT on-policy rollouts for EK-FAC factor accumulation.

GPT-Neo reused its ~20,832 PPO training rollouts (≈600k response tokens) as the
Fisher data. OLMo-SFT has no PPO trajectory, so the analog is fresh on-policy
draws y ~ pi_theta*(.|x) over the RTP prompt set. accumulate_AS only needs the
token positions (the Fisher uses pseudo-labels sampled from the model), so these
rollouts carry no reward/KL — keeping generation lean.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ekfac.data import Rollout
from phase1.sample import sample_responses


def generate_factor_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    k_per_prompt: int,
    token_budget: int,
    device: str,
    *,
    seed_base: int = 42,
    log_every: int = 50,
) -> tuple[list[Rollout], int]:
    """Draw responses prompt-by-prompt until >= ``token_budget`` response tokens.

    Returns (rollouts, n_response_tokens). Each Rollout has reward=0.0/step=0
    (unused by the pseudo-label Fisher).
    """
    rollouts: list[Rollout] = []
    n_tok = 0
    for i, ptext in enumerate(prompts):
        samples = sample_responses(model, tokenizer, ptext, k=k_per_prompt,
                                   device=device, seed=seed_base + i)
        for s in samples:
            r = s.response_token_ids
            if r.numel() < 1:
                continue
            rollouts.append(Rollout(prompt_ids=s.prompt_token_ids, response_ids=r,
                                    reward=0.0, step=0))
            n_tok += int(r.numel())
        if (i + 1) % log_every == 0:
            print(f"    [factor-rollouts] {i + 1} prompts, {len(rollouts)} rollouts, "
                  f"{n_tok}/{token_budget} tokens")
        if n_tok >= token_budget:
            break
    return rollouts, n_tok
