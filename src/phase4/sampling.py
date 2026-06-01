"""On-policy rollout sampling at theta* for E2/E3.

These are FRESH draws y ~ pi_theta*(.|x) (CLAUDE.md / spec: NOT the stored
training rollouts), because E2/E3 estimate expectations under the current
policy.  We reuse Phase 1's sampler + scorer so the generation kwargs
(min_new_tokens=10, top_k=0, max_new_tokens=30) and the k1-KL / R_tilde
definitions are bit-identical to Phase 1.

A sample stores only token ids + scalars (reward, k1-KL, R_tilde); the
~2.36M-dim per-sample score s is recomputed in a streaming pass so we never
hold thousands of 9 MB tensors at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from phase1.score import score_one_prompt
from phase4.models import Models


@dataclass
class OnPolicySample:
    prompt_idx: int                  # index into the prompt list
    prompt_text: str
    prompt_ids: torch.Tensor         # (P,) cpu int64
    response_ids: torch.Tensor       # (R,) cpu int64
    reward: float                    # RoBERTa nothate logit
    k1_kl: float                     # sum_t [log pi_theta* - log pi_ref]
    R_tilde: float                   # reward - beta * k1_kl


def sample_on_policy_pool(
    prompts: list[str],
    k_per_prompt: int,
    seed_base: int,
    models: Models,
    beta: float,
    device: str,
) -> list[OnPolicySample]:
    """Draw ``k_per_prompt`` responses per prompt at theta*; score each.

    Prompt ``i`` uses generation seed ``seed_base + i`` so all prompts have
    independent streams and the pool is reproducible from (prompts, seed_base).
    """
    pool: list[OnPolicySample] = []
    for i, prompt_text in enumerate(prompts):
        samples, rows = score_one_prompt(
            prompt_text=prompt_text,
            k=k_per_prompt,
            seed=seed_base + i,
            device=device,
            tokenizer=models.tokenizer,
            model_star=models.policy,
            model_ref=models.ref,
            reward_tokenizer=models.reward_tokenizer,
            reward_model=models.reward,
            beta=beta,
        )
        for s, row in zip(samples, rows):
            pool.append(
                OnPolicySample(
                    prompt_idx=i,
                    prompt_text=prompt_text,
                    prompt_ids=s.prompt_token_ids.cpu(),
                    response_ids=s.response_token_ids.cpu(),
                    reward=float(row["reward"]),
                    k1_kl=float(row["k1_kl"]),
                    R_tilde=float(row["R_tilde"]),
                )
            )
    return pool
