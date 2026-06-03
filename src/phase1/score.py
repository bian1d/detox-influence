"""Per-response scoring: task reward, k1 KL, effective reward.

The k1 estimator is the unbiased signed log-ratio (Schulman 2020):
    KL_k1(y|x) = sum_t [log π_θ(y_t|x,y_<t) - log π_ref(y_t|x,y_<t)]
Sum is over response tokens only (prompt-masked by construction in
get_per_token_logprobs, which slices response positions). Phase 1 must use
k1 (not k2) for two reasons: (a) k1 is unbiased w.r.t. true KL, so the
cancellation identity R̃ = β log Z(x) is the correct test at the KL-RL
optimum; (b) Phase 0 PPO used TRL kl_penalty='kl' = k1, so the diagnostic
matches the training convention.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from phase1.sample import SampledResponse, sample_responses
from rl_common import compute_rewards, get_per_token_logprobs


def compute_reward_scores(
    response_texts: list[str],
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    device: str,
) -> list[float]:
    """Raw nothate logit (logits[:, 0]) from RoBERTa hate-speech model.

    Continuation-only (response text, no prompt concat) — same protocol PPO
    used during training. Higher = less toxic.
    """
    return compute_rewards(response_texts, reward_model, reward_tokenizer, device)


def response_logprobs(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Per-token log π(y_t | x, y_<t) over response positions; shape (T_r,)."""
    return get_per_token_logprobs(model, prompt_ids, response_ids, device)


def k1_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> float:
    """k1 KL estimator: sum_t [log π_θ - log π_ref] over response tokens.

    Signed; can be negative on a single sample but unbiased in expectation.
    """
    assert policy_logprobs.shape == ref_logprobs.shape
    return (policy_logprobs.float() - ref_logprobs.float()).sum().item()


def effective_reward(reward: float, k1_kl_value: float, beta: float) -> float:
    """R̃(x, y, θ) = r(x, y) - β · KL_k1(y|x)."""
    return reward - beta * k1_kl_value


def score_one_prompt(
    prompt_text: str,
    k: int,
    seed: int,
    device: str,
    tokenizer: AutoTokenizer,
    model_star: AutoModelForCausalLM,
    model_ref: AutoModelForCausalLM,
    reward_tokenizer: AutoTokenizer,
    reward_model: AutoModelForSequenceClassification,
    beta: float,
) -> tuple[list[SampledResponse], list[dict]]:
    """Sample K responses from θ* for one prompt; score each.

    Returns (samples, rows). Each row: response_text, response_len,
    reward, k1_kl, R_tilde. Per-token logprobs are NOT stored in rows
    to keep Stage 2 output size manageable; they can be re-computed if
    needed since seeds are recorded.
    """
    samples = sample_responses(
        model_star, tokenizer, prompt_text, k=k, device=device, seed=seed
    )
    response_texts = [s.response_text for s in samples]
    rewards = compute_reward_scores(response_texts, reward_model, reward_tokenizer, device)

    rows: list[dict] = []
    for s, r in zip(samples, rewards):
        pol_lp = response_logprobs(model_star, s.prompt_token_ids, s.response_token_ids, device)
        ref_lp = response_logprobs(model_ref, s.prompt_token_ids, s.response_token_ids, device)
        kl = k1_kl(pol_lp, ref_lp)
        R_tilde = effective_reward(r, kl, beta)
        rows.append(
            {
                "response_text": s.response_text,
                "response_len": int(s.response_token_ids.numel()),
                "reward": float(r),
                "k1_kl": float(kl),
                "R_tilde": float(R_tilde),
            }
        )
    return samples, rows
