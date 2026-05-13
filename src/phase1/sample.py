"""Multi-rollout sampling at θ*.

Uses the same generation_kwargs as Phase 0 PPO training (min_new_tokens=10,
top_k=0, top_p=1.0, max_new_tokens=30) so that drawn y ~ π_θ* are from the
same distribution PPO trained against. Prompt tokenization matches training
convention (add_special_tokens=False, truncate to MAX_PROMPT_TOKENS).
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase1.config import GENERATION_KWARGS, MAX_PROMPT_TOKENS


@dataclass
class SampledResponse:
    prompt_text: str
    prompt_token_ids: torch.Tensor   # (T_p,) on CPU, int64
    response_text: str               # decoded with skip_special_tokens=True
    response_token_ids: torch.Tensor # (T_r,) on CPU, int64; EOS stripped at end


def tokenize_prompt(tokenizer: AutoTokenizer, prompt_text: str) -> torch.Tensor:
    """Tokenize a single prompt the same way Phase 0 training did."""
    ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    return ids


def sample_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    k: int,
    device: str,
    seed: int,
) -> list[SampledResponse]:
    """Sample k responses from π_θ for a single prompt.

    Uses `num_return_sequences=k` in a single generate call so all K samples
    advance from the same RNG state set by `torch.manual_seed(seed)` here.
    Reproducibility: identical (model, prompt, k, seed) gives identical
    samples regardless of caller's prior RNG state.
    """
    torch.manual_seed(seed)

    prompt_ids = tokenize_prompt(tokenizer, prompt_text).to(device)  # (T_p,)
    T_p = prompt_ids.shape[0]

    input_ids = prompt_ids.unsqueeze(0)  # (1, T_p)
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_KWARGS,
        )  # (k, T_p + T_new)

    eos_id = tokenizer.eos_token_id
    samples: list[SampledResponse] = []
    for row in out:
        new_ids = row[T_p:].tolist()
        if eos_id in new_ids:
            new_ids = new_ids[: new_ids.index(eos_id)]
        response_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        samples.append(
            SampledResponse(
                prompt_text=prompt_text,
                prompt_token_ids=prompt_ids.detach().cpu(),
                response_text=response_text,
                response_token_ids=torch.tensor(new_ids, dtype=torch.long),
            )
        )
    return samples
