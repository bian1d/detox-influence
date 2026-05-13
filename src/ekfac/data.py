"""Rollout loading + prompt-mask label construction.

Disk schema (one ``.pt`` file per PPO step, each a ``list[dict]``):

    prompt_text          str
    prompt_token_ids     LongTensor (P,)
    response_text        str
    response_token_ids   LongTensor (R,)
    policy_logprobs      Float16Tensor (R,)   # under π_θ at training time
    ref_logprobs         Float16Tensor (R,)   # under π_ref
    reward               float
    step                 int

For Stage 1A and Stage 3 we only need the token ids; the logprobs and
reward stay on disk. Prompt mask is applied at label-construction time
via ``ignore_index = -100`` so that ``F.cross_entropy(...,
ignore_index=-100, reduction='mean')`` divides by the response length R,
not P + R - 1 (CLAUDE.md hard constraints 1 and 3, CONTEXT.md §18.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch


@dataclass(frozen=True)
class Rollout:
    """One PPO rollout. ``step`` and ``reward`` are kept for traceability."""

    prompt_ids: torch.Tensor   # (P,)  dtype long
    response_ids: torch.Tensor  # (R,)  dtype long
    reward: float
    step: int

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_ids.shape[0])

    @property
    def response_len(self) -> int:
        return int(self.response_ids.shape[0])


def load_rollout_file(path: Path) -> list[Rollout]:
    """Load a single ``step_NNNN.pt`` file into Rollout dataclasses."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    out: list[Rollout] = []
    for r in raw:
        prompt_ids = r["prompt_token_ids"].long()
        response_ids = r["response_token_ids"].long()
        if prompt_ids.ndim != 1 or response_ids.ndim != 1:
            raise ValueError(
                f"unexpected token tensor shapes in {path}: "
                f"prompt {tuple(prompt_ids.shape)}, response {tuple(response_ids.shape)}"
            )
        if prompt_ids.numel() == 0 or response_ids.numel() == 0:
            raise ValueError(f"empty prompt/response in {path}")
        out.append(
            Rollout(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                reward=float(r["reward"]),
                step=int(r["step"]),
            )
        )
    return out


def iter_rollouts(paths: Iterable[Path]) -> Iterator[Rollout]:
    """Yield rollouts from a sorted sequence of step files."""
    for p in sorted(paths):
        yield from load_rollout_file(p)


def build_input_and_labels(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    *,
    response_labels: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_ids, labels) for a single rollout.

    ``input_ids = [prompt | response]`` of length ``P + R``.
    ``labels    = [-100]*P | response_labels`` of length ``P + R``.

    The shift-by-one cross-entropy alignment
    ``F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)``
    then yields one loss term per response token (R terms total), with no
    prompt-position contribution.

    Worked example (P=3 prompt = [a,b,c], R=2 response = [d,e]):

        positions:  0   1   2   3   4
        input_ids:  a   b   c   d   e
        labels:    -100 -100 -100 d   e

        shift_logits = logits[0..3]
        shift_labels = [-100, -100, d, e]
        # logits[2] predicts d (first response token, using prompt context)
        # logits[3] predicts e (second response token)
        # reduction='mean' divides by 2 = R.

    Args:
        prompt_ids: (P,) long tensor.
        response_ids: (R,) long tensor. Used as the ``input_ids`` response
            segment; also the default for ``response_labels``.
        response_labels: optional (R,) long tensor of *labels* for the
            response positions. Use this to inject pseudo-labels (Stage 1A)
            while keeping the actual response tokens as the input context.
        ignore_index: cross-entropy sentinel for prompt positions.

    Returns:
        input_ids: (P+R,) long.
        labels:    (P+R,) long, with ``[:P]`` set to ``ignore_index``.
    """
    if response_labels is None:
        response_labels = response_ids
    if response_labels.shape != response_ids.shape:
        raise ValueError(
            f"response_labels shape {tuple(response_labels.shape)} "
            f"!= response_ids shape {tuple(response_ids.shape)}"
        )
    input_ids = torch.cat([prompt_ids, response_ids], dim=0)
    labels = torch.cat(
        [
            torch.full(
                (prompt_ids.shape[0],), ignore_index, dtype=torch.long,
                device=prompt_ids.device,
            ),
            response_labels.long(),
        ],
        dim=0,
    )
    return input_ids, labels
