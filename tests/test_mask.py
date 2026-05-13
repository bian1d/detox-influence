"""Prompt-mask correctness for ``build_input_and_labels``.

The hard constraint from CLAUDE.md C1: prompt positions must contribute
zero loss to the cross-entropy used in Stage 1A and Stage 3. After the
standard shift-by-one alignment, the number of contributing positions must
equal the response length R, and ``reduction='mean'`` must divide by R.

This is the canonical (P=3, R=2) worked example from the kickoff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.data import build_input_and_labels  # noqa: E402


def test_canonical_mask_shapes_and_values() -> None:
    prompt = torch.tensor([10, 11, 12])   # P=3, [a,b,c]
    response = torch.tensor([20, 21])     # R=2, [d,e]
    input_ids, labels = build_input_and_labels(prompt, response)

    assert input_ids.tolist() == [10, 11, 12, 20, 21]
    assert labels.tolist() == [-100, -100, -100, 20, 21]


def test_shift_alignment_isolates_response() -> None:
    """After shifting, only the R=2 response positions contribute to loss."""
    prompt = torch.tensor([10, 11, 12])
    response = torch.tensor([20, 21])
    _, labels = build_input_and_labels(prompt, response)

    shift_labels = labels[1:]  # length P+R-1 = 4
    assert shift_labels.tolist() == [-100, -100, 20, 21]
    n_active = int((shift_labels != -100).sum().item())
    assert n_active == response.shape[0] == 2


def test_mean_reduction_divides_by_R() -> None:
    """Synthetic logits → ``F.cross_entropy(..., reduction='mean')`` divides
    by 2 (response length), not by 4 (shift-window length)."""
    prompt = torch.tensor([0, 1, 2])
    response = torch.tensor([3, 4])
    _, labels = build_input_and_labels(prompt, response)

    V = 7
    torch.manual_seed(0)
    logits = torch.randn(prompt.numel() + response.numel(), V)
    shift_logits = logits[:-1]
    shift_labels = labels[1:]

    loss_mean = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean")
    loss_sum = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
    R = response.numel()
    assert torch.allclose(loss_mean, loss_sum / R, atol=1e-7), (
        f"mean/sum ratio {(loss_sum / loss_mean).item():.4f} != R={R}"
    )


def test_pseudo_label_injection() -> None:
    """``response_labels`` overrides the response segment of ``labels`` but
    leaves ``input_ids`` untouched — used by Stage 1A to inject pseudo
    labels while keeping the original response as conditioning context."""
    prompt = torch.tensor([0, 1])
    response = torch.tensor([2, 3])
    pseudo = torch.tensor([5, 6])
    input_ids, labels = build_input_and_labels(prompt, response, response_labels=pseudo)

    assert input_ids.tolist() == [0, 1, 2, 3]
    assert labels.tolist() == [-100, -100, 5, 6]


def test_empty_or_shape_mismatch_raises() -> None:
    prompt = torch.tensor([0, 1])
    response = torch.tensor([2, 3])
    pseudo_wrong = torch.tensor([5, 6, 7])
    with pytest.raises(ValueError):
        build_input_and_labels(prompt, response, response_labels=pseudo_wrong)
