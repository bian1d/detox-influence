"""Cross-validation: Phase 1's reward scoring must reproduce the raw
nothate logits recorded by Phase 0 PPO at training time.

Phase 1 uses the same `compute_rewards` function from src/train_ppo.py
(continuation-only, raw nothate logit, no softmax) — so the test is
trivially expected to pass. It still has value as a regression guard
in case the import path or model loading silently changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase1.config import REWARD_MODEL_ID  # noqa: E402
from phase1.score import compute_reward_scores  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

ROLLOUT_FILE = REPO / "data" / "rollouts" / "step_0000.pt"


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def reward_pair(device: str):
    tok = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID).to(device).eval()
    return model, tok


def test_reward_matches_recorded(reward_pair, device):
    """All 32 rewards in step_0000.pt match my re-scoring at high precision.

    Same model, same protocol → should match to fp32 precision. Allow
    atol=1e-4 to cover CUDA non-determinism across runs.
    """
    rollouts = torch.load(ROLLOUT_FILE, weights_only=False)
    response_texts = [r["response_text"] for r in rollouts]
    stored = torch.tensor([r["reward"] for r in rollouts], dtype=torch.float32)

    model, tok = reward_pair
    recomputed = torch.tensor(
        compute_reward_scores(response_texts, model, tok, device),
        dtype=torch.float32,
    )
    assert recomputed.shape == stored.shape
    torch.testing.assert_close(recomputed, stored, atol=1e-3, rtol=1e-4)
