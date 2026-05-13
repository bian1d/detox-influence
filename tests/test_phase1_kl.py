"""Cross-validation: my k1 KL pipeline must reproduce the policy/ref
log-probabilities that were recorded during PPO training.

Off-by-one convention (critical):
    data/ppo_checkpoints/step_N/     — weights POST-update at step N
    data/rollouts/step_N.pt          — policy_logprobs were computed by
                                       the pre-update policy at step N
                                       (i.e. the post-update policy at
                                       step N-1)

So to verify Phase 1's pipeline against θ* = step_0650's weights, the
right comparison file is data/rollouts/step_0651.pt — those stored
policy_logprobs were produced by the same post-step-650 policy whose
weights we now load from ppo_checkpoints/step_0650/.

Tolerance: stored logprobs are fp16 → ~1e-3 relative is the right scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase1.config import CKPT_DIR, REF_MODEL_ID  # noqa: E402
from phase1.score import response_logprobs  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ROLLOUT_FILE = REPO / "data" / "rollouts" / "step_0651.pt"


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_star(device: str):
    m = AutoModelForCausalLM.from_pretrained(CKPT_DIR).to(device).eval()
    return m


@pytest.fixture(scope="module")
def model_ref(device: str):
    m = AutoModelForCausalLM.from_pretrained(REF_MODEL_ID).to(device).eval()
    return m


@pytest.fixture(scope="module")
def rollout_samples():
    """First 3 rollouts from step_0651.pt — enough to catch systematic bugs."""
    data = torch.load(ROLLOUT_FILE, weights_only=False)
    return data[:3]


def test_policy_logprobs_match_recorded(model_star, rollout_samples, device):
    """θ* recomputed per-token logprobs must match stored fp16 values."""
    for i, sample in enumerate(rollout_samples):
        p_ids = sample["prompt_token_ids"]
        r_ids = sample["response_token_ids"]
        stored = sample["policy_logprobs"].float()  # promote fp16→fp32

        recomputed = response_logprobs(model_star, p_ids, r_ids, device).cpu().float()

        assert recomputed.shape == stored.shape, (
            f"sample {i}: shape mismatch {recomputed.shape} vs {stored.shape}"
        )
        # fp16 → fp32 promotion introduces noise ~1e-3 in log-probs that
        # are O(1-10) in magnitude. atol=2e-3 covers both stored
        # rounding and small CUDA non-determinism.
        torch.testing.assert_close(recomputed, stored, atol=2e-3, rtol=1e-3)


def test_ref_logprobs_match_recorded(model_ref, rollout_samples, device):
    """π_ref recomputed per-token logprobs must match stored fp16 values."""
    for i, sample in enumerate(rollout_samples):
        p_ids = sample["prompt_token_ids"]
        r_ids = sample["response_token_ids"]
        stored = sample["ref_logprobs"].float()

        recomputed = response_logprobs(model_ref, p_ids, r_ids, device).cpu().float()

        assert recomputed.shape == stored.shape
        torch.testing.assert_close(recomputed, stored, atol=2e-3, rtol=1e-3)


def test_k1_kl_sign_and_scale(model_star, model_ref, rollout_samples, device):
    """k1 KL on PPO-trained policy vs ref should be predominantly positive
    (policy has moved away from ref) and O(1-30) in sum over a 10-30
    token response. Catches gross bugs like prompt/response slicing
    off-by-one.
    """
    from phase1.score import k1_kl  # noqa: E402

    kls = []
    for sample in rollout_samples:
        p_ids = sample["prompt_token_ids"]
        r_ids = sample["response_token_ids"]
        pol = response_logprobs(model_star, p_ids, r_ids, device)
        ref = response_logprobs(model_ref, p_ids, r_ids, device)
        kls.append(k1_kl(pol, ref))
    assert all(-5.0 < k < 50.0 for k in kls), f"unreasonable KL values: {kls}"
