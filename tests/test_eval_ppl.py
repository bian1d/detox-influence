"""
Unit tests for src/eval_ppl.py.

All tests use tiny synthetic data and a minimal GPT2 config — no Wikitext
download, no full model forward pass in CI.
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eval_ppl import compute_ppl_from_ids, save_baseline_ppl


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model():
    config = GPT2Config(
        vocab_size=100, n_embd=16, n_layer=2, n_head=2, n_inner=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    torch.manual_seed(0)
    return GPT2LMHeadModel(config).eval()


# ── compute_ppl_from_ids ──────────────────────────────────────────────────────

class TestComputePplFromIds:
    def test_returns_finite_positive_float(self, tiny_model):
        ids = torch.randint(0, 100, (1, 300))
        ppl = compute_ppl_from_ids(tiny_model, ids, stride=128, max_length=256, device="cpu")
        assert isinstance(ppl, float)
        assert math.isfinite(ppl)
        assert ppl > 1.0

    def test_shorter_than_one_window(self, tiny_model):
        """Sequence shorter than max_length should still work (one window)."""
        ids = torch.randint(0, 100, (1, 50))
        ppl = compute_ppl_from_ids(tiny_model, ids, stride=512, max_length=1024, device="cpu")
        assert math.isfinite(ppl)
        assert ppl > 1.0

    def test_exactly_one_stride(self, tiny_model):
        """Sequence exactly stride tokens long — verify single-window case."""
        ids = torch.randint(0, 100, (1, 100))
        ppl = compute_ppl_from_ids(tiny_model, ids, stride=100, max_length=100, device="cpu")
        assert math.isfinite(ppl)

    def test_uniform_model_ppl_near_vocab_size(self):
        """
        A model that always outputs uniform logits over V tokens should give
        PPL ≈ V.  We verify this analytically.
        """
        V = 8

        class UniformLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Dummy parameter so .parameters() works
                self._dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, labels=None, **kwargs):
                B, T = input_ids.shape
                # Uniform logits
                logits = torch.zeros(B, T, V)
                loss = None
                if labels is not None:
                    import torch.nn.functional as F
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, V),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="mean",
                    )

                class Out:
                    pass
                out = Out()
                out.logits = logits
                out.loss = loss
                return out

        model = UniformLM()
        ids = torch.randint(0, V, (1, 100))
        ppl = compute_ppl_from_ids(model, ids, stride=50, max_length=100, device="cpu")
        # Expected PPL ≈ V = 8 (uniform distribution over V tokens)
        assert abs(ppl - V) < 0.5, f"Expected PPL ≈ {V}, got {ppl:.3f}"

    def test_context_masking_does_not_affect_result(self, tiny_model):
        """
        Changing stride should change which tokens are used as context vs target,
        but the function should still return a finite result regardless.
        """
        ids = torch.randint(0, 100, (1, 400))
        ppl_s512 = compute_ppl_from_ids(tiny_model, ids, stride=512, max_length=512, device="cpu")
        ppl_s128 = compute_ppl_from_ids(tiny_model, ids, stride=128, max_length=256, device="cpu")
        # Both should be valid (not testing exact equality — different strides give different PPL)
        assert math.isfinite(ppl_s512) and ppl_s512 > 1.0
        assert math.isfinite(ppl_s128) and ppl_s128 > 1.0


# ── save_baseline_ppl ─────────────────────────────────────────────────────────

class TestSaveBaselinePpl:
    def test_writes_correct_json(self, tmp_path):
        out = tmp_path / "baseline_ppl.json"
        save_baseline_ppl(22.3, out)
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert "ppl_wikitext2_test" in data
        assert abs(data["ppl_wikitext2_test"] - 22.3) < 1e-9

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "baseline_ppl.json"
        save_baseline_ppl(21.0, out)
        assert out.exists()

    def test_value_is_float_not_int(self, tmp_path):
        out = tmp_path / "ppl.json"
        save_baseline_ppl(22.0, out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data["ppl_wikitext2_test"], float)
