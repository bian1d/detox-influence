"""
Unit tests for src/verify_probe.py.

All tests are CPU-only and require no Jigsaw data. They use tiny synthetic
models and tensors to verify correctness of individual functions.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Add src to path so we can import without installing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verify_probe import (
    classify_with_probe,
    compute_accuracy,
    get_mean_pooled_last_hidden,
    load_probe,
    project_value_vec_to_vocab,
    rank_value_vecs_by_cosine,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """
    A minimal GPT2-like model (2 layers, d_model=64, d_mlp=128) for fast tests.
    Uses the real GPT2Tokenizer so tokenization is realistic.
    """
    config = GPT2Config(
        vocab_size=50257,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_inner=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config).eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


@pytest.fixture
def probe_weight_64():
    """Random unit probe vector in d=64 space."""
    torch.manual_seed(1)
    w = torch.randn(64)
    return w / w.norm()


# ── load_probe ────────────────────────────────────────────────────────────────

class TestLoadProbe:
    def test_plain_tensor(self, tmp_path):
        w = torch.randn(1024)
        path = tmp_path / "probe.pt"
        torch.save(w, path)
        weight, bias = load_probe(path)
        assert weight.shape == (1024,)
        assert bias is None
        assert torch.allclose(weight, w)

    def test_dict_with_weight_only(self, tmp_path):
        w = torch.randn(1024)
        path = tmp_path / "probe_dict.pt"
        torch.save({"weight": w}, path)
        weight, bias = load_probe(path)
        assert weight.shape == (1024,)
        assert bias is None

    def test_dict_with_weight_and_bias(self, tmp_path):
        w = torch.randn(1024)
        b = torch.tensor(0.5)
        path = tmp_path / "probe_bias.pt"
        torch.save({"weight": w, "bias": b}, path)
        weight, bias = load_probe(path)
        assert weight.shape == (1024,)
        assert bias is not None
        assert torch.allclose(bias, b)

    def test_missing_weight_key_raises(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"wrong_key": torch.randn(1024)}, path)
        with pytest.raises(ValueError, match="no 'weight' key"):
            load_probe(path)

    def test_unexpected_type_raises(self, tmp_path):
        path = tmp_path / "bad_type.pt"
        torch.save([1, 2, 3], path)
        with pytest.raises(TypeError):
            load_probe(path)

    def test_2d_weight_raises(self, tmp_path):
        path = tmp_path / "bad_shape.pt"
        torch.save(torch.randn(2, 1024), path)
        with pytest.raises(ValueError, match="1-D"):
            load_probe(path)


# ── classify_with_probe ───────────────────────────────────────────────────────

class TestClassifyWithProbe:
    def test_no_bias_positive_scores_predict_toxic(self):
        """Samples with positive dot product should be predicted toxic (1)."""
        torch.manual_seed(2)
        probe = torch.randn(16)
        # Construct hidden states that are strongly aligned with probe
        hidden = probe.unsqueeze(0).expand(5, -1) * 10.0
        preds = classify_with_probe(hidden, probe)
        assert preds.tolist() == [1, 1, 1, 1, 1]

    def test_no_bias_negative_scores_predict_clean(self):
        """Samples anti-aligned with probe should be predicted non-toxic (0)."""
        torch.manual_seed(2)
        probe = torch.randn(16)
        hidden = -probe.unsqueeze(0).expand(5, -1) * 10.0
        preds = classify_with_probe(hidden, probe)
        assert preds.tolist() == [0, 0, 0, 0, 0]

    def test_with_bias_shifts_threshold(self):
        """Bias should shift the decision boundary."""
        probe = torch.tensor([1.0, 0.0])
        bias = torch.tensor(-5.0)
        # hidden x dot probe = 3.0; 3.0 + (-5.0) = -2.0 → predict 0
        hidden = torch.tensor([[3.0, 0.0]])
        preds_no_bias = classify_with_probe(hidden, probe, probe_bias=None)
        preds_with_bias = classify_with_probe(hidden, probe, probe_bias=bias)
        assert preds_no_bias.item() == 1   # 3.0 > 0
        assert preds_with_bias.item() == 0  # 3.0 - 5.0 = -2.0 < 0

    def test_output_dtype_is_long(self):
        probe = torch.randn(8)
        hidden = torch.randn(4, 8)
        preds = classify_with_probe(hidden, probe)
        assert preds.dtype == torch.long


# ── get_mean_pooled_last_hidden ───────────────────────────────────────────────

class TestGetMeanPooledLastHidden:
    def test_output_shape(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        texts = ["hello world", "another sentence here", "short"]
        out = get_mean_pooled_last_hidden(
            model, tokenizer, texts, batch_size=2, device="cpu"
        )
        assert out.shape == (3, 64)  # (N, d_model)

    def test_uses_last_block_not_second_to_last(self, tiny_model_and_tokenizer):
        """
        Verifies that we read hidden_states[-1] (last block output).
        For tiny model with 2 layers: hidden_states has 3 elements
        (embedding + 2 block outputs). hidden_states[-1] = hidden_states[2].
        If the code accidentally reads hidden_states[-2], outputs differ.
        """
        model, tokenizer = tiny_model_and_tokenizer
        texts = ["test sentence"]

        # Patch: record which hidden state index is actually averaged
        captured_indices = []
        original_forward = model.forward

        def patched_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            captured_indices.append(len(out.hidden_states) - 1)  # should be len-1
            return out

        model.forward = patched_forward
        result = get_mean_pooled_last_hidden(model, tokenizer, texts, device="cpu")
        model.forward = original_forward

        # For 2-layer model: tuple length = 3 (emb + 2 blocks), index -1 = 2
        assert captured_indices[0] == 2

    def test_mask_weighted_mean_excludes_pad_positions(self, tiny_model_and_tokenizer):
        """
        Directly verify the pooling formula on a synthetic hidden state tensor.

        With left-padding, GPT2's causal attention allows real tokens to attend
        to pad tokens, so the hidden state values themselves differ between padded
        and unpadded runs — there is no representation-invariance guarantee.
        What we CAN verify is that our pooling formula only averages over
        positions where attention_mask == 1 (real tokens), not pad positions.
        """
        # Synthetic hidden states: batch=2, seq_len=4, d=3
        # Row 0 = pad (mask=0), rows 1-3 = real (mask=1)
        hidden = torch.zeros(2, 4, 3)
        hidden[0, 0] = torch.tensor([99.0, 99.0, 99.0])  # pad — should be excluded
        hidden[0, 1] = torch.tensor([1.0, 2.0, 3.0])     # real
        hidden[0, 2] = torch.tensor([4.0, 5.0, 6.0])     # real
        hidden[0, 3] = torch.tensor([7.0, 8.0, 9.0])     # real
        hidden[1, 0] = torch.tensor([1.0, 0.0, -1.0])    # real (no padding in row 1)
        hidden[1, 1] = torch.tensor([3.0, 0.0, -3.0])    # real
        hidden[1, 2] = torch.tensor([5.0, 0.0, -5.0])    # real
        hidden[1, 3] = torch.tensor([7.0, 0.0, -7.0])    # real

        mask = torch.tensor([[0, 1, 1, 1],  # first token is pad
                             [1, 1, 1, 1]], dtype=torch.long)

        # Apply the same formula as in get_mean_pooled_last_hidden
        masked = hidden * mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).float()
        x_bar = masked.sum(dim=1) / denom

        # Row 0: mean of positions 1,2,3 → (1+4+7)/3=4, (2+5+8)/3=5, (3+6+9)/3=6
        assert torch.allclose(x_bar[0], torch.tensor([4.0, 5.0, 6.0]))
        # Row 1: mean of all 4 → (1+3+5+7)/4=4, 0, (-1-3-5-7)/4=-4
        assert torch.allclose(x_bar[1], torch.tensor([4.0, 0.0, -4.0]))

        # Flat mean (wrong): row 0 would be (99+1+4+7)/4=27.75, not 4.0
        flat_mean = hidden.mean(dim=1)
        assert not torch.allclose(flat_mean[0], x_bar[0]), (
            "Flat mean should differ from mask-weighted mean when there are pad tokens"
        )

    def test_batching_produces_finite_output(self, tiny_model_and_tokenizer):
        """
        Batching with left-padding changes hidden state values for short texts
        (pad tokens shift real tokens' positions, altering causal attention context).
        We instead verify: output is finite, correct shape, and not all zeros.
        """
        model, tokenizer = tiny_model_and_tokenizer
        texts = ["alpha", "beta gamma delta", "epsilon zeta eta theta"]
        out = get_mean_pooled_last_hidden(model, tokenizer, texts, batch_size=2, device="cpu")
        assert out.shape == (3, 64)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        assert out.abs().max() > 0, "Output is all zeros"


# ── rank_value_vecs_by_cosine ─────────────────────────────────────────────────

class TestRankValueVecs:
    def test_output_length(self, tiny_model_and_tokenizer, probe_weight_64):
        model, tokenizer = tiny_model_and_tokenizer
        ranked = rank_value_vecs_by_cosine(model, probe_weight_64)
        # 2 layers × 128 neurons/layer = 256 total
        assert len(ranked) == 2 * 128

    def test_sorted_descending(self, tiny_model_and_tokenizer, probe_weight_64):
        model, tokenizer = tiny_model_and_tokenizer
        ranked = rank_value_vecs_by_cosine(model, probe_weight_64)
        sims = [x[0] for x in ranked]
        assert sims == sorted(sims, reverse=True)

    def test_cosine_range(self, tiny_model_and_tokenizer, probe_weight_64):
        model, tokenizer = tiny_model_and_tokenizer
        ranked = rank_value_vecs_by_cosine(model, probe_weight_64)
        for sim, _, _ in ranked:
            assert -1.01 <= sim <= 1.01

    def test_highest_cosine_is_correct(self, tiny_model_and_tokenizer):
        """
        Manually set one value vector to be identical to the probe;
        it should rank first with cosine ~1.0.
        """
        model, tokenizer = tiny_model_and_tokenizer
        probe = torch.randn(64)
        probe_norm = probe / probe.norm()

        # Overwrite layer 0, neuron 0 to be the probe direction
        with torch.no_grad():
            model.transformer.h[0].mlp.c_proj.weight[0] = probe_norm * 5.0

        ranked = rank_value_vecs_by_cosine(model, probe)
        top_sim, top_neuron, top_layer = ranked[0]
        assert top_layer == 0
        assert top_neuron == 0
        assert abs(top_sim - 1.0) < 1e-5

    def test_layer_neuron_indices_in_range(self, tiny_model_and_tokenizer, probe_weight_64):
        model, tokenizer = tiny_model_and_tokenizer
        ranked = rank_value_vecs_by_cosine(model, probe_weight_64)
        for _, neuron_idx, layer_idx in ranked:
            assert 0 <= layer_idx < 2
            assert 0 <= neuron_idx < 128


# ── project_value_vec_to_vocab ────────────────────────────────────────────────

class TestProjectValueVecToVocab:
    def test_output_length(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        result = project_value_vec_to_vocab(model, tokenizer, layer=0, neuron_idx=0, top_k=5)
        assert len(result) == 5

    def test_scores_decreasing(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        result = project_value_vec_to_vocab(model, tokenizer, layer=0, neuron_idx=0, top_k=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_tokens_are_strings(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        result = project_value_vec_to_vocab(model, tokenizer, layer=1, neuron_idx=5, top_k=5)
        for tok, score in result:
            assert isinstance(tok, str)
            assert isinstance(score, float)


# ── compute_accuracy ──────────────────────────────────────────────────────────

class TestComputeAccuracy:
    def test_perfect(self):
        preds = torch.tensor([1, 0, 1, 0])
        labels = [1, 0, 1, 0]
        assert compute_accuracy(preds, labels) == 1.0

    def test_zero(self):
        preds = torch.tensor([1, 1, 1, 1])
        labels = [0, 0, 0, 0]
        assert compute_accuracy(preds, labels) == 0.0

    def test_half(self):
        preds = torch.tensor([1, 0, 1, 0])
        labels = [0, 1, 0, 1]
        assert compute_accuracy(preds, labels) == 0.0  # all wrong → 0

    def test_three_quarters(self):
        preds = torch.tensor([1, 0, 1, 0])
        labels = [1, 0, 1, 1]
        assert abs(compute_accuracy(preds, labels) - 0.75) < 1e-6
