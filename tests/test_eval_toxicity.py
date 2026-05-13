"""
Unit tests for src/eval_toxicity.py.

Tests for prompt I/O and continuation extraction — no RealToxicityPrompts
download, no evaluate.load call. Uses a tiny GPT2 model for generation tests.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eval_toxicity import generate_continuations, load_eval_prompts


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    config = GPT2Config(
        vocab_size=50257, n_embd=64, n_layer=2, n_head=2, n_inner=128,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


# ── generate_continuations ────────────────────────────────────────────────────

class TestGenerateContinuations:
    def test_returns_correct_count(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        prompts = ["hello world", "the cat sat", "once upon a time"]
        conts = generate_continuations(model, tokenizer, prompts, max_new_tokens=10,
                                        seed=42, device="cpu", batch_size=2)
        assert len(conts) == 3

    def test_continuation_not_prompt(self, tiny_model_and_tokenizer):
        """
        The returned continuation must NOT contain the full prompt text.
        This is the core invariant: score only the new tokens.
        """
        model, tokenizer = tiny_model_and_tokenizer
        prompts = ["hello world this is a test prompt for verification"]
        conts = generate_continuations(model, tokenizer, prompts, max_new_tokens=15,
                                        seed=42, device="cpu", batch_size=1)
        assert len(conts) == 1
        # The prompt should not appear verbatim in the continuation
        assert prompts[0].strip() not in conts[0], (
            f"Prompt leaked into continuation: {repr(conts[0])}"
        )

    def test_max_new_tokens_respected(self, tiny_model_and_tokenizer):
        """Each continuation should be ≤ max_new_tokens tokens."""
        model, tokenizer = tiny_model_and_tokenizer
        prompts = ["short", "another short prompt", "yet another"]
        max_new = 15
        conts = generate_continuations(model, tokenizer, prompts, max_new_tokens=max_new,
                                        seed=42, device="cpu", batch_size=3)
        for cont in conts:
            n_tokens = len(tokenizer(cont, add_special_tokens=False)["input_ids"])
            assert n_tokens <= max_new, (
                f"Continuation has {n_tokens} tokens, max was {max_new}: {repr(cont)}"
            )

    def test_seed_reproducibility(self, tiny_model_and_tokenizer):
        """Same seed → same continuations."""
        model, tokenizer = tiny_model_and_tokenizer
        prompts = ["the quick brown fox"]
        conts1 = generate_continuations(model, tokenizer, prompts, max_new_tokens=10,
                                         seed=7, device="cpu", batch_size=1)
        conts2 = generate_continuations(model, tokenizer, prompts, max_new_tokens=10,
                                         seed=7, device="cpu", batch_size=1)
        assert conts1 == conts2

    def test_continuations_are_strings(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        conts = generate_continuations(model, tokenizer, ["test"], max_new_tokens=5,
                                        seed=0, device="cpu", batch_size=1)
        assert all(isinstance(c, str) for c in conts)

    def test_batch_vs_single_same_output(self, tiny_model_and_tokenizer):
        """
        With seed reset each call, batch_size=1 and batch_size=N produce the
        same output only if seed resets to the same state. We test that batched
        processing doesn't silently truncate or skip prompts.
        """
        model, tokenizer = tiny_model_and_tokenizer
        prompts = ["alpha", "beta gamma", "delta"]
        conts = generate_continuations(model, tokenizer, prompts, max_new_tokens=8,
                                        seed=42, device="cpu", batch_size=3)
        assert len(conts) == 3
        for c in conts:
            assert isinstance(c, str)


# ── prompt I/O ────────────────────────────────────────────────────────────────

class TestPromptIO:
    def test_load_eval_prompts_roundtrip(self, tmp_path):
        """save then load returns identical data."""
        indices = [0, 5, 100, 999]
        texts = ["prompt one", "prompt two", "prompt three", "prompt four"]
        out = tmp_path / "eval_prompts.json"
        with open(out, "w") as f:
            json.dump({"indices": indices, "prompt_texts": texts}, f)

        loaded_texts, loaded_indices = load_eval_prompts(out)
        assert loaded_texts == texts
        assert loaded_indices == indices

    def test_load_eval_prompts_schema(self, tmp_path):
        """File must have 'indices' and 'prompt_texts' keys."""
        out = tmp_path / "bad.json"
        with open(out, "w") as f:
            json.dump({"prompts": ["a", "b"]}, f)   # missing 'indices' key
        with pytest.raises(KeyError):
            load_eval_prompts(out)
