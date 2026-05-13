"""Stage 0 acceptance: toy model forward pass returns correct shape.

Per phases/phase2.md Stage 0 acceptance: "instantiates toy model, forwards
a random input, returns logits of expected shape".
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402


def test_toy_forward_shape() -> None:
    torch.manual_seed(42)
    model = ToyTransformer()
    B, T = 3, 4
    input_ids = torch.randint(0, model.vocab_size, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, model.vocab_size), (
        f"expected logits {(B, T, model.vocab_size)}, got {tuple(logits.shape)}"
    )


def test_toy_target_layer_resolves() -> None:
    """``config.layer_name`` must resolve on the toy model via the same
    ``get_submodule`` call used on GPT-Neo. Use layer 1 (the only valid index
    in a 2-layer toy) for this test, not the GPT-Neo default 9."""
    model = ToyTransformer()
    layer_name = "transformer.h.1.mlp.c_proj"
    layer = model.get_submodule(layer_name)
    assert isinstance(layer, torch.nn.Linear)
    assert layer.weight.shape == (model.d_model, model.d_mlp)


def test_toy_summary_runs() -> None:
    model = ToyTransformer()
    s = model.summary()
    assert "ToyTransformer" in s
    assert "transformer.h.0.mlp.c_proj.weight" in s


def test_config_defaults() -> None:
    cfg = EKFACConfig()
    assert cfg.toy_damping == 0.01
    assert cfg.damping_floor == 1e-5
    assert cfg.damping_alpha == 0.1
    assert cfg.ignore_index == -100
    assert cfg.seed == 42
