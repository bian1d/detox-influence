"""OLMo-2-shaped toy transformer for re-validating the EK-FAC / Delta-VP gates
on a **gated** SwiGLU MLP (Phase 5 Section 2).

GPT-Neo's MLP is plain (``c_fc -> GELU -> c_proj``); OLMo-2's is gated SwiGLU
(``down_proj( silu(gate_proj(x)) * up_proj(x) )``). The phi subspace is the
final "back-to-hidden" linear in both cases — ``c_proj`` there, ``down_proj``
here. The gating nonlinearity sits *upstream* of down_proj, so the second-order
graph w.r.t. down_proj.weight should be structurally the same as the plain
case — but "should be" is exactly what the toy gates must verify, since the
double-backward now runs through silu*gate*up.

The module hierarchy keeps ``transformer.h.<i>.mlp.<name>`` so that a
``layer_name = "transformer.h.<i>.mlp.down_proj"`` resolves via
``model.get_submodule(layer_name)`` exactly like the real OLMo path
``model.layers.<i>.mlp.down_proj`` does on the 1B model.

Default size (d_model=8, d_mlp=16, n_layer=2) gives a target
``down_proj.weight`` of shape (8, 16) — Fisher 128x128, directly invertible —
matching ``ekfac.toy.ToyTransformer`` so the two toys are head-to-head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekfac.toy import ToyAttention


class ToyGatedMLP(nn.Module):
    """OLMo-2-style gated SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    bias=False on all three projections, matching OLMo-2's Olmo2MLP.
    """

    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyGatedBlock(nn.Module):
    """Pre-norm block with a gated MLP. (Norm *placement* is irrelevant to the
    down_proj double-backward we are validating; the gated MLP is the point.)"""

    def __init__(self, d_model: int, d_mlp: int, n_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = ToyAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = ToyGatedMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ToyGatedTransformer(nn.Module):
    """OLMo-2-shaped tiny transformer (gated SwiGLU MLP).

    ``forward(input_ids)`` returns next-token logits of shape (B, T, vocab).
    phi target: ``transformer.h.<i>.mlp.down_proj``.
    """

    def __init__(
        self,
        vocab_size: int = 20,
        d_model: int = 8,
        d_mlp: int = 16,
        n_layer: int = 2,
        n_heads: int = 2,
        max_len: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.max_len = max_len

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(vocab_size, d_model),
                "wpe": nn.Embedding(max_len, d_model),
                "h": nn.ModuleList(
                    [ToyGatedBlock(d_model, d_mlp, n_heads) for _ in range(n_layer)]
                ),
                "ln_f": nn.LayerNorm(d_model),
            }
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len={self.max_len}")
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)
