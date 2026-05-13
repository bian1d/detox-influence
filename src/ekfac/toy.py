"""Toy transformer for EK-FAC numerical correctness validation.

The MLP block matches GPT-Neo's structure (``c_fc → GELU → c_proj``) and
the module hierarchy uses the same names, so a config ``layer_name =
"transformer.h.<i>.mlp.c_proj"`` resolves identically on the toy model and
on the real GPT-Neo-125M via ``model.get_submodule(layer_name)``.

Default size (d_model=8, d_mlp=16, n_layer=2, vocab=20, T_max=4) gives a
target ``c_proj.weight`` of shape (8, 16) — the full Fisher matrix has
dimensions 128×128, small enough to invert directly with ``torch.linalg.inv``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    """``c_fc → GELU → c_proj`` matching GPT-Neo's ``GPTNeoMLP``."""

    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_mlp)
        self.c_proj = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class ToyAttention(nn.Module):
    """Minimal multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.d_head)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, D)
        return self.out(out)


class ToyBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, n_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = ToyAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = ToyMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ToyTransformer(nn.Module):
    """GPT-Neo-shaped tiny transformer.

    ``forward(input_ids)`` returns next-token logits of shape (B, T, vocab).
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
                    [ToyBlock(d_model, d_mlp, n_heads) for _ in range(n_layer)]
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

    def summary(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        lines = [
            f"ToyTransformer: vocab={self.vocab_size}, d_model={self.d_model}, "
            f"d_mlp={self.d_mlp}, n_layer={self.n_layer}, n_heads={self.n_heads}, "
            f"max_len={self.max_len}",
            f"Total params: {n_params}",
            "Parameter shapes:",
        ]
        for name, p in self.named_parameters():
            lines.append(f"  {name:<45s} {tuple(p.shape)} = {p.numel()}")
        return "\n".join(lines)
