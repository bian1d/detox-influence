"""Forward/backward hooks for EK-FAC factor capture on one ``nn.Linear``.

For a linear layer ``out = W @ m`` (PyTorch ``nn.Linear`` stores ``W`` of
shape ``(d_out, d_in)``), EK-FAC needs:

    m     :  input to the layer            (..., d_in)
    delta :  grad of loss w.r.t. ``out``   (..., d_out)

For c_proj (the MLP value matrix W₂): ``d_in = d_mlp``, ``d_out = d_model``.
The per-token outer product ``delta_t ⊗ m_t`` is the per-token contribution
to ``∇_W loss`` (CONTEXT.md §12).

Tensors are detached and cloned in the hooks: the forward input and the
gradient passing through the layer are normally freed once backward
completes, and ``.detach()`` alone would leave a dangling view into freed
storage. ``.clone()`` materialises an independent buffer that survives.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn


@dataclass
class CProjCache:
    """Captures m and δ for the layer enclosed by ``capture_c_proj``."""

    m: torch.Tensor | None = None       # (..., d_in)
    delta: torch.Tensor | None = None   # (..., d_out)

    def clear(self) -> None:
        self.m = None
        self.delta = None


@contextmanager
def capture_c_proj(layer: nn.Linear) -> Iterator[CProjCache]:
    """Context manager that records ``m`` and ``δ`` for ``layer``.

    After a forward+backward call inside the context, ``cache.m`` holds the
    layer's input and ``cache.delta`` holds the gradient at the layer's
    output. Both are detached clones, safe to read after backward.

    Hooks are removed on exit (also on exception).
    """
    cache = CProjCache()

    def forward_hook(
        _module: nn.Module,
        args: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        cache.m = args[0].detach().clone()

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        go = grad_output[0]
        if go is None:
            raise RuntimeError("backward hook received grad_output[0] = None")
        cache.delta = go.detach().clone()

    fwd_handle = layer.register_forward_hook(forward_hook)
    bwd_handle = layer.register_full_backward_hook(backward_hook)
    try:
        yield cache
    finally:
        fwd_handle.remove()
        bwd_handle.remove()
