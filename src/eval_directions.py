"""Robust 100-pair averaged toxicity eval directions for GPT-Neo, per MLP layer.

Replaces the old single-template ``f_toxic`` (one hand-written (x_eval, y_toxic)
pair, high variance) with the baseline-subtracted average over ~100 Lee
toxic/non-toxic continuation pairs of the SAME prompt:

    g_f[layer] = mean_i grad_W[layer] log π(toxic_i | prompt_i)
               - mean_i grad_W[layer] log π(nontoxic_i | prompt_i)

The subtraction cancels the topic component shared by a prompt's toxic and
non-toxic continuations, leaving the toxicity direction (Lee et al.; the same
construction Phase 5 uses for OLMo, here ported to GPT-Neo and generalised to
all MLP layers in one backward via ``per_sample_grads_multi``). A non-subtracted
"toxic-only" version is kept for contrast. The Lee data fields are text, so the
GPT-Neo tokenizer re-tokenises them correctly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ekfac.config import EKFACConfig
from ekfac.data import Rollout
from ekfac.multilayer import per_sample_grads_multi


@dataclass
class MultiLayerGF:
    g_subtracted: dict[str, torch.Tensor]   # layer -> mean(toxic) - mean(nontoxic)
    g_toxic_only: dict[str, torch.Tensor]   # layer -> mean(toxic)
    g_nontoxic: dict[str, torch.Tensor]     # layer -> mean(nontoxic)
    n_pairs: int
    per_layer_norms: dict[str, dict]        # layer -> {norm_subtracted, norm_mean_toxic, cos_tox_nontox}


def _ids(tokenizer, prompt: str, cont: str, *, p_max: int, r_max: int):
    p = tokenizer(prompt, return_tensors="pt")["input_ids"][0][:p_max]
    r = tokenizer(cont, return_tensors="pt", add_special_tokens=False)["input_ids"][0][:r_max]
    return p, r


def build_g_f_multi(
    model: nn.Module,
    layer_weights: dict[str, torch.Tensor],
    tokenizer,
    pairs: list[dict],
    cfg: EKFACConfig,
    device: torch.device,
    *,
    p_max: int = 32,
    r_max: int = 64,
    log_every: int = 25,
) -> MultiLayerGF:
    """Per-layer baseline-subtracted toxicity direction over ``pairs``.

    Each pair contributes one toxic and one non-toxic per-sample gradient (full
    autograd over every layer weight, mean reduction, prompt mask), accumulated
    per layer. Returns the subtracted and toxic-only directions plus a
    per-layer sanity dict.
    """
    names = list(layer_weights)
    tox_acc = {n: torch.zeros_like(layer_weights[n], dtype=torch.float64) for n in names}
    non_acc = {n: torch.zeros_like(layer_weights[n], dtype=torch.float64) for n in names}
    n = 0
    for i, rec in enumerate(pairs):
        p_ids, tox_ids = _ids(tokenizer, rec["prompt_text"], rec["pert_gen_text"], p_max=p_max, r_max=r_max)
        _, non_ids = _ids(tokenizer, rec["prompt_text"], rec["unpert_gen_text"], p_max=p_max, r_max=r_max)
        if tox_ids.numel() < 2 or non_ids.numel() < 2:
            continue
        r_tox = Rollout(prompt_ids=p_ids, response_ids=tox_ids, reward=0.0, step=0)
        r_non = Rollout(prompt_ids=p_ids, response_ids=non_ids, reward=0.0, step=0)
        s_tox = per_sample_grads_multi(model, layer_weights, r_tox, cfg, device=device)
        s_non = per_sample_grads_multi(model, layer_weights, r_non, cfg, device=device)
        for nm in names:
            tox_acc[nm] += s_tox[nm].double()
            non_acc[nm] += s_non[nm].double()
        n += 1
        if (i + 1) % log_every == 0:
            print(f"    [g_f multi] {i + 1}/{len(pairs)} pairs")
    if n == 0:
        raise RuntimeError("no usable Lee pairs")

    g_tox = {nm: tox_acc[nm] / n for nm in names}
    g_non = {nm: non_acc[nm] / n for nm in names}
    g_sub = {nm: g_tox[nm] - g_non[nm] for nm in names}
    norms = {}
    for nm in names:
        nt = float(g_tox[nm].norm().item())
        nn_ = float(g_non[nm].norm().item())
        ns = float(g_sub[nm].norm().item())
        cos = float((g_tox[nm] * g_non[nm]).sum().item() / (nt * nn_ + 1e-30))
        norms[nm] = {"norm_mean_toxic": nt, "norm_mean_nontoxic": nn_,
                     "norm_subtracted": ns, "cos_tox_nontox": cos}
    return MultiLayerGF(g_sub, g_tox, g_non, n, norms)
