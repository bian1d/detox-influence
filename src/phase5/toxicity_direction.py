"""Broad toxicity direction g_f in the phi-subspace, from Lee toxicity_pairwise.

g_f = mean_i grad_phi log pi_theta*(toxic_i | prompt_i)
      - mean_i grad_phi log pi_theta*(nontoxic_i | prompt_i)      (baseline-subtracted)

The baseline subtraction cancels the "topic" component shared by the toxic and
non-toxic continuations of the same Wikitext-2 prompt, leaving the "toxicity"
direction (spec Section 3). A non-subtracted version (mean toxic grads only) is
also built for the thermometer contrast. Each per-sentence gradient uses the
same conventions as GPT-Neo: phi subspace, mean reduction, prompt mask, batch=1
(score_logpi_phi). Re-tokenises from the TEXT fields (the stored *_input_ids are
GPT-2 BPE, wrong for OLMo).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from phase4.grad_phi import score_logpi_phi
from phase5.config import LEE_DIR


@dataclass
class GFResult:
    g_f_subtracted: torch.Tensor       # mean(toxic) - mean(nontoxic)
    g_f_toxic_only: torch.Tensor       # mean(toxic)   (non-subtracted contrast)
    g_mean_nontoxic: torch.Tensor
    n_pairs: int
    norms: dict


def load_lee_pairs(n: int, *, split: str = "split_0.jsonl") -> list[dict]:
    """First ``n`` records with non-empty prompt / toxic / nontoxic text."""
    out: list[dict] = []
    with open(Path(LEE_DIR) / split) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("prompt_text") and rec.get("pert_gen_text") and rec.get("unpert_gen_text"):
                out.append(rec)
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} usable Lee pairs (< {n})")
    return out


def _ids(tokenizer, prompt: str, cont: str, *, p_max: int, r_max: int):
    p = tokenizer(prompt, return_tensors="pt")["input_ids"][0][:p_max]
    r = tokenizer(cont, return_tensors="pt", add_special_tokens=False)["input_ids"][0][:r_max]
    return p, r


def build_g_f(
    model: nn.Module,
    weight: torch.Tensor,
    tokenizer,
    pairs: list[dict],
    device: str,
    *,
    p_max: int = 32,
    r_max: int = 64,
    log_every: int = 25,
) -> GFResult:
    """Accumulate mean toxic and mean non-toxic phi-gradients over ``pairs``."""
    tox_acc = torch.zeros_like(weight, dtype=torch.float64)
    non_acc = torch.zeros_like(weight, dtype=torch.float64)
    n = 0
    for i, rec in enumerate(pairs):
        p_ids, tox_ids = _ids(tokenizer, rec["prompt_text"], rec["pert_gen_text"], p_max=p_max, r_max=r_max)
        _, non_ids = _ids(tokenizer, rec["prompt_text"], rec["unpert_gen_text"], p_max=p_max, r_max=r_max)
        if tox_ids.numel() < 2 or non_ids.numel() < 2:
            continue
        s_tox = score_logpi_phi(model, weight, p_ids, tox_ids, device).to(torch.float64)
        s_non = score_logpi_phi(model, weight, p_ids, non_ids, device).to(torch.float64)
        tox_acc += s_tox
        non_acc += s_non
        n += 1
        if (i + 1) % log_every == 0:
            print(f"    [g_f] {i + 1}/{len(pairs)} pairs")
    g_tox = tox_acc / n
    g_non = non_acc / n
    g_sub = g_tox - g_non
    norms = {
        "n_pairs": n,
        "norm_mean_toxic": float(g_tox.norm().item()),
        "norm_mean_nontoxic": float(g_non.norm().item()),
        "norm_subtracted": float(g_sub.norm().item()),
        "cos_tox_nontox": float(
            (g_tox * g_non).sum().item() / (g_tox.norm().item() * g_non.norm().item() + 1e-30)
        ),
    }
    return GFResult(g_sub, g_tox, g_non, n, norms)
