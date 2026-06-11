"""Block B-3 gate — robust 100-pair baseline-subtracted toxicity direction on
real GPT-Neo, per MLP layer. (GPU smoke; uses a modest pair count.)

Sanity that the baseline subtraction is meaningful:
  S1. toxic and non-toxic per-layer gradients are POSITIVELY correlated
      (they share the prompt's topic direction): cos(tox, nontox) > 0 for the
      large majority of layers; report the per-layer cosines.
  S2. subtraction REMOVES shared mass: ||g_subtracted|| < ||g_mean_toxic|| for
      the dominant layers (where the toxic signal actually lives).
  S3. the historical primary layer (transformer.h.9.mlp.c_proj) shows both.

Run (base env, GPU): python3 tests/test_robust_directions_gptneo.py [n_pairs]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from eval_directions import build_g_f_multi  # noqa: E402
from ekfac.multilayer import gptneo_mlp_layer_names, mlp_layer_dict  # noqa: E402
from phase5.toxicity_direction import load_lee_pairs  # noqa: E402

CKPT = Path("data/ppo_checkpoints/step_0650")
PRIMARY = "transformer.h.9.mlp.c_proj"


def main() -> int:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EKFACConfig()
    print(f"Block B-3: robust toxicity directions on GPT-Neo, {n_pairs} Lee pairs, dev {dev}")
    print(f"torch {torch.__version__}, numpy {np.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(CKPT.as_posix()).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(CKPT.as_posix())
    tok.pad_token = tok.eos_token
    names = gptneo_mlp_layer_names(12)
    layers = mlp_layer_dict(model, names)
    for p in model.parameters():
        p.requires_grad_(False)
    weights = {}
    for nm, layer in layers.items():
        layer.weight.requires_grad_(True)
        weights[nm] = layer.weight

    pairs = load_lee_pairs(n_pairs)
    gf = build_g_f_multi(model, weights, tok, pairs, cfg, dev, log_every=10)

    print(f"\n  built over {gf.n_pairs} pairs. Per-layer (sorted by toxic norm):")
    rows = sorted(gf.per_layer_norms.items(), key=lambda kv: -kv[1]["norm_mean_toxic"])
    n_pos_cos = 0
    n_sub_reduces = 0
    for nm, d in rows:
        cos = d["cos_tox_nontox"]
        reduces = d["norm_subtracted"] < d["norm_mean_toxic"]
        n_pos_cos += cos > 0
        n_sub_reduces += reduces
    # print top/bottom few
    for nm, d in rows[:6] + rows[-3:]:
        print(f"    {nm:34s} cos(tox,non)={d['cos_tox_nontox']:+.3f} "
              f"||tox||={d['norm_mean_toxic']:.3e} ||sub||={d['norm_subtracted']:.3e} "
              f"{'(sub<tox)' if d['norm_subtracted']<d['norm_mean_toxic'] else '(sub>=tox)'}")

    L = len(rows)
    prim = gf.per_layer_norms[PRIMARY]
    gates = {
        "S1 cos(tox,nontox)>0 for >=80% of layers": n_pos_cos >= 0.8 * L,
        "S2 subtraction reduces norm for >=60% of layers": n_sub_reduces >= 0.6 * L,
        "S3 primary layer9 c_proj: cos>0 and sub<tox":
            prim["cos_tox_nontox"] > 0 and prim["norm_subtracted"] < prim["norm_mean_toxic"],
    }
    print()
    for g, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {g}")
    print(f"  (pos-cos layers {n_pos_cos}/{L}, sub-reduces layers {n_sub_reduces}/{L}; "
          f"primary cos={prim['cos_tox_nontox']:+.3f}, ||sub||/||tox||="
          f"{prim['norm_subtracted']/prim['norm_mean_toxic']:.3f})")
    n_fail = sum(1 for ok in gates.values() if not ok)
    print(f"\ngates: {len(gates)-n_fail}/{len(gates)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
