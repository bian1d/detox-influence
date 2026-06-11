"""Block B-1 gate — multi-layer EK-FAC == per-layer single-layer, element-wise.

The whole point of the multi-layer machinery is that hooking N MLP layers at
once and accumulating each must give EXACTLY what the validated single-layer
pipeline gives for each layer run alone. If hooks cross-talked (one layer's
capture leaking into another), or the pseudo-label draw order differed, the
factors would diverge. Both paths use the same generator seeding (cfg.seed for
A/S, cfg.seed+1 for Λ) and the same row-aligned helper, and the hooks are pure
observers, so the match must be at machine precision (bit-identical up to fp
summation order, which here is the same).

Gates (toy transformer with 4 MLP linears: 2 blocks x {c_fc=W_1, c_proj=W_2}):
  G1. accumulate_AS_multi[L].A == accumulate_AS(L).A   (and S)  per layer
  G2. fit_lambda_multi[L]      == fit_lambda(L)                  per layer
  G3. per_sample_grads_multi[L] == compute_per_sample_grad(L)    per layer
  G4. covers BOTH orientations (c_fc 16x8 and c_proj 8x16) so the W_1/W_2
      transpose conventions are exercised.

Run:  python3 tests/test_multilayer_consistency.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.multilayer import (  # noqa: E402
    accumulate_AS_multi,
    fit_lambda_multi,
    gptneo_mlp_layer_names,
    mlp_layer_dict,
    per_sample_grads_multi,
)
from ekfac.toy import ToyTransformer  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402

DEV = torch.device("cpu")
LAYER_NAMES = [
    "transformer.h.0.mlp.c_fc", "transformer.h.0.mlp.c_proj",
    "transformer.h.1.mlp.c_fc", "transformer.h.1.mlp.c_proj",
]
RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _rollouts(n: int, model: ToyTransformer, seed: int) -> list[Rollout]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        T = model.max_len
        P = int(torch.randint(1, T, (1,), generator=g).item())
        ids = torch.randint(0, model.vocab_size, (T,), generator=g, dtype=torch.long)
        out.append(Rollout(prompt_ids=ids[:P], response_ids=ids[P:], reward=0.0, step=0))
    return out


def main() -> int:
    print("Block B-1: multi-layer EK-FAC == per-layer single-layer (no hook cross-talk)")
    print(f"torch {torch.__version__}")
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer().eval().double()
    layers = mlp_layer_dict(model, LAYER_NAMES)
    rolls = _rollouts(120, model, seed=321)

    # ---- multi-layer factors (one pass) ----
    multi = accumulate_AS_multi(model, layers, rolls, cfg, device=DEV)
    Qs = {name: (eigendecompose(A, S)[0], eigendecompose(A, S)[2])
          for name, (A, S, _) in multi.items()}
    Lam_multi = fit_lambda_multi(model, layers, rolls, cfg, Qs=Qs, device=DEV)

    # ---- per-layer single-layer factors, compare element-wise ----
    worst_A = worst_S = worst_L = 0.0
    for name in LAYER_NAMES:
        layer = layers[name]
        A_s, S_s, n_s = accumulate_AS(model, layer, rolls, cfg, device=DEV)
        A_m, S_m, n_m = multi[name]
        dA = (A_m - A_s).abs().max().item()
        dS = (S_m - S_s).abs().max().item()
        worst_A = max(worst_A, dA); worst_S = max(worst_S, dS)
        assert n_s == n_m
        # Λ uses the SAME Q from single-layer eigendecompose for an apples
        # comparison (eigh sign/order is deterministic for the same A,S).
        Q_A_s, _, Q_S_s, _ = eigendecompose(A_s, S_s)
        Lam_s, _ = fit_lambda(model, layer, rolls, cfg, Q_A=Q_A_s, Q_S=Q_S_s, device=DEV)
        Lam_m, _ = Lam_multi[name]
        dL = (Lam_m - Lam_s).abs().max().item()
        worst_L = max(worst_L, dL)
        gate(f"factors[{name}] multi==single", dA < 1e-12 and dS < 1e-12 and dL < 1e-12,
             f"max|ΔA|={dA:.1e} max|ΔS|={dS:.1e} max|ΔΛ|={dL:.1e}")

    # ---- scores: multi vs single, element-wise ----
    weights = {name: layers[name].weight for name in LAYER_NAMES}
    worst_g = 0.0
    for r in rolls[:10]:
        gm = per_sample_grads_multi(model, weights, r, cfg, device=DEV)
        for name in LAYER_NAMES:
            gs = compute_per_sample_grad(model, layers[name], r, cfg, device=DEV)
            d = (gm[name] - gs).abs().max().item()
            worst_g = max(worst_g, d)
    gate("scores multi==single (all layers, 10 rollouts)", worst_g < 1e-12,
         f"max|Δs_m| = {worst_g:.1e}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\nworst over all layers: ΔA={worst_A:.1e} ΔS={worst_S:.1e} "
          f"ΔΛ={worst_L:.1e} Δscore={worst_g:.1e}")
    print(f"gates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


# pytest wrappers
def test_multilayer_factors_match_single() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
