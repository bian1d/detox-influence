"""Block B-2 gate — cache-free streaming influence == direct reference.

The new influence path recomputes each rollout's per-layer score on the fly and
reduces against every target's precomputed per-layer left-vector, summing over
the MLP block-diagonal subspace — NO 184 GB g_scaled_z cache. This gate proves
the streaming reduction equals the obvious (slow, explicit) reference computed
one layer / one rollout / one target at a time with the validated single-layer
primitives.

Gates (toy transformer, 4 MLP linears, 2 eval targets):
  G1. influence_streaming[target][m] == sign * Σ_layer <p[layer], s_m[layer]>,
      where s_m[layer] = compute_per_sample_grad(layer) and the layer sum uses
      influence_score_klrl — machine precision, per target, all rollouts.
  G2. KL-RL sign (-1) and supervised sign (+1) both handled.
  G3. multi-target single pass returns the same as scoring each target alone.

Run:  python3 tests/test_streaming_influence.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda, inverse_hvp  # noqa: E402
from ekfac.influence import influence_score_klrl, influence_score_supervised  # noqa: E402
from ekfac.multilayer import (  # noqa: E402
    accumulate_AS_multi,
    influence_streaming,
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


def gate(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _rollouts(n, model, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        T = model.max_len
        P = int(torch.randint(1, T, (1,), generator=g).item())
        ids = torch.randint(0, model.vocab_size, (T,), generator=g, dtype=torch.long)
        out.append(Rollout(prompt_ids=ids[:P], response_ids=ids[P:], reward=0.0, step=0))
    return out


def main() -> int:
    print("Block B-2: cache-free streaming influence == direct reference")
    print(f"torch {torch.__version__}")
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer().eval().double()
    layers = mlp_layer_dict(model, LAYER_NAMES)
    weights = {n: layers[n].weight for n in LAYER_NAMES}
    rolls = _rollouts(60, model, seed=321)
    evals = _rollouts(2, model, seed=999)

    # per-layer factors + per-layer p_f for two eval targets (the two eval grads)
    multi = accumulate_AS_multi(model, layers, rolls, cfg, device=DEV)
    QLam = {}
    for name, (A, S, _) in multi.items():
        Q_A, _, Q_S, _ = eigendecompose(A, S)
        Lam, _ = fit_lambda(model, layers[name], rolls, cfg, Q_A=Q_A, Q_S=Q_S, device=DEV)
        QLam[name] = (Q_A.double(), Q_S.double(), Lam.double())

    targets = {}
    for t, ev in enumerate([f"tgt{t}" for t in range(len(evals))]):
        pass
    target_names = [f"tgt{t}" for t in range(len(evals))]
    p_per_target: dict[str, dict[str, torch.Tensor]] = {}
    for t, ev in enumerate(evals):
        g_ev = per_sample_grads_multi(model, weights, ev, cfg, device=DEV)
        p_layer = {}
        for name in LAYER_NAMES:
            Q_A, Q_S, Lam = QLam[name]
            p_layer[name] = inverse_hvp(
                g_ev[name].double(), Q_A, Q_S, Lam,
                damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha,
            )
        p_per_target[target_names[t]] = p_layer

    # ---- streaming (KL-RL sign) ----
    I_stream = influence_streaming(model, weights, rolls, p_per_target, cfg,
                                   device=DEV, sign=-1.0, log_every=0)

    # ---- direct reference: per layer, per rollout, summed ----
    worst = 0.0
    for tn in target_names:
        I_ref = np.empty(len(rolls))
        for m, r in enumerate(rolls):
            acc = 0.0
            for name in LAYER_NAMES:
                s_m = compute_per_sample_grad(model, layers[name], r, cfg, device=DEV).double()
                acc += influence_score_klrl(s_m, p_per_target[tn][name])  # sums -<p,s>
            I_ref[m] = acc
        d = float(np.max(np.abs(I_stream[tn].numpy() - I_ref)))
        rel = d / (np.max(np.abs(I_ref)) + 1e-30)
        worst = max(worst, rel)
        gate(f"streaming == direct reference [{tn}]", rel < 1e-12,
             f"max abs diff = {d:.1e} (rel {rel:.1e})")

    # ---- supervised sign sanity (+1) ----
    I_sup = influence_streaming(model, weights, rolls, {"t0": p_per_target[target_names[0]]},
                                cfg, device=DEV, sign=+1.0, log_every=0)
    I_sup_ref = np.empty(len(rolls))
    for m, r in enumerate(rolls):
        acc = 0.0
        for name in LAYER_NAMES:
            s_m = compute_per_sample_grad(model, layers[name], r, cfg, device=DEV).double()
            acc += influence_score_supervised(s_m, p_per_target[target_names[0]][name])
        I_sup_ref[m] = acc
    d_sup = float(np.max(np.abs(I_sup["t0"].numpy() - I_sup_ref)))
    gate("supervised sign (+1) handled", d_sup < 1e-12, f"max abs diff = {d_sup:.1e}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\ngates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed  (worst rel {worst:.1e})")
    return 1 if n_fail else 0


def test_streaming_matches_direct() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
