"""Phase 5 Stage 1 — thermometer ||Delta~ p_f|| / ||g_f|| on OLMo-2-1B-SFT.

Reuses the Phase-4 E3 machinery unchanged (delta_vp_per_prompt double-backward,
U-statistic ||.||^2, two-half cosine); only the model, phi=down_proj, the OLMo
EK-FAC factors, and the broad-toxicity g_f (from Lee pairs) differ.

  p_f   = F^-1 g_f                       (inverse_hvp with the OLMo factors)
  Delta~ p_f over a 200x16 on-policy pool at theta* (advantage A = R~ - mean_y R~)
  thermometer = ||Delta~ p_f|| / ||g_f||,  two-half cosine for convergence.

Both g_f versions: baseline-subtracted (mean toxic - mean nontoxic) and
toxic-only (non-subtracted contrast).

    /root/miniconda3/envs/olmo/bin/python src/run_phase5_stage1_thermometer.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.eigen import inverse_hvp  # noqa: E402
from phase4.e3_delta import DeltaSample, delta_vp_per_prompt  # noqa: E402
from phase4.grad_phi import get_phi_weight  # noqa: E402
from phase4.sampling import sample_on_policy_pool  # noqa: E402
from phase5.config import (  # noqa: E402
    BETA_NOMINAL, EVAL_PROMPTS_PATH, GPTNEO_BASELINE, OUT_DIR, PHI_LAYER, SEED,
)
from phase5.models_olmo import load_olmo_models  # noqa: E402
from phase5.toxicity_direction import build_g_f, load_lee_pairs  # noqa: E402

FACTOR_DIR = OUT_DIR / "factors"


def _pick_prompts(n: int, seed: int) -> list[str]:
    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    allp = ep["prompt_texts"]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(allp), size=n, replace=False).tolist())
    return [allp[i] for i in idx]


def _build_delta_samples(pool) -> list[DeltaSample]:
    by_prompt: dict[int, list] = defaultdict(list)
    for s in pool:
        by_prompt[s.prompt_idx].append(s)
    out: list[DeltaSample] = []
    for samples in by_prompt.values():
        mean_R = float(np.mean([s.R_tilde for s in samples]))
        for s in samples:
            out.append(DeltaSample(prompt_idx=s.prompt_idx, prompt_ids=s.prompt_ids,
                                   response_ids=s.response_ids, A=float(s.R_tilde - mean_R)))
    return out


def _streaming_stats(d_p: torch.Tensor) -> dict:
    """U-stat ||mean d_p||^2 + jackknife SE + naive + two-half cosine WITHOUT
    materialising the (N, D) fp64 matrix.

    OLMo's phi is 16.78M params, so (200, 16.78M) fp64 = 26.8 GB and the
    Phase-4 jackknife_ci/_flat path OOMs; but for the identity metric every
    quantity it uses reduces to S = sum_p d_p, <S,S>, <d_p,d_p> and <d_p,S>,
    all computable by streaming one 134 MB fp64 prompt-vector at a time.
    Closed-form identical to phase4.e2_stationarity.jackknife_ci (M = I):
    there SMgq = gqMS = <d_q, S>, so cross_q = cross - 2<d_q,S> + <d_q,d_q>.
    Gated by tests/diag_streaming_stats_check.py against jackknife_ci.
    """
    N = d_p.shape[0]
    half = N // 2
    a = torch.zeros(d_p.shape[1:], dtype=torch.float64, device=d_p.device)
    b = torch.zeros_like(a)
    for p in range(N):                       # pass 1: half-sums (S = a + b)
        (a if p < half else b).add_(d_p[p].to(torch.float64))
    S = a + b
    cos_half = float(((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item())

    diag_each = np.empty(N)
    gS = np.empty(N)
    for p in range(N):                       # pass 2: per-prompt reductions
        gp = d_p[p].to(torch.float64)
        diag_each[p] = float((gp * gp).sum().item())
        gS[p] = float((gp * S).sum().item())
    cross_all = float((S * S).sum().item())
    diag_all = float(diag_each.sum())

    U_full = (cross_all - diag_all) / (N * (N - 1))
    m = N - 1
    loo = (cross_all - 2.0 * gS + diag_each - (diag_all - diag_each)) / (m * (m - 1))
    var = (N - 1) / N * float(((loo - loo.mean()) ** 2).sum())
    return {
        "point": U_full,
        "jackknife_se": float(np.sqrt(max(var, 0.0))),
        "naive": cross_all / (N * N),
        "two_half_cosine": cos_half,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--tag", type=str, default="", help="output filename suffix")
    args = ap.parse_args()
    device = "cuda"
    n_prompts = args.n_prompts or (4 if args.smoke else 200)
    k = args.k or (4 if args.smoke else 16)
    n_pairs = 8 if args.smoke else 100
    cfg = EKFACConfig(layer_name=PHI_LAYER)

    print("loading OLMo models (eager attn for double-backward) ...")
    models = load_olmo_models(device, policy_requires_grad=False, policy_attn_implementation="eager")
    weight = get_phi_weight(models.policy, PHI_LAYER)
    weight.requires_grad_(True)

    print("loading OLMo EK-FAC factors ...")
    Q_A = torch.load(FACTOR_DIR / "Q_A.pt", map_location=device).to(torch.float64)
    Q_S = torch.load(FACTOR_DIR / "Q_S.pt", map_location=device).to(torch.float64)
    Lambda = torch.load(FACTOR_DIR / "Lambda.pt", map_location=device).to(torch.float64)

    print(f"building g_f from {n_pairs} Lee pairs (subtracted + toxic-only) ...")
    pairs = load_lee_pairs(n_pairs)
    gf = build_g_f(models.policy, weight, models.tokenizer, pairs, device)
    print(f"  g_f norms: subtracted={gf.norms['norm_subtracted']:.4e} "
          f"toxic-only={gf.norms['norm_mean_toxic']:.4e} cos(tox,nontox)={gf.norms['cos_tox_nontox']:.3f}")

    g_versions = {"subtracted": gf.g_f_subtracted.to(device), "toxic_only": gf.g_f_toxic_only.to(device)}
    p_versions = {name: inverse_hvp(g, Q_A, Q_S, Lambda,
                                    damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha)
                  for name, g in g_versions.items()}

    print(f"sampling on-policy pool {n_prompts}x{k} at theta* (beta={BETA_NOMINAL}) ...")
    prompts = _pick_prompts(n_prompts, SEED)
    pool = sample_on_policy_pool(prompts, k, seed_base=SEED, models=models, beta=BETA_NOMINAL, device=device)
    samples = _build_delta_samples(pool)
    print(f"  pool: {len(pool)} samples over {n_prompts} prompts")

    names = list(g_versions)
    vs = [p_versions[n] for n in names]
    print("computing Delta~ p_f (matrix-free double-backward) ...")
    # store_device="cpu": (N_p, 2048, 8192) accumulators are 25 GiB each at
    # N_p=400 — two targets exceed the GPU; accumulate off-GPU instead.
    d_p, order, _diag = delta_vp_per_prompt(samples, models.policy, weight, BETA_NOMINAL, vs, device,
                                            store_device="cpu")

    results = {"model": "OLMo-2-0425-1B-SFT", "beta": BETA_NOMINAL, "n_prompts": n_prompts,
               "k_per_prompt": k, "n_lee_pairs": gf.n_pairs, "g_f_norms": gf.norms,
               "gptneo_baseline": GPTNEO_BASELINE, "targets": {}}
    print("\n" + "=" * 70)
    for idx, name in enumerate(names):
        st = _streaming_stats(d_p[idx])
        dp_norm = float(np.sqrt(max(st["point"], 0.0)))
        gf_norm = gf.norms["norm_subtracted" if name == "subtracted" else "norm_mean_toxic"]
        therm = dp_norm / gf_norm
        results["targets"][name] = {
            "g_f_norm": gf_norm, "delta_p_norm": dp_norm,
            "delta_p_norm_sq_ustat": st["point"], "delta_p_norm_sq_jackknife_se": st["jackknife_se"],
            "delta_p_norm_sq_naive": st["naive"],
            "thermometer_ratio": therm, "two_half_cosine": st["two_half_cosine"],
        }
        print(f"  [{name}] ||Delta~ p||={dp_norm:.4e}  ||g_f||={gf_norm:.4e}  "
              f"THERMOMETER={therm:.4f}  (two-half cos={st['two_half_cosine']:.3f}, "
              f"jackSE={st['jackknife_se']:.2e})")

    thermos = {n: results["targets"][n]["thermometer_ratio"] for n in names}
    max_t = max(thermos.values())
    verdict = ("DEATH (>0.3: A2-failure persists on SFT)" if max_t > 0.3
               else "LIFE (<0.1: A2 holds on SFT)" if max_t < 0.1
               else "MIDBAND (0.1-0.3)")
    results["max_thermometer"] = max_t
    results["thermometer_verdict"] = verdict
    print(f"\n  thermometers: " + ", ".join(f"{n}={thermos[n]:.4f}" for n in names))
    print(f"  verdict (thermometer only): {verdict}")
    print("  NB: read with the ICC reward/KL decomposition to confirm the source.")

    out = OUT_DIR / ("stage1_thermometer_smoke.json" if args.smoke
                     else f"stage1_thermometer{args.tag}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
