"""Phase 4 E2 driver: ||G(theta*)|| first-order non-stationarity.

Samples N=256 prompts x K=8 on-policy rollouts at theta*, computes the
split-half unbiased ||G||^2 and natural-gradient step 0.5*G^T F^-1 G (reusing
the cached EK-FAC factors), and compares the step KL to the training per-step
KL.  Two baselines (R_tilde-beta primary, LOO-advantage cross-check).

Output: data/phase4/e2_stationarity.json, data/phase4/e2_onpolicy_pool.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from phase4.config import (  # noqa: E402
    BETA,
    EVAL_PROMPTS_PATH,
    PHASE4_DATA_DIR,
    SEED,
)
from phase4.e2_stationarity import stationarity_report  # noqa: E402
from phase4.grad_phi import get_phi_weight  # noqa: E402
from phase4.models import load_models  # noqa: E402
from phase4.sampling import sample_on_policy_pool  # noqa: E402

EKFAC_RUN = Path(__file__).resolve().parent.parent / "data" / "ekfac" / "influence" / "run_step0650"
PPO_LOG = Path(__file__).resolve().parent.parent / "data" / "ppo_train_log.jsonl"

N_PROMPTS = 256
K_PER_PROMPT = 8
N_SPLITS = 25
SPLIT_SEED = 1000

# Smoke config: tiny pool to validate the GPU pipeline end-to-end.
SMOKE = {"n_prompts": 4, "k": 4, "n_splits": 5}


def _pick_prompts(n: int, seed: int) -> list[str]:
    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    allp = ep["prompt_texts"]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(allp), size=n, replace=False).tolist())
    return [allp[i] for i in idx]


def _training_kl_comparators() -> dict:
    rows = [json.loads(l) for l in open(PPO_LOG)]
    near = [r for r in rows if 600 <= r.get("step", -1) <= 700]
    approxkl = np.array([r["approxkl"] for r in near if "approxkl" in r])
    r650 = next(r for r in rows if r.get("step") == 650)
    return {
        "approxkl_per_step_mean_600_700": float(approxkl.mean()),
        "approxkl_per_step_at_650": float(r650["approxkl"]),
        "k1_kl_policy_vs_ref_at_650": float(r650["k1_kl_naive"]),
        "adaptive_kl_target": 6.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 E2")
    ap.add_argument("--smoke", action="store_true", help="tiny pool, validate pipeline")
    args = ap.parse_args()
    n_prompts = SMOKE["n_prompts"] if args.smoke else N_PROMPTS
    k_per = SMOKE["k"] if args.smoke else K_PER_PROMPT
    n_splits = SMOKE["n_splits"] if args.smoke else N_SPLITS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    PHASE4_DATA_DIR.mkdir(parents=True, exist_ok=True)

    prompts = _pick_prompts(n_prompts, seed=SEED)
    print(f"E2{'(smoke)' if args.smoke else ''}: {n_prompts} prompts x {k_per} "
          f"on-policy rollouts at theta*, beta={BETA:.17g}")

    print("Loading models ...")
    models = load_models(device, policy_requires_grad=False)
    weight = get_phi_weight(models.policy)
    weight.requires_grad_(True)
    print(f"phi = transformer.h.9.mlp.c_proj.weight, shape {tuple(weight.shape)}")

    print("Sampling on-policy pool ...")
    pool = sample_on_policy_pool(
        prompts, k_per, seed_base=SEED, models=models, beta=BETA, device=device
    )
    print(f"pool size: {len(pool)} samples")

    suffix = "_smoke" if args.smoke else ""
    # Persist the lightweight pool (token ids + scalars) for reproducibility.
    pool_path = PHASE4_DATA_DIR / f"e2_onpolicy_pool{suffix}.jsonl"
    with open(pool_path, "w") as f:
        for s in pool:
            f.write(json.dumps({
                "prompt_idx": s.prompt_idx,
                "prompt_ids": s.prompt_ids.tolist(),
                "response_ids": s.response_ids.tolist(),
                "reward": s.reward, "k1_kl": s.k1_kl, "R_tilde": s.R_tilde,
            }) + "\n")
    print(f"wrote {pool_path}")

    # Cached EK-FAC factors -> device, fp64.
    print("Loading EK-FAC factors ...")
    Q_A = torch.load(EKFAC_RUN / "Q_A.pt", map_location=device).to(torch.float64)
    Q_S = torch.load(EKFAC_RUN / "Q_S.pt", map_location=device).to(torch.float64)
    Lambda = torch.load(EKFAC_RUN / "Lambda.pt", map_location=device).to(torch.float64)

    print("Computing split-half ||G||^2 and G^T F^-1 G ...")
    res = stationarity_report(
        pool, models.policy, weight, device, BETA,
        Q_A, Q_S, Lambda,
    )
    res["training_kl"] = _training_kl_comparators()
    res["prompts_seed"] = SEED
    res["n_prompts"] = n_prompts
    res["k_per_prompt"] = k_per

    out_path = PHASE4_DATA_DIR / f"e2_stationarity{suffix}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_path}")

    tk = res["training_kl"]
    print("\n=== E2 summary (all-pairs U-statistic, jackknife-over-prompts CI) ===")
    print(f"training per-step approxkl ~ {tk['approxkl_per_step_at_650']:.4f}; "
          f"policy-vs-ref k1 KL ~ {tk['k1_kl_policy_vs_ref_at_650']:.2f}; target {tk['adaptive_kl_target']}")
    for name in ("primary_Rtilde_minus_beta", "loo_advantage"):
        r = res[name]
        print(f"\n[{name}]")
        print(f"  ||G||^2 = {r['G_norm_sq_ustat']:.4e}  "
              f"jackknife SE {r['G_norm_sq_jackknife_se']:.2e} (z={r['G_norm_sq_z_jackknife']:.1f}); "
              f"subsample SE {r['G_norm_sq_subsample_se']:.2e} (z={r['G_norm_sq_z_subsample']:.1f}); "
              f"naive biased {r['G_norm_sq_naive_biased']:.4e}")
        print(f"  ||G|| = {r['G_norm']:.4e}")
        print(f"  G^T F^-1 G = {r['GtFiG_ustat']:.4e}  "
              f"jackknife SE {r['GtFiG_jackknife_se']:.2e} (z={r['GtFiG_z_jackknife']:.1f}); "
              f"subsample SE {r['GtFiG_subsample_se']:.2e} (z={r['GtFiG_z_subsample']:.1f}); "
              f"naive {r['GtFiG_naive_biased']:.4e}")
        if r["natgrad_step_kl"] == r["natgrad_step_kl"]:  # not nan
            ratio = r["natgrad_step_kl"] / tk["approxkl_per_step_at_650"]
            print(f"  natural-grad step KL (0.5 G^T F^-1 G) = {r['natgrad_step_kl']:.4e}"
                  f"  -> {ratio:.1f}x a single PPO step, "
                  f"{r['natgrad_step_kl']/tk['k1_kl_policy_vs_ref_at_650']*100:.1f}% of policy-vs-ref KL")
        else:
            print(f"  natural-grad step KL: undefined (G^T F^-1 G <= 0 within noise)")


if __name__ == "__main__":
    main()
