"""Phase 5 Stage 1 — ICC + reward/KL/cov decomposition on OLMo-2-1B-SFT.

Reports the THREE things the spec requires together (never ICC alone):
  (a) OLMo-SFT's actual toxicity: mean reward + reward variance (the confound
      size — is the model even toxic?).
  (b) within-prompt variance decomposition into reward / KL / covariance shares
      (phase4.e1_advantage.variance_decomposition) — the scalpel separating the
      A2 signal (KL part) from the detox-ness confound (reward part).
  (c) ICC = mean within-prompt var / total var, swept over beta.

Sampling matches GPT-Neo Phase 1 exactly: the identical seed-42 100-prompt RTP
subset, K=32 raw-completion responses from theta*, k1 KL vs the OLMo base ref.
beta sweep is free: sample once, recombine R_tilde = reward - beta*KL per beta.

    /root/miniconda3/envs/olmo/bin/python src/run_phase5_stage1_icc.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1.score import score_one_prompt  # noqa: E402
from phase1.variance import aggregate_statistics, per_prompt_variance  # noqa: E402
from phase4.e1_advantage import variance_decomposition  # noqa: E402
from phase5.config import (  # noqa: E402
    BETA_NOMINAL, BETA_SWEEP, GPTNEO_BASELINE, ICC_K_SAMPLES, ICC_N_PROMPTS,
    OUT_DIR, SEED, select_icc_prompts,
)
from phase5.models_olmo import load_olmo_models  # noqa: E402


def _records_for_beta(raw: list[dict], beta: float) -> list[dict]:
    """Recompute R_tilde = reward - beta*KL per response for a given beta."""
    out = []
    for rec in raw:
        r = np.asarray(rec["reward_samples"], dtype=np.float64)
        kl = np.asarray(rec["k1_kl_samples"], dtype=np.float64)
        out.append({
            "prompt_key": rec["prompt_key"],
            "prompt_text": rec["prompt_text"],
            "reward_samples": rec["reward_samples"],
            "k1_kl_samples": rec["k1_kl_samples"],
            "response_lens": rec["response_lens"],
            "R_tilde_samples": (r - beta * kl).tolist(),
        })
    return out


def _icc_for_beta(records: list[dict]) -> dict:
    pp = [per_prompt_variance(rec["prompt_key"], rec["prompt_text"],
                              [{"R_tilde": rt, "response_len": L, "reward": rw,
                                "k1_kl": kl, "response_text": ""}
                               for rt, L, rw, kl in zip(rec["R_tilde_samples"], rec["response_lens"],
                                                        rec["reward_samples"], rec["k1_kl_samples"])])
          for rec in records]
    return aggregate_statistics(pp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 prompts x 4 samples")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    n_prompts = args.n_prompts or (3 if args.smoke else ICC_N_PROMPTS)
    k = args.k or (4 if args.smoke else ICC_K_SAMPLES)
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading OLMo models (theta*=SFT, ref=base, reward=RoBERTa) ...")
    models = load_olmo_models(device, policy_requires_grad=False)

    prompts = select_icc_prompts(n_prompts, SEED)
    print(f"{len(prompts)} prompts (seed {SEED}), K={k} raw-completion responses each\n")

    raw: list[dict] = []
    t0 = time.time()
    for pos, (key, ptext) in enumerate(prompts):
        _, rows = score_one_prompt(
            prompt_text=ptext, k=k, seed=SEED + key, device=device,
            tokenizer=models.tokenizer, model_star=models.policy, model_ref=models.ref,
            reward_tokenizer=models.reward_tokenizer, reward_model=models.reward,
            beta=BETA_NOMINAL,
        )
        raw.append({
            "prompt_key": key, "prompt_text": ptext,
            "reward_samples": [r["reward"] for r in rows],
            "k1_kl_samples": [r["k1_kl"] for r in rows],
            "response_lens": [r["response_len"] for r in rows],
            "response_texts": [r["response_text"] for r in rows],
        })
        if (pos + 1) % 10 == 0 or args.smoke:
            print(f"  [{pos + 1}/{len(prompts)}] elapsed={time.time() - t0:.0f}s")

    # (a) toxicity level: mean reward (RoBERTa nothate logit; higher=less toxic) + variance.
    all_reward = np.array([rw for rec in raw for rw in rec["reward_samples"]], dtype=np.float64)
    all_kl = np.array([kl for rec in raw for kl in rec["k1_kl_samples"]], dtype=np.float64)
    toxicity_level = {
        "mean_reward_nothate_logit": float(all_reward.mean()),
        "var_reward": float(all_reward.var(ddof=1)),
        "std_reward": float(all_reward.std(ddof=1)),
        "reward_quartiles": [float(q) for q in np.quantile(all_reward, [0, .25, .5, .75, 1])],
        "mean_k1_kl": float(all_kl.mean()),
        "var_k1_kl": float(all_kl.var(ddof=1)),
    }

    # (b)+(c) per beta: decomposition + ICC.
    per_beta = {}
    for beta in BETA_SWEEP:
        recs = _records_for_beta(raw, beta)
        decomp = variance_decomposition(recs, beta)
        icc = _icc_for_beta(recs)
        per_beta[f"{beta}"] = {
            "beta": beta,
            "icc": icc["icc_within_over_total"],
            "mean_within_var": icc["mean_within_prompt_var"],
            "between_var": icc["between_prompt_var_of_means"],
            "total_var": icc["total_var"],
            "share_reward": decomp["aggregate"]["weighted_share_reward"],
            "share_kl": decomp["aggregate"]["weighted_share_kl"],
            "share_cov": decomp["aggregate"]["weighted_share_cov"],
            "recon_max_abs_err": decomp["reconstruction_max_abs_err"],
            "pooled_corr_len_R": icc["pooled_corr_length_R"],
        }

    # a few example (prompt -> responses) for the report, to show raw-completion
    # generation is sensible (and how toxic OLMo-SFT actually gets).
    examples = [
        {"prompt": rec["prompt_text"],
         "responses": [{"text": t, "reward": rw} for t, rw in
                       zip(rec["response_texts"][:3], rec["reward_samples"][:3])]}
        for rec in raw[:3]
    ]

    result = {
        "model": "OLMo-2-0425-1B-SFT", "ref": "OLMo-2-0425-1B",
        "n_prompts": len(prompts), "k_samples": k, "seed": SEED,
        "beta_nominal": BETA_NOMINAL, "beta_sweep": BETA_SWEEP,
        "toxicity_level": toxicity_level,
        "per_beta": per_beta,
        "examples": examples,
        "gptneo_baseline": GPTNEO_BASELINE,
    }
    out_path = OUT_DIR / ("stage1_icc_smoke.json" if args.smoke else "stage1_icc.json")
    out_path.write_text(json.dumps(result, indent=2))

    # console summary
    print("\n" + "=" * 70)
    print(f"(a) toxicity level: mean nothate logit = {toxicity_level['mean_reward_nothate_logit']:+.3f} "
          f"(higher=less toxic), var = {toxicity_level['var_reward']:.3f}")
    print(f"(b)+(c) per beta  [GPT-Neo: ICC 0.892, shares reward 0.47 / KL 0.56]:")
    print(f"  {'beta':>7} {'ICC':>7} {'reward':>8} {'KL':>8} {'cov':>8}")
    for b, d in per_beta.items():
        print(f"  {float(b):>7.4g} {d['icc']:>7.3f} {d['share_reward']:>8.3f} "
              f"{d['share_kl']:>8.3f} {d['share_cov']:>8.3f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
