"""Phase 4 E5 (diagnostic only) — k1 vs k3 KL estimator effect on ICC.

The paper used k1 (signed; on suppressed toxic tokens it flips to a reward).
k3 = sum_t[(rho_t - 1) - log rho_t] >= 0 with rho_t = pi_ref/pi_theta* is the
unbiased, low-variance, always-non-negative estimator.  E5 asks: how much of
the ICC=0.892 within-prompt variance is k1's signed-estimator artifact vs real
policy suboptimality?  If k3 markedly shrinks within-prompt variance, part of
"A2 failure" is k1-induced (a transition-leaning, honest caveat).

Diagnostic ONLY: no PPO retrain, k1 stays the main line.  The Fisher F is a
pure score outer product and does NOT depend on the KL estimator, so only
R_tilde / A / ICC change.

Faithfulness: we reproduce Phase 1 stage2's exact responses (same prompts, same
per-prompt seeds) and compute k1 AND k3 on the *identical* responses, so the
comparison isolates the estimator (ICC_k1 should reproduce ~0.892).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from phase1.config import (  # noqa: E402
    BETA, EVAL_PROMPTS_PATH, K_SAMPLES_STAGE2, N_PROMPTS_STAGE2, SEED,
)
from phase1.sample import sample_responses  # noqa: E402
from phase1.score import compute_reward_scores, response_logprobs  # noqa: E402
from phase4.config import PHASE4_DATA_DIR  # noqa: E402
from phase4.models import load_models  # noqa: E402


def _pick_stage2_prompts(n: int, seed: int) -> list[tuple[int, str]]:
    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    allp = ep["prompt_texts"]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(allp), size=n, replace=False).tolist())
    return [(int(i), allp[i]) for i in idx]


def _icc(within: np.ndarray, all_vals: list[np.ndarray]) -> float:
    pooled = np.concatenate(all_vals)
    total = float(pooled.var(ddof=1))
    return float(within.mean() / total) if total > 0 else float("nan")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PHASE4_DATA_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models(device, policy_requires_grad=False)

    records = _pick_stage2_prompts(N_PROMPTS_STAGE2, seed=SEED)
    print(f"E5: reproducing stage2 {N_PROMPTS_STAGE2} prompts x {K_SAMPLES_STAGE2}, comparing k1 vs k3")

    within_k1, within_k3 = [], []
    mean_k1, mean_k3 = [], []
    allR_k1, allR_k3 = [], []
    k1_repro_diff = 0.0
    kl_k1_all, kl_k3_all, lens_all = [], [], []
    for pos, (key, prompt_text) in enumerate(records):
        seed = SEED + key
        samples = sample_responses(models.policy, models.tokenizer, prompt_text,
                                   k=K_SAMPLES_STAGE2, device=device, seed=seed)
        rewards = compute_reward_scores([s.response_text for s in samples],
                                        models.reward, models.reward_tokenizer, device)
        R_k1, R_k3 = [], []
        for s, r in zip(samples, rewards):
            pol = response_logprobs(models.policy, s.prompt_token_ids, s.response_token_ids, device)
            ref = response_logprobs(models.ref, s.prompt_token_ids, s.response_token_ids, device)
            log_rho = (ref - pol).double()                 # log(pi_ref/pi_theta*)
            kl_k1 = float((pol - ref).double().sum().item())          # sum_t (logpol - logref)
            kl_k3 = float((log_rho.exp() - 1.0 - log_rho).sum().item())  # sum_t[(rho-1)-log rho]
            R_k1.append(r - BETA * kl_k1)
            R_k3.append(r - BETA * kl_k3)
            kl_k1_all.append(kl_k1); kl_k3_all.append(kl_k3)
            lens_all.append(int(s.response_token_ids.numel()))
        R_k1 = np.array(R_k1); R_k3 = np.array(R_k3)
        within_k1.append(R_k1.var(ddof=1)); within_k3.append(R_k3.var(ddof=1))
        mean_k1.append(R_k1.mean()); mean_k3.append(R_k3.mean())
        allR_k1.append(R_k1); allR_k3.append(R_k3)
        if (pos + 1) % 20 == 0:
            print(f"  {pos+1}/{len(records)}  within_k1={R_k1.var(ddof=1):.3f} within_k3={R_k3.var(ddof=1):.3f}")

    within_k1 = np.array(within_k1); within_k3 = np.array(within_k3)
    icc_k1 = _icc(within_k1, allR_k1)
    icc_k3 = _icc(within_k3, allR_k3)
    kl_k1_all = np.array(kl_k1_all); kl_k3_all = np.array(kl_k3_all)

    out = {
        "beta": BETA, "n_prompts": len(records), "k": K_SAMPLES_STAGE2,
        "icc_k1": icc_k1, "icc_k3": icc_k3,
        "phase1_icc_k1_reference": 0.8916784010219726,
        "mean_within_prompt_var_k1": float(within_k1.mean()),
        "mean_within_prompt_var_k3": float(within_k3.mean()),
        "within_var_shrink_ratio_k3_over_k1": float(within_k3.mean() / within_k1.mean()),
        "kl_k1_mean": float(kl_k1_all.mean()), "kl_k1_std": float(kl_k1_all.std()),
        "kl_k1_frac_negative": float((kl_k1_all < 0).mean()),
        "kl_k3_mean": float(kl_k3_all.mean()), "kl_k3_std": float(kl_k3_all.std()),
    }
    (PHASE4_DATA_DIR / "e5_estimator.json").write_text(json.dumps(out, indent=2))
    print("\n=== E5 ===")
    print(f"  ICC k1 = {icc_k1:.4f} (Phase 1 ref 0.8917)   ICC k3 = {icc_k3:.4f}")
    print(f"  mean within-prompt var: k1={within_k1.mean():.4f}  k3={within_k3.mean():.4f}  "
          f"(k3/k1 = {within_k3.mean()/within_k1.mean():.3f})")
    print(f"  k1 KL: mean={kl_k1_all.mean():.3f} frac<0={out['kl_k1_frac_negative']:.3f}; "
          f"k3 KL: mean={kl_k3_all.mean():.3f} (always >=0)")
    print(f"  wrote {PHASE4_DATA_DIR / 'e5_estimator.json'}")


if __name__ == "__main__":
    main()
