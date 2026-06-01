"""E1: sampling-error band on the ICC=0.892 statistic.

What this measures (and what it does NOT)
------------------------------------------
The original Phase 4 draft proposed a "fix the response, resample, watch
R_tilde jitter" noise floor.  That is conceptually empty: with (x, y, theta*)
all fixed, R_tilde = r(x,y) - beta*KL(y) is a *deterministic* number (RoBERTa
scores fixed text identically; a fixed model assigns fixed log-probs).  There
is no within-(x,y) randomness to measure.

The honest quantity is the sampling error of the ICC *statistic*: ICC=0.892
was computed from one draw of 32 responses per prompt.  Draw a fresh 32 (new
seed) and ICC moves a little.  We quantify that run-to-run wobble on a 20-prompt
subset by redrawing responses ``ICC_N_RESEEDS`` times and reporting the spread.

This is NOT a claim that within-prompt variance contains a noise floor
independent of A: by definition Var_y[R_tilde] *is* the A2-failure signal.
RoBERTa's fragility to paraphrase contributes to it, but that is part of the
reward definition and a legitimate attribution target, not noise to discount.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from phase1.score import score_one_prompt
from phase1.variance import aggregate_statistics, per_prompt_variance
from phase4.models import Models


# --------------------------------------------------------------------------- #
# Subset selection (Q2: uniform-by-within-variance, top-3 guaranteed)
# --------------------------------------------------------------------------- #
def select_icc_subset(records: Sequence[dict], n: int) -> list[int]:
    """Indices into ``records`` for the ICC subset.

    The 3 highest-within-variance prompts are force-included: they have the
    largest |A| tail and are the most likely to reorder the IF ranking, so
    their ICC contribution is exactly where the error band must be measured.
    The remaining n-3 slots are spread evenly across the within-variance rank
    order of the other prompts, so the subset covers the whole variance range
    rather than clustering.  Deterministic (no RNG): the selection is a fixed
    function of the variance ranking.
    """
    if n < 3:
        raise ValueError("n must be >= 3 to guarantee the top-3 within-var prompts")
    var = np.array([rec["var_R_tilde"] for rec in records], dtype=np.float64)
    order_desc = np.argsort(-var)            # highest variance first
    top3 = order_desc[:3].tolist()

    rest = order_desc[3:]                     # remaining, still high->low
    # Evenly spaced positions across the remaining rank range.
    positions = np.linspace(0, len(rest) - 1, n - 3).round().astype(int)
    picked_rest = [int(rest[p]) for p in dict.fromkeys(positions.tolist())]

    # Guard against linspace collisions reducing the count.
    chosen = top3 + picked_rest
    if len(chosen) < n:
        for idx in rest.tolist():
            if idx not in chosen:
                chosen.append(int(idx))
            if len(chosen) == n:
                break
    return sorted(chosen[:n])


# --------------------------------------------------------------------------- #
# ICC on a subset
# --------------------------------------------------------------------------- #
def _subset_icc_from_phase1(records: Sequence[dict], subset_idx: Sequence[int]) -> float:
    """Anchor: ICC on the subset using Phase 1's already-drawn responses,
    via the identical aggregate_statistics estimator."""
    per_prompt = []
    for i in subset_idx:
        rec = records[i]
        rows = [
            {
                "R_tilde": rt,
                "response_len": L,
                "reward": rw,
                "k1_kl": kl,
                "response_text": "",
            }
            for rt, L, rw, kl in zip(
                rec["R_tilde_samples"],
                rec["response_lens"],
                rec["reward_samples"],
                rec["k1_kl_samples"],
            )
        ]
        per_prompt.append(per_prompt_variance(rec.get("prompt_key", i), rec.get("prompt_text", ""), rows))
    return aggregate_statistics(per_prompt)["icc_within_over_total"]


def resample_icc(
    records: Sequence[dict],
    subset_idx: Sequence[int],
    models: Models,
    device: str,
    beta: float,
    n_reseeds: int,
    k: int,
    seed_base: int,
) -> dict:
    """Redraw k on-policy responses per subset prompt ``n_reseeds`` times;
    recompute the subset ICC each time; report the spread.

    Each (reseed, prompt) uses a distinct seed so all response streams are
    independent.  The estimator is Phase 1's aggregate_statistics, so the
    resampled ICCs are on the same footing as the published 0.892.
    """
    prompts = [records[i].get("prompt_text", "") for i in subset_idx]
    keys = [records[i].get("prompt_key", i) for i in subset_idx]

    icc_values: list[float] = []
    t0 = time.time()
    for s in range(n_reseeds):
        per_prompt = []
        for pos, (key, prompt_text) in enumerate(zip(keys, prompts)):
            seed = seed_base + s * 100_000 + pos
            _, rows = score_one_prompt(
                prompt_text=prompt_text,
                k=k,
                seed=seed,
                device=device,
                tokenizer=models.tokenizer,
                model_star=models.policy,
                model_ref=models.ref,
                reward_tokenizer=models.reward_tokenizer,
                reward_model=models.reward,
                beta=beta,
            )
            per_prompt.append(per_prompt_variance(key, prompt_text, rows))
        icc = aggregate_statistics(per_prompt)["icc_within_over_total"]
        icc_values.append(float(icc))
        print(
            f"  [icc-resample] reseed {s + 1}/{n_reseeds}  "
            f"ICC={icc:.4f}  elapsed={time.time() - t0:.0f}s"
        )

    icc_arr = np.array(icc_values, dtype=np.float64)
    boot = _bootstrap_mean_ci(icc_arr, n_boot=2000, seed=1000)
    return {
        "n_prompts_subset": len(subset_idx),
        "subset_idx": list(subset_idx),
        "subset_prompt_keys": [int(k) for k in keys],
        "n_reseeds": n_reseeds,
        "k_samples": k,
        "seed_base": seed_base,
        "icc_phase1_full_100": None,        # filled by driver from summary
        "icc_phase1_on_subset": _subset_icc_from_phase1(records, subset_idx),
        "icc_resample_values": icc_values,
        "icc_resample_mean": float(icc_arr.mean()),
        "icc_resample_std": float(icc_arr.std(ddof=1)),
        "icc_resample_min": float(icc_arr.min()),
        "icc_resample_max": float(icc_arr.max()),
        "icc_resample_mean_ci95": boot,
    }


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    n = values.size
    means = np.array(
        [values[rng.randint(0, n, size=n)].mean() for _ in range(n_boot)]
    )
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
