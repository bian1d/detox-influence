"""E1 analytics: advantage A(x,y) distribution and within-prompt variance
decomposition.

These are *pure* functions over the Phase 1 stage2 per-prompt records
(``data/phase1/stage2_per_prompt.json``); they touch no model and no GPU, so
they are exactly reproducible and reuse the already-paid sampling cost.

Why advantage matters for Phase 4
---------------------------------
The curvature-correction term central to E3 is
``Delta = E_{x, y~pi}[ A(x,y) * (s s^T + grad^2 log pi) ]`` with
``A(x,y) = R_tilde(x,y) - E_{y'}[R_tilde(x,y')]`` the same-prompt-centred
effective-reward advantage.  Only the *y-varying* part of R_tilde survives
into Delta; its constant part cancels.  So before measuring Delta (E3) we
characterise A itself: how heavy/asymmetric its tails are tells us which
prompts (large |A|) can drive Delta and are most able to reorder the IF
ranking.

Why the variance decomposition matters
--------------------------------------
ICC=0.892 says within-prompt Var_y[R_tilde] dominates, i.e. A2 fails at the
y-level.  But "fails" can mean two very different things:
  * the reward surface r(x,.) itself is genuinely spread out over responses
    (a legitimate attribution target, not a defect), or
  * the policy sits far from the KL-RL optimum in KL-divergence terms.
The identity Var_y[R_tilde] = Var_y[r] + b^2 Var_y[KL] - 2b Cov_y[r,KL]
separates these cleanly, per prompt.  A reward/length-dominated decomposition
leans transition; a KL-deviation-dominated one leans kill.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_records(
    per_prompt_path: Path,
    samples_jsonl_path: Path | None = None,
) -> list[dict]:
    """Load the Phase 1 per-prompt records, optionally enriching each with its
    ordered ``response_texts`` from the samples jsonl.

    The serialized per-prompt JSON drops response texts to stay small; the
    jsonl keeps them.  We re-attach by (prompt position, sample_idx) so that
    advantage_distribution can show what an extreme-|A| rollout reads like.
    The join is positional: jsonl ``prompt_idx_in_subset`` == list index in
    per_prompt, and ``sample_idx`` orders the K samples identically to the
    *_samples arrays.
    """
    records = json.loads(Path(per_prompt_path).read_text())
    if samples_jsonl_path is not None and Path(samples_jsonl_path).exists():
        texts: dict[int, dict[int, str]] = {}
        with open(samples_jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                texts.setdefault(row["prompt_idx_in_subset"], {})[
                    row["sample_idx"]
                ] = row["response_text"]
        for i, rec in enumerate(records):
            by_sample = texts.get(i, {})
            k = len(rec["R_tilde_samples"])
            rec["response_texts"] = [by_sample.get(j) for j in range(k)]
    return records


# --------------------------------------------------------------------------- #
# Per-prompt record access
# --------------------------------------------------------------------------- #
def _arrays(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull (reward, k1_kl, R_tilde, response_len) as float64 arrays.

    R_tilde is recomputed-free here: we trust the stored R_tilde_samples,
    which Phase 1 wrote as r - beta*k1_kl (verified bit-exact at load time by
    :func:`assert_records_consistent`).
    """
    r = np.asarray(record["reward_samples"], dtype=np.float64)
    kl = np.asarray(record["k1_kl_samples"], dtype=np.float64)
    R = np.asarray(record["R_tilde_samples"], dtype=np.float64)
    L = np.asarray(record["response_lens"], dtype=np.float64)
    return r, kl, R, L


def assert_records_consistent(records: Sequence[dict], beta: float, tol: float = 1e-9) -> None:
    """Guard: every stored R_tilde equals reward - beta*k1_kl at ``beta``.

    If this fails, the per-prompt JSON was produced with a different beta than
    the one E1 is analysing with, and every downstream advantage/decomposition
    number would be silently inconsistent with Phase 1.  Surface immediately.
    """
    worst = 0.0
    for rec in records:
        r, kl, R, _ = _arrays(rec)
        worst = max(worst, float(np.abs(R - (r - beta * kl)).max()))
    if worst > tol:
        raise ValueError(
            f"stored R_tilde inconsistent with reward - {beta}*k1_kl: "
            f"max abs diff {worst:.3e} > tol {tol:.1e}. The per-prompt file was "
            f"likely generated with a different beta."
        )


# --------------------------------------------------------------------------- #
# Advantage distribution
# --------------------------------------------------------------------------- #
def pooled_advantages(records: Sequence[dict]) -> np.ndarray:
    """A(x,y) = R_tilde(x,y) - mean_y R_tilde(x,.), pooled across all prompts.

    Within each prompt the centring makes mean_y A = 0 exactly, so with equal
    K per prompt the pooled mean is 0 up to float error (reported as a sanity
    check, not assumed).
    """
    chunks = []
    for rec in records:
        _, _, R, _ = _arrays(rec)
        chunks.append(R - R.mean())
    return np.concatenate(chunks)


def _tail_concentration(A: np.ndarray, frac: float) -> dict:
    """How the squared-advantage mass and the count split between the extreme
    upper/lower ``frac`` of A.

    Delta ~ E[A * curvature]; the prompts that can move it are the large-|A|
    tail.  We therefore report, for the most extreme ``frac`` by signed value
    on each side: the threshold, mean A there, and the share of total sum(A^2)
    it carries.  Asymmetry between the + and - sides is the "asymmetric long
    tail" Phase 1 noted.
    """
    n = A.size
    k = max(1, int(round(frac * n)))
    order = np.argsort(A)              # ascending
    lower = A[order[:k]]               # most negative
    upper = A[order[-k:]]              # most positive
    total_sq = float((A ** 2).sum())
    return {
        "frac": frac,
        "k_each_side": k,
        "lower_threshold": float(lower.max()),   # A <= this is in lower tail
        "upper_threshold": float(upper.min()),   # A >= this is in upper tail
        "lower_mean": float(lower.mean()),
        "upper_mean": float(upper.mean()),
        "lower_sq_share": float((lower ** 2).sum() / total_sq) if total_sq > 0 else float("nan"),
        "upper_sq_share": float((upper ** 2).sum() / total_sq) if total_sq > 0 else float("nan"),
        # Asymmetry: |most negative| vs |most positive| reach.
        "tail_reach_ratio_neg_over_pos": (
            float(abs(lower.min()) / abs(upper.max())) if upper.max() != 0 else float("nan")
        ),
    }


def _qq_reference(A: np.ndarray, n_points: int = 200) -> dict:
    """Theoretical-vs-sample quantiles for a normal Q-Q plot, thinned to
    ``n_points`` so the JSON stays small and the plot is regenerable."""
    a_sorted = np.sort(A)
    n = a_sorted.size
    # Sample quantiles at evenly spaced plotting positions; matched normal
    # theoretical quantiles (standardised to A's mean/std).
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.norm.ppf(probs, loc=A.mean(), scale=A.std(ddof=1))
    idx = np.linspace(0, n - 1, min(n_points, n)).round().astype(int)
    return {
        "theoretical_quantiles": theo[idx].tolist(),
        "sample_quantiles": a_sorted[idx].tolist(),
    }


def advantage_distribution(records: Sequence[dict]) -> dict:
    """Full moment + tail characterisation of pooled A, plus the most extreme
    individual responses (for qualitative reading)."""
    A = pooled_advantages(records)
    percentile_grid = [0.1, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.9]

    # Most extreme responses by signed A, with their text/R_tilde, to eyeball
    # what a large-|A| rollout actually looks like.
    flat: list[dict] = []
    for pi, rec in enumerate(records):
        _, _, R, _ = _arrays(rec)
        a = R - R.mean()
        texts = rec.get("response_texts") or [None] * len(a)
        for j in range(len(a)):
            flat.append(
                {
                    "prompt_key": rec.get("prompt_key", pi),
                    "prompt_text": rec.get("prompt_text", ""),
                    "A": float(a[j]),
                    "R_tilde": float(R[j]),
                    "response_text": texts[j] if j < len(texts) else None,
                }
            )
    flat.sort(key=lambda d: d["A"])
    extreme = {
        "most_negative": flat[:5],
        "most_positive": flat[-5:][::-1],
    }

    return {
        "n_samples": int(A.size),
        "n_prompts": len(records),
        "mean": float(A.mean()),                       # ~0 by construction
        "std": float(A.std(ddof=1)),
        "skewness": float(stats.skew(A)),
        "excess_kurtosis": float(stats.kurtosis(A, fisher=True)),
        "min": float(A.min()),
        "max": float(A.max()),
        "percentiles": {
            str(p): float(np.percentile(A, p)) for p in percentile_grid
        },
        "tail_5pct": _tail_concentration(A, 0.05),
        "tail_1pct": _tail_concentration(A, 0.01),
        "qq_normal": _qq_reference(A),
        "extreme_responses": extreme,
    }


# --------------------------------------------------------------------------- #
# Within-prompt variance decomposition
# --------------------------------------------------------------------------- #
def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else float("nan")


def variance_decomposition(records: Sequence[dict], beta: float) -> dict:
    """Decompose each prompt's Var_y[R_tilde] into reward / KL / covariance
    contributions and aggregate.

    Identity (sample variance/covariance, ddof=1 throughout so it is *exact*):
        Var_y[R_tilde] = Var_y[r] + beta^2 Var_y[KL] - 2 beta Cov_y[r, KL]

    Per-prompt "shares" are each term divided by Var_y[R_tilde]; they sum to 1
    but individual shares may be negative (the covariance term) or exceed 1.
    The headline aggregate is the *variance-weighted* share
    ``sum_x term_x / sum_x Var_x`` — i.e. each term's fraction of the total
    within-prompt variance pooled over prompts, which is what "三项在总 within
    方差中的占比" asks for.
    """
    per_prompt: list[dict] = []
    sum_varR = sum_var_r = sum_b2_varkl = sum_cov_term = 0.0
    recon_max_err = 0.0

    for pi, rec in enumerate(records):
        r, kl, R, L = _arrays(rec)
        var_r = float(np.var(r, ddof=1))
        var_kl = float(np.var(kl, ddof=1))
        cov_r_kl = float(np.cov(r, kl, ddof=1)[0, 1])
        var_R = float(np.var(R, ddof=1))

        b2_varkl = beta * beta * var_kl
        cov_term = -2.0 * beta * cov_r_kl
        reconstructed = var_r + b2_varkl + cov_term
        recon_max_err = max(recon_max_err, abs(reconstructed - var_R))

        per_prompt.append(
            {
                "prompt_key": rec.get("prompt_key", pi),
                "var_R": var_R,
                "var_r": var_r,
                "var_kl": var_kl,
                "cov_r_kl": cov_r_kl,
                "term_reward": var_r,
                "term_kl": b2_varkl,
                "term_cov": cov_term,
                "share_reward": _safe_ratio(var_r, var_R),
                "share_kl": _safe_ratio(b2_varkl, var_R),
                "share_cov": _safe_ratio(cov_term, var_R),
                "corr_len_r": _pearson(L, r),
                "corr_len_kl": _pearson(L, kl),
                "corr_len_R": _pearson(L, R),
            }
        )
        sum_varR += var_R
        sum_var_r += var_r
        sum_b2_varkl += b2_varkl
        sum_cov_term += cov_term

    # Variance-weighted aggregate shares (sum to 1 exactly).
    agg = {
        "weighted_share_reward": _safe_ratio(sum_var_r, sum_varR),
        "weighted_share_kl": _safe_ratio(sum_b2_varkl, sum_varR),
        "weighted_share_cov": _safe_ratio(sum_cov_term, sum_varR),
        "total_within_var": sum_varR,
    }
    # Distribution of the per-prompt (unweighted) shares.
    for name in ("share_reward", "share_kl", "share_cov"):
        vals = np.array([p[name] for p in per_prompt], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        agg[f"{name}_median"] = float(np.median(vals))
        agg[f"{name}_quartiles"] = [
            float(q) for q in np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0])
        ]

    # Length lens, pooled across all (prompt, sample).
    all_r, all_kl, all_R, all_L = (np.concatenate(x) for x in zip(
        *[(_arrays(rec)) for rec in records]
    ))
    length_lens = {
        "pooled_corr_len_r": _pearson(all_L, all_r),
        "pooled_corr_len_kl": _pearson(all_L, all_kl),
        "pooled_corr_len_R": _pearson(all_L, all_R),
        "pooled_rho2_len_R": _pearson(all_L, all_R) ** 2,
        "per_prompt_corr_len_R_mean": float(
            np.nanmean([p["corr_len_R"] for p in per_prompt])
        ),
    }

    return {
        "beta": beta,
        "reconstruction_max_abs_err": recon_max_err,
        "aggregate": agg,
        "length_lens": length_lens,
        "per_prompt": per_prompt,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
