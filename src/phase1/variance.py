"""Per-prompt and aggregate variance of effective reward, plus length
correlation and ICC.

The KL-RL optimum has R̃(x, y, θ*) = β log Z(x): for fixed x all y samples
yield identical R̃. Variance ACROSS y at fixed x measures deviation from
the optimum. Variance ACROSS x is structurally non-zero (β log Z(x) varies
by prompt) and is NOT a problem.

Three quantities matter:
  - within_prompt_var(x): Var_y[R̃ | x] — the diagnostic signal.
  - between_prompt_var: Var_x[mean_y R̃(x,·)] — structural, expected to
    dominate at a KL-RL optimum.
  - ICC = mean_x[within(x)] / total_var(R̃) — small ICC (< 0.1 strict,
    < 0.3 acceptable) supports the cancellation identity.

Length-stratification: a high corr(length, R̃) within a prompt suggests
the R̃ variance is driven by length variation in our sampling (which
under min_new_tokens=10 + max_new_tokens=30 + EOS can vary 10-30 tokens)
rather than genuine policy suboptimality. The R̃ formula already
subtracts β·KL so any KL-with-length scaling is partially absorbed; what
remains is the residual length dependence.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PromptVariance:
    prompt_idx: int
    prompt_text: str
    R_tilde_samples: list[float]
    response_lens: list[int]
    reward_samples: list[float]
    k1_kl_samples: list[float]
    response_texts: list[str] = field(default_factory=list)
    mean_R_tilde: float = 0.0
    var_R_tilde: float = 0.0           # sample variance, ddof=1
    cv_R_tilde: float = 0.0            # |std / mean|, NaN if mean ≈ 0
    corr_len_R: float = 0.0            # Pearson r(length, R̃); NaN if degenerate


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns NaN if either series is constant."""
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def per_prompt_variance(
    prompt_idx: int,
    prompt_text: str,
    rows: list[dict],
) -> PromptVariance:
    """Aggregate K rows for one prompt into variance + correlations."""
    R = np.array([r["R_tilde"] for r in rows], dtype=np.float64)
    L = np.array([r["response_len"] for r in rows], dtype=np.float64)
    mean = float(R.mean())
    var = float(R.var(ddof=1))
    cv = float(np.sqrt(var) / abs(mean)) if abs(mean) > 1e-12 else float("nan")
    return PromptVariance(
        prompt_idx=prompt_idx,
        prompt_text=prompt_text,
        R_tilde_samples=R.tolist(),
        response_lens=L.astype(int).tolist(),
        reward_samples=[r["reward"] for r in rows],
        k1_kl_samples=[r["k1_kl"] for r in rows],
        response_texts=[r["response_text"] for r in rows],
        mean_R_tilde=mean,
        var_R_tilde=var,
        cv_R_tilde=cv,
        corr_len_R=_safe_corr(L, R),
    )


def aggregate_statistics(per_prompt: list[PromptVariance]) -> dict:
    """Mean/median within-prompt variance, total variance, ICC, length corr.

    ICC = mean within / total. Close to 0 means most variance is
    between-prompt (β log Z(x) heterogeneity, expected and fine). Close
    to 1 means most variance is within-prompt (deviation from KL-RL
    optimum, bad).
    """
    within = np.array([p.var_R_tilde for p in per_prompt], dtype=np.float64)
    means = np.array([p.mean_R_tilde for p in per_prompt], dtype=np.float64)

    # Pool all (R̃, length) across all (prompt, sample) for the overall stats
    all_R = np.concatenate(
        [np.asarray(p.R_tilde_samples, dtype=np.float64) for p in per_prompt]
    )
    all_L = np.concatenate(
        [np.asarray(p.response_lens, dtype=np.float64) for p in per_prompt]
    )
    total_var = float(all_R.var(ddof=1))

    # Per-prompt corr(length, R̃) summary stats
    per_prompt_corr = np.array(
        [p.corr_len_R for p in per_prompt], dtype=np.float64
    )
    per_prompt_corr_clean = per_prompt_corr[~np.isnan(per_prompt_corr)]

    return {
        "n_prompts": len(per_prompt),
        "k_samples_per_prompt": (
            len(per_prompt[0].R_tilde_samples) if per_prompt else 0
        ),
        # Within-prompt variance distribution
        "mean_within_prompt_var": float(within.mean()),
        "median_within_prompt_var": float(np.median(within)),
        "within_var_quartiles": [
            float(q) for q in np.quantile(within, [0.0, 0.25, 0.5, 0.75, 1.0])
        ],
        # Between-prompt variance of per-prompt means
        "between_prompt_var_of_means": float(means.var(ddof=1)),
        # Pooled total variance (all 100*32 = 3200 samples)
        "total_var": total_var,
        # ICC (intraclass correlation)
        "icc_within_over_total": (
            float(within.mean() / total_var) if total_var > 0 else float("nan")
        ),
        # Length correlation
        "pooled_corr_length_R": _safe_corr(all_L, all_R),
        "per_prompt_corr_length_R_mean": (
            float(per_prompt_corr_clean.mean()) if per_prompt_corr_clean.size else float("nan")
        ),
        "per_prompt_corr_length_R_median": (
            float(np.median(per_prompt_corr_clean)) if per_prompt_corr_clean.size else float("nan")
        ),
        "n_prompts_with_nan_corr": int(np.isnan(per_prompt_corr).sum()),
    }


def outliers_by_within_variance(
    per_prompt: list[PromptVariance],
    n_top: int = 10,
) -> list[PromptVariance]:
    """Return the n_top prompts with largest within-prompt variance."""
    return sorted(per_prompt, key=lambda p: p.var_R_tilde, reverse=True)[:n_top]
