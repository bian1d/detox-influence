"""E1 figures: advantage distribution, normal Q-Q, variance-share composition.

Regenerated from the per-prompt records (cheap), so the JSON stays lean and
the plots are reproducible from data alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from phase4.e1_advantage import _arrays, pooled_advantages, variance_decomposition


def plot_advantage_hist(records: Sequence[dict], out_path: Path) -> None:
    A = pooled_advantages(records)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(A, bins=80, density=True, alpha=0.65, color="#4477aa", label="A(x,y) pooled")
    xs = np.linspace(A.min(), A.max(), 400)
    ax.plot(
        xs,
        stats.norm.pdf(xs, A.mean(), A.std(ddof=1)),
        "r-",
        lw=1.6,
        label=f"N(0, {A.std(ddof=1):.2f}²) reference",
    )
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("advantage  A(x,y) = R̃ - mean_y R̃_x")
    ax.set_ylabel("density")
    ax.set_title(
        f"E1: effective-reward advantage  (skew={stats.skew(A):.2f}, "
        f"excess kurt={stats.kurtosis(A):.2f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_advantage_qq(records: Sequence[dict], out_path: Path) -> None:
    A = pooled_advantages(records)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    stats.probplot(A, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(2.5)
    ax.get_lines()[0].set_alpha(0.5)
    ax.set_title("E1: normal Q-Q of advantage A")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_variance_decomposition(records: Sequence[dict], beta: float, out_path: Path) -> None:
    """Per-prompt stacked composition of within-variance into reward / KL /
    covariance terms, ordered by total within-variance."""
    decomp = variance_decomposition(records, beta)
    pp = decomp["per_prompt"]
    order = sorted(range(len(pp)), key=lambda i: pp[i]["var_R"])
    term_r = np.array([pp[i]["term_reward"] for i in order])
    term_kl = np.array([pp[i]["term_kl"] for i in order])
    term_cov = np.array([pp[i]["term_cov"] for i in order])
    var_R = np.array([pp[i]["var_R"] for i in order])
    x = np.arange(len(order))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Absolute terms (cov can be negative -> plot as signed bars stacked).
    ax1.bar(x, term_r, color="#4477aa", label="Var[r] (reward)")
    ax1.bar(x, term_kl, bottom=term_r, color="#66ccee", label="β²·Var[KL]")
    ax1.bar(x, term_cov, bottom=term_r + term_kl, color="#ee6677",
            label="−2β·Cov[r,KL]")
    ax1.plot(x, var_R, "k-", lw=1.2, label="Var[R̃] (sum)")
    ax1.set_ylabel("variance contribution")
    ax1.set_title("E1: within-prompt variance decomposition (prompts sorted by Var[R̃])")
    ax1.legend(fontsize=8, loc="upper right")

    # Weighted aggregate shares as text.
    agg = decomp["aggregate"]
    ax1.text(
        0.02, 0.95,
        f"weighted shares  reward={agg['weighted_share_reward']:.2f}  "
        f"KL={agg['weighted_share_kl']:.2f}  cov={agg['weighted_share_cov']:.2f}",
        transform=ax1.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    # Normalised shares per prompt.
    share_r = term_r / var_R
    share_kl = term_kl / var_R
    share_cov = term_cov / var_R
    ax2.bar(x, share_r, color="#4477aa")
    ax2.bar(x, share_kl, bottom=share_r, color="#66ccee")
    ax2.bar(x, share_cov, bottom=share_r + share_kl, color="#ee6677")
    ax2.axhline(1.0, color="k", lw=0.7, ls=":")
    ax2.set_ylabel("share of Var[R̃]")
    ax2.set_xlabel("prompt (sorted by within-prompt variance)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
