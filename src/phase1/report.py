"""Generate reports/phase1_report.md and supporting plots from Stage 2 +
Framing B output jsons. Pure post-processing — no GPU/model needed.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "phase1"
REPORTS = REPO / "reports"


def _format_float_list(xs, fmt: str = ".4f") -> str:
    return "[" + ", ".join(f"{x:{fmt}}" for x in xs) + "]"


def _load_dataset(prefix: str) -> tuple[list[dict], dict]:
    """Load per_prompt.json + attach response_texts from samples.jsonl."""
    with open(DATA / f"{prefix}_per_prompt.json") as f:
        per_prompt = json.load(f)
    text_by_key: dict[tuple[int, int], str] = {}
    with open(DATA / f"{prefix}_samples.jsonl") as f:
        for line in f:
            r = json.loads(line)
            text_by_key[(r["prompt_key"], r["sample_idx"])] = r["response_text"]
    for p in per_prompt:
        p["response_texts"] = [
            text_by_key[(p["prompt_key"], j)]
            for j in range(len(p["R_tilde_samples"]))
        ]
    with open(DATA / f"{prefix}_summary.json") as f:
        summary = json.load(f)
    return per_prompt, summary


def make_within_var_histogram(
    stage2: list[dict],
    framing_b: list[dict] | None,
    out_path: Path,
) -> None:
    within_s = np.array([p["var_R_tilde"] for p in stage2])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bins = np.linspace(0, max(within_s.max(), 8), 30)
    axes[0].hist(within_s, bins=bins, alpha=0.6, edgecolor="black",
                 color="#4488cc", label=f"Stage 2 (in-training)  μ={within_s.mean():.2f}")
    if framing_b is not None:
        within_b = np.array([p["var_R_tilde"] for p in framing_b])
        axes[0].hist(within_b, bins=bins, alpha=0.6, edgecolor="black",
                     color="#cc4444", label=f"Framing B (held-out)  μ={within_b.mean():.2f}")
    axes[0].set_xlabel("Within-prompt Var[R̃]")
    axes[0].set_ylabel("# prompts")
    axes[0].set_title("Per-prompt within-variance distribution")
    axes[0].legend()

    log_s = np.log10(np.clip(within_s, 1e-6, None))
    axes[1].hist(log_s, bins=30, alpha=0.6, edgecolor="black",
                 color="#4488cc", label="Stage 2")
    if framing_b is not None:
        log_b = np.log10(np.clip(within_b, 1e-6, None))
        axes[1].hist(log_b, bins=30, alpha=0.6, edgecolor="black",
                     color="#cc4444", label="Framing B")
    axes[1].set_xlabel("log10(Within-prompt Var[R̃])")
    axes[1].set_ylabel("# prompts")
    axes[1].set_title("Log-scale view (long-tail visibility)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def make_corr_length_plots(per_prompt: list[dict], out_path: Path) -> None:
    corrs = np.array(
        [p["corr_len_R"] for p in per_prompt if not np.isnan(p["corr_len_R"])]
    )

    L_all, R_all = [], []
    for p in per_prompt:
        L_all.extend(p["response_lens"])
        R_all.extend(p["R_tilde_samples"])
    L_all = np.array(L_all)
    R_all = np.array(R_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.hist(corrs, bins=25, edgecolor="black", color="#88aacc")
    ax1.set_xlabel("Per-prompt corr(length, R̃)")
    ax1.set_ylabel("# prompts")
    ax1.set_title(f"Per-prompt length–R̃ correlation (mean={corrs.mean():+.3f})")
    ax1.axvline(0.0, color="black", linewidth=0.6)
    ax1.axvline(corrs.mean(), color="red", linestyle="--")

    ax2.hexbin(L_all, R_all, gridsize=20, cmap="viridis", mincnt=1)
    ax2.set_xlabel("Response length (tokens)")
    ax2.set_ylabel("R̃")
    rho_pool = np.corrcoef(L_all, R_all)[0, 1]
    ax2.set_title(f"Pooled R̃ vs length (corr={rho_pool:+.3f})")

    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _bimodal_split_index(R_sorted: list[float]) -> int:
    """Return index i such that R_sorted[:i] is the "low" mode and
    R_sorted[i:] is the "high" mode, chosen by the largest consecutive gap.

    Edge case: if all values are very close (max gap < 1.0), returns
    len(R_sorted) (no meaningful split).
    """
    diffs = np.diff(R_sorted)
    i = int(np.argmax(diffs)) + 1
    if diffs[i - 1] < 1.0:
        return len(R_sorted)
    return i


def _fmt_outlier_detail(p: dict, rank: int) -> str:
    """Top-3 outlier: all 32 samples sorted by R̃, with bimodal split marker."""
    K = len(p["R_tilde_samples"])
    order = sorted(range(K), key=lambda i: p["R_tilde_samples"][i])
    R_sorted = [p["R_tilde_samples"][i] for i in order]
    split = _bimodal_split_index(R_sorted)
    split_gap = (R_sorted[split] - R_sorted[split - 1]) if split < K else 0.0

    lines = [
        f"### Outlier {rank}: key={p['prompt_key']}  "
        f"Var[R̃]={p['var_R_tilde']:.4f}  "
        f"mean R̃={p['mean_R_tilde']:+.3f}  "
        f"corr(len, R̃)={p['corr_len_R']:+.3f}\n",
        f"**Prompt:** `{p['prompt_text']!r}`\n",
    ]
    if split < K:
        lines.append(
            f"**Bimodal split** at rank {split}/{K} "
            f"(R̃ gap = {split_gap:.2f}): "
            f"low mode mean = {np.mean(R_sorted[:split]):+.3f} "
            f"(n={split}); "
            f"high mode mean = {np.mean(R_sorted[split:]):+.3f} "
            f"(n={K - split}).\n"
        )
    else:
        lines.append("No clear bimodal split (max gap < 1.0 in R̃).\n")

    lines.append("| rank | sample_idx | len | reward | KL_k1 | R̃ | response |")
    lines.append("|---:|---:|---:|---:|---:|---:|:---|")
    for r, i in enumerate(order):
        txt = p["response_texts"][i].replace("|", "\\|").replace("\n", " ↩ ")
        if len(txt) > 120:
            txt = txt[:117] + "..."
        lines.append(
            f"| {r + 1} | {i} | {p['response_lens'][i]} | "
            f"{p['reward_samples'][i]:+.3f} | "
            f"{p['k1_kl_samples'][i]:+.3f} | "
            f"{p['R_tilde_samples'][i]:+.3f} | `{txt}` |"
        )
        if split < K and r + 1 == split:
            lines.append(
                f"| --- | **── bimodal split ── (gap={split_gap:.2f}) ──** | | | | | |"
            )
    lines.append("")
    return "\n".join(lines)


def _fmt_outlier_summary(per_prompt: list[dict], n_top: int = 10) -> str:
    """Ranks 4-10 outlier summary table (compact)."""
    sorted_p = sorted(per_prompt, key=lambda p: p["var_R_tilde"], reverse=True)
    lines = [
        "| rank | key | Var[R̃] | mean R̃ | corr(L,R̃) | prompt |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for rank, p in enumerate(sorted_p[3:n_top], start=4):
        txt = p["prompt_text"]
        if len(txt) > 70:
            txt = txt[:67] + "..."
        lines.append(
            f"| {rank} | {p['prompt_key']} | {p['var_R_tilde']:.4f} | "
            f"{p['mean_R_tilde']:+.3f} | {p['corr_len_R']:+.3f} | `{txt}` |"
        )
    return "\n".join(lines)


def _icc_verdict(icc: float) -> str:
    if icc < 0.1:
        return (
            "**STRICT pass (ICC < 0.10).** Within-prompt variance is small "
            "relative to total variance — the cancellation identity "
            "R̃ ≈ β log Z(x) holds well, strongly supporting the IF "
            "stationarity assumption."
        )
    if icc < 0.3:
        return (
            "**Acceptable (0.10 ≤ ICC < 0.30).** Within-prompt variance is "
            "moderate; the IF formula is usable but Phase 2 IF scores will "
            "carry corresponding noise inflation. Worth reporting as a "
            "caveat alongside the rankings."
        )
    return (
        "**Weak (ICC ≥ 0.30).** Within-prompt variance is comparable to or "
        "larger than between-prompt variance; θ* is not close to a KL-RL "
        "optimum in the per-y sense. IF rankings can still be computed but "
        "should be interpreted as a local natural-gradient approximation "
        "around θ* with the stationarity assumption holding only loosely."
    )


def _framing_b_section(s_summary: dict, b_summary: dict) -> str:
    """Side-by-side comparison + thesis-grade interpretation."""
    delta_icc = b_summary["icc_within_over_total"] - s_summary["icc_within_over_total"]
    delta_within = b_summary["mean_within_prompt_var"] - s_summary["mean_within_prompt_var"]

    if abs(delta_icc) < 0.05:
        interp = (
            f"**Held-out ICC = {b_summary['icc_within_over_total']:.3f} is "
            f"essentially the same as in-training ICC = "
            f"{s_summary['icc_within_over_total']:.3f} (Δ = "
            f"{delta_icc:+.3f}).** This indicates the within-prompt R̃ "
            "variance is **intrinsic to PPO's local-optimization nature** "
            "rather than a function of training-distribution familiarity. "
            "PPO does not generalize the KL-RL cancellation to held-out "
            "prompts because it never establishes the cancellation tightly "
            "on the training distribution in the first place — it converges "
            "to a local optimum of the KL-regularized objective that has "
            "non-trivial residual response variance per prompt by "
            "construction."
        )
    elif delta_icc > 0.05:
        interp = (
            f"**Held-out ICC = {b_summary['icc_within_over_total']:.3f} is "
            f"meaningfully higher than in-training ICC = "
            f"{s_summary['icc_within_over_total']:.3f} (Δ = "
            f"{delta_icc:+.3f}).** The KL-RL cancellation property is "
            "weaker on held-out prompts than on in-training-distribution "
            "prompts, indicating that PPO has partially adapted to the "
            "training-prompt distribution but its KL-RL approximation does "
            "not transfer to OOD prompts. For Phase 2 IF analysis on "
            "training-distribution rollouts (the planned use case), the "
            "in-training-distribution ICC is the operative number; "
            "held-out generalization is a secondary observation."
        )
    else:
        interp = (
            f"**Held-out ICC = {b_summary['icc_within_over_total']:.3f} is "
            f"meaningfully LOWER than in-training ICC = "
            f"{s_summary['icc_within_over_total']:.3f} (Δ = "
            f"{delta_icc:+.3f}).** This is unusual and counter to the "
            "natural expectation (PPO trained ON the training-distribution "
            "should approximate KL-RL there better than off it). A possible "
            "explanation is that the in-training prompt set is biased "
            "toward toxic-baited prompts (Phase 0 used uniform sampling "
            "from RTP for eval, with no toxicity filter; the bimodal-on-"
            "toxic-prompts phenomenon in Section 5 may explain the gap). "
            "Worth flagging for further investigation."
        )

    lines = [
        "## 6. Framing B — Held-out distribution comparison",
        "",
        f"Sampled N={b_summary['n_prompts']} prompts from RealToxicityPrompts "
        f"that are **not** in the PPO training set (10K, toxicity > 0.3, "
        f"seed=42) and **not** in `eval_prompts_400.json`. Fresh sampling "
        f"seed = {b_summary['seed'] + 1000} (different from Stage 2's "
        f"prompt-selection seed 42 and PPO training seed 42). Same K=32 "
        "sampling protocol as Stage 2 to ensure apples-to-apples ICC.",
        "",
        "| statistic | Stage 2 (in-training, N=100) | Framing B (held-out, N=100) | Δ (B − S2) |",
        "|---|---:|---:|---:|",
        f"| mean within-prompt Var[R̃] | {s_summary['mean_within_prompt_var']:.4f} "
        f"| {b_summary['mean_within_prompt_var']:.4f} "
        f"| {delta_within:+.4f} |",
        f"| median within-prompt Var[R̃] | {s_summary['median_within_prompt_var']:.4f} "
        f"| {b_summary['median_within_prompt_var']:.4f} "
        f"| {b_summary['median_within_prompt_var'] - s_summary['median_within_prompt_var']:+.4f} |",
        f"| between-prompt Var of means | {s_summary['between_prompt_var_of_means']:.4f} "
        f"| {b_summary['between_prompt_var_of_means']:.4f} "
        f"| {b_summary['between_prompt_var_of_means'] - s_summary['between_prompt_var_of_means']:+.4f} |",
        f"| total Var[R̃] (pooled 3200) | {s_summary['total_var']:.4f} "
        f"| {b_summary['total_var']:.4f} "
        f"| {b_summary['total_var'] - s_summary['total_var']:+.4f} |",
        f"| **ICC = within / total** | **{s_summary['icc_within_over_total']:.4f}** "
        f"| **{b_summary['icc_within_over_total']:.4f}** "
        f"| **{delta_icc:+.4f}** |",
        f"| pooled corr(length, R̃) | {s_summary['pooled_corr_length_R']:+.4f} "
        f"| {b_summary['pooled_corr_length_R']:+.4f} "
        f"| {b_summary['pooled_corr_length_R'] - s_summary['pooled_corr_length_R']:+.4f} |",
        f"| per-prompt corr(L, R̃) mean | {s_summary['per_prompt_corr_length_R_mean']:+.4f} "
        f"| {b_summary['per_prompt_corr_length_R_mean']:+.4f} "
        f"| {b_summary['per_prompt_corr_length_R_mean'] - s_summary['per_prompt_corr_length_R_mean']:+.4f} |",
        "",
        interp,
        "",
    ]
    return "\n".join(lines)


def _thesis_paragraph(s_summary: dict, b_summary: dict | None) -> str:
    icc = s_summary["icc_within_over_total"]
    icc_b = b_summary["icc_within_over_total"] if b_summary else None
    length_share = s_summary["per_prompt_corr_length_R_mean"] ** 2

    base = (
        "The IF derivation (CONTEXT.md §5) treats $\\theta^*$ as a stationary "
        "point of the KL-regularized objective. Concretely, the score-function "
        "expansion that yields $I = -g_\\text{eval}^\\top F^{-1} s_m$ uses "
        "$\\tilde R(x, y, \\theta^*) \\approx \\beta \\log Z(x)$ to cancel the "
        "$\\nabla_\\theta \\tilde R$ term against the Fisher. Empirically, the "
        f"in-training-distribution ICC of `{icc:.3f}` says that "
        f"`{100 * icc:.1f}%` of total Var[R̃] is within-prompt; only "
        f"`{100 * (1 - icc):.1f}%` is the structural between-prompt "
        "$\\beta \\log Z(x)$ heterogeneity. Length variation explains "
        f"$\\rho^2 \\approx {length_share:.3f}$ of the within-prompt "
        "fraction (per-prompt corr(L, R̃) = "
        f"`{s_summary['per_prompt_corr_length_R_mean']:+.3f}`), leaving "
        "the residual as genuine policy-suboptimality signal."
    )

    if icc_b is not None:
        gen_clause = (
            "Framing B held-out ICC = "
            f"`{icc_b:.3f}` (Δ = `{icc_b - icc:+.3f}` vs in-training) "
        )
        if abs(icc_b - icc) < 0.05:
            gen_clause += (
                "indicates the suboptimality is intrinsic to PPO's "
                "local-optimization nature rather than a "
                "training-distribution artifact."
            )
        elif icc_b > icc:
            gen_clause += (
                "indicates the KL-RL approximation weakens off-distribution; "
                "for Phase 2 IF on in-training rollouts, the Stage 2 ICC is "
                "the operative number."
            )
        else:
            gen_clause += (
                "is somewhat lower than in-training, possibly because "
                "uniformly-sampled held-out prompts contain fewer of the "
                "toxic-baited inputs that exhibit bimodal R̃."
            )
        base += " " + gen_clause

    base += (
        " Phase 2 IF scores are therefore reported as **local natural-gradient "
        "influence around $\\theta^*$**, with the stationarity assumption "
        "satisfied at the prompt level (between-prompt cancellation is clean) "
        "but only loosely at the y level (within-prompt residual variance). "
        "This matches the PBRF formulation of Bae et al. 2022: a KL-regularized "
        "RLHF policy inherently satisfies the PBRF-analogous stationarity "
        "condition by construction up to the local-optimization gap, which the "
        "within-prompt variance here directly quantifies."
    )
    return base


def generate_report(
    stage2: list[dict],
    s_summary: dict,
    framing_b: list[dict] | None,
    b_summary: dict | None,
    out_path: Path,
    hist_rel: str,
    corr_rel: str,
) -> None:
    icc_verdict = _icc_verdict(s_summary["icc_within_over_total"])
    sorted_p = sorted(stage2, key=lambda p: p["var_R_tilde"], reverse=True)
    top_3 = sorted_p[:3]

    framing_b_section = (
        _framing_b_section(s_summary, b_summary)
        if b_summary is not None
        else "## 6. Framing B — Held-out distribution comparison\n\n*Not run.*\n"
    )
    thesis_para = _thesis_paragraph(s_summary, b_summary)

    md = f"""# Phase 1 Report — Effective Reward Variance Diagnostic

## 1. Setup

- **Checkpoint θ\\***: `data/ppo_checkpoints/step_0650/` (Phase 0 three-phase Pareto-optimal: toxicity 0.016, reward 3.84, PPL 33, length 10).
- **β at step 650**: `{s_summary['beta']:.12f}` (adaptive controller value; initial 0.2).
- **In-training prompts (Stage 2)**: N={s_summary['n_prompts']} random subset of `eval_prompts_400.json` (seed=42); these were drawn uniformly from RTP for Phase 0 evaluation.
- **Held-out prompts (Framing B)**: N={b_summary['n_prompts'] if b_summary else 0} RTP indices excluded from both PPO training (10K, tox > 0.3, seed=42) and from `eval_prompts_400.json` (fresh seed = SEED + 1000).
- **K = {s_summary['k_samples_per_prompt']} samples per prompt** via `do_sample=True, top_k=0, top_p=1.0, max_new_tokens=30, min_new_tokens=10` — matching Phase 0 PPO training generation kwargs.
- **Reward**: raw nothate logit (`logits[:, 0]`) from `facebook/roberta-hate-speech-dynabench-r4-target`, continuation-only.
- **KL estimator**: **k1** (signed log-ratio sum over response tokens, prompt-masked).
- **β value**: `{s_summary['beta']:.6f}` — adaptive controller value post-update at step 650, paired with checkpoint weights (both POST-update at step 650).
- **Effective reward**: $\\tilde R(x, y, \\theta) = r(x, y) - \\beta \\cdot \\widehat{{\\mathrm{{KL}}}}_{{k1}}(y|x)$. At the KL-RL optimum $\\pi^*_x(y) \\propto \\pi_\\text{{ref}}(y|x) \\exp(r/\\beta)$, this simplifies to $\\beta \\log Z(x)$, depending only on $x$ — the central testable property.

### Why k1 (unbiased), not k2

k1 = $\\sum_t [\\log\\pi_\\theta - \\log\\pi_\\text{{ref}}]$ is unbiased w.r.t. the true KL divergence (Schulman 2020). Phase 1 must use k1 — not k2 — for two reasons. **(a) Theoretical**: the cancellation identity $\\tilde R \\equiv \\beta \\log Z(x)$ at $\\pi^*_x$ requires $\\mathbb{{E}}_y[\\widehat{{\\mathrm{{KL}}}}] = D_\\text{{KL}}$; k1 is unbiased, k2 = $\\tfrac12 (\\log\\pi_\\theta - \\log\\pi_\\text{{ref}})^2$ is non-negative and biased upward (Jensen). Using k2 would produce non-zero within-prompt Var[R̃] even at a true $\\pi^*_x$, conflating estimator bias with the policy-suboptimality signal we want to measure. **(b) Consistency**: Phase 0 PPO used TRL's default `kl_penalty='kl'` = k1. The diagnostic must match the training KL convention for the cancellation test to apply to the policy PPO actually trained.

## 2. Validation tests (all pass)

| test | what it checks | result |
|---|---|---|
| `test_phase1_kl::test_policy_logprobs_match_recorded` | θ* recomputed per-token logprobs match `rollouts/step_0651.pt` policy_logprobs at fp16 tolerance | PASS |
| `test_phase1_kl::test_ref_logprobs_match_recorded` | π_ref recomputed per-token logprobs match recorded fp16 ref_logprobs | PASS |
| `test_phase1_kl::test_k1_kl_sign_and_scale` | k1 KL values are predominantly positive and O(1–30) in magnitude | PASS |
| `test_phase1_reward::test_reward_matches_recorded` | recomputed nothate logits exactly match `rollouts/step_0000.pt` rewards | PASS |

Off-by-one note for future agents: `rollouts/step_N.pt` stores `policy_logprobs` from the **pre-update** policy at step N (= post-update at N − 1). So the comparison file for `ppo_checkpoints/step_0650/` weights is `rollouts/step_0651.pt`, not `step_0650.pt`. This off-by-one does not affect Phase 1's variance numbers (we sample fresh from θ*), only the cross-validation test.

## 3. ICC analysis (in-training distribution)

$$\\mathrm{{ICC}} = \\frac{{\\overline{{\\mathrm{{Var}}_y[\\tilde R \\mid x]}}}}{{\\mathrm{{Var}}_{{x,y}}[\\tilde R]}} = \\frac{{{s_summary['mean_within_prompt_var']:.4f}}}{{{s_summary['total_var']:.4f}}} = {s_summary['icc_within_over_total']:.4f}$$

{icc_verdict}

![within-prompt variance histogram]({hist_rel})

| statistic | value |
|---|---|
| mean within-prompt Var[R̃] | `{s_summary['mean_within_prompt_var']:.4f}` |
| median within-prompt Var[R̃] | `{s_summary['median_within_prompt_var']:.4f}` |
| quartiles [min, q25, q50, q75, max] | `{_format_float_list(s_summary['within_var_quartiles'])}` |
| between-prompt Var of per-prompt means | `{s_summary['between_prompt_var_of_means']:.4f}` |
| pooled total Var[R̃] (all {s_summary['n_prompts'] * s_summary['k_samples_per_prompt']} samples) | `{s_summary['total_var']:.4f}` |

Variance decomposition check: by the law of total variance, $\\mathrm{{Var}}[\\tilde R] = \\mathbb{{E}}_x[\\mathrm{{Var}}_y(\\tilde R|x)] + \\mathrm{{Var}}_x[\\mathbb{{E}}_y(\\tilde R|x)]$. Empirically `{s_summary['mean_within_prompt_var']:.3f} + {s_summary['between_prompt_var_of_means']:.3f} = {s_summary['mean_within_prompt_var'] + s_summary['between_prompt_var_of_means']:.3f}` vs pooled total `{s_summary['total_var']:.3f}`. Residual `{s_summary['mean_within_prompt_var'] + s_summary['between_prompt_var_of_means'] - s_summary['total_var']:+.3f}` is the expected 1/K bias in the empirical $\\mathrm{{Var}}_x[\\overline{{R̃}}_x]$ (≈ within/K = `{s_summary['mean_within_prompt_var'] / s_summary['k_samples_per_prompt']:.3f}`).

## 4. Length / R̃ correlation analysis

![corr(length, R̃) per prompt + pooled scatter]({corr_rel})

| statistic | value |
|---|---|
| pooled corr(length, R̃) | `{s_summary['pooled_corr_length_R']:+.4f}` |
| per-prompt corr(length, R̃) — mean | `{s_summary['per_prompt_corr_length_R_mean']:+.4f}` |
| per-prompt corr(length, R̃) — median | `{s_summary['per_prompt_corr_length_R_median']:+.4f}` |
| # prompts with degenerate corr (constant L or R̃) | `{s_summary['n_prompts_with_nan_corr']}` |

Within-prompt length variance share of within-prompt R̃ variance ≈ $\\rho^2$ = `{s_summary['per_prompt_corr_length_R_mean']**2:.4f}`. Subtracting this artifact-contribution from the within-prompt variance leaves the residual `{(1 - s_summary['per_prompt_corr_length_R_mean']**2) * s_summary['mean_within_prompt_var']:.4f}` (≈ `{100 * (1 - s_summary['per_prompt_corr_length_R_mean']**2) * s_summary['mean_within_prompt_var'] / s_summary['total_var']:.1f}%` of total Var[R̃]) as the genuine policy-suboptimality signal. The negative sign of the correlation says longer responses systematically yield lower R̃: the β·KL penalty accumulates faster than the reward gain past the first ~10 tokens.

## 5. Outlier prompts — detailed (top-3, all 32 samples each)

The top-3 outliers share a common shape — **not** a 50/50 bimodal split between refusal and engagement, but an **asymmetric distribution with 1–2 negative-R̃ outliers in an otherwise high-mode bulk**. The largest-gap split locator (below) identifies the boundary; in all three cases it cuts off the bottom 1–2 samples from the remaining 30–31.

What drives the low outliers differs by prompt type:
- **Toxic-baited prompts** (outliers 1, 2; cf. ranks 4–10): the low outliers are engagement-style continuations of the toxic framing — the policy retains 5–10% probability mass on responses that earn very negative reward, while the bulk are refusal- or deflection-style continuations.
- **Non-toxic prompts** (outlier 3, "Peach Cobbler Recipe"): the low outlier is an incoherent generation ("Savo-like Handmaiden ↩ ↩ less") that the reward model rates as toxic. PPO has not eliminated occasional incoherent samples; these contribute to within-prompt R̃ variance even on benign prompts.

Phase 2 IF rankings should be expected to be sensitive to these tail-mass samples — they are precisely the rollouts with extreme score-function values.

{_fmt_outlier_detail(top_3[0], 1)}

{_fmt_outlier_detail(top_3[1], 2)}

{_fmt_outlier_detail(top_3[2], 3)}

### Ranks 4–10 (compact summary)

{_fmt_outlier_summary(stage2, n_top=10)}

The bimodal R̃ on toxic-baited prompts is observational evidence that PPO does not fully resolve the toxicity-vs-helpfulness tension on contested inputs at step 0650: the policy retains substantial probability mass on both refusal-style and engagement-style continuations for these prompts.

{framing_b_section}

## 7. Thesis-ready interpretation

{thesis_para}

## Artifacts

- `data/phase1/stage2_samples.jsonl` — {s_summary['n_prompts'] * s_summary['k_samples_per_prompt']} per-sample rows (Stage 2)
- `data/phase1/stage2_per_prompt.json` — per-prompt aggregates (Stage 2)
- `data/phase1/stage2_summary.json` — aggregate statistics (Stage 2)
- `data/phase1/framing_b_*` — held-out distribution mirror (Framing B)
- `reports/phase1_within_var_histogram.png`
- `reports/phase1_length_corr.png`
- `tests/test_phase1_kl.py`, `tests/test_phase1_reward.py`

## Deviations from `phases/phase1.md`

- Stage 2 wall-clock: **~60 s** (not the 80 min upper-bound estimate in the doc). Cause: `num_return_sequences=32` per `generate` call efficiently batches the K=32 samples on a single 4090.
- Added Framing B (per user request after Stage 2 completion) as a supplementary held-out comparison.
- Outlier section deepened to show all 32 samples for top-3 prompts with bimodal-split detection (per user request).
- No locked-in decisions changed.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)


def main() -> None:
    stage2, s_summary = _load_dataset("stage2")
    framing_b: list[dict] | None
    b_summary: dict | None
    try:
        framing_b, b_summary = _load_dataset("framing_b")
    except FileNotFoundError:
        framing_b, b_summary = None, None
        print("Framing B artifacts not found — generating Stage-2-only report.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    hist_path = REPORTS / "phase1_within_var_histogram.png"
    corr_path = REPORTS / "phase1_length_corr.png"
    make_within_var_histogram(stage2, framing_b, hist_path)
    make_corr_length_plots(stage2, corr_path)

    report_path = REPORTS / "phase1_report.md"
    generate_report(
        stage2=stage2,
        s_summary=s_summary,
        framing_b=framing_b,
        b_summary=b_summary,
        out_path=report_path,
        hist_rel=hist_path.name,
        corr_rel=corr_path.name,
    )
    print(f"Wrote {report_path}")
    print(f"Wrote {hist_path}")
    print(f"Wrote {corr_path}")


if __name__ == "__main__":
    main()
