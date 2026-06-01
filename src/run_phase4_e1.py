"""Phase 4 E1 driver.

Two independently runnable parts:

  --advantage   Pure-analysis: advantage A(x,y) distribution + within-prompt
                variance decomposition + figures.  No model, seconds.

  --icc-ci      Sampling-error band on ICC=0.892.  Loads theta*/ref/reward and
                redraws on-policy responses for a 20-prompt subset.  Minutes.

Outputs:
  data/phase4/e1_advantage_stats.json
  data/phase4/e1_icc_resample.json
  reports/phase4/e1_advantage_hist.png
  reports/phase4/e1_advantage_qq.png
  reports/phase4/e1_variance_decomposition.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from phase4.config import (  # noqa: E402
    BETA,
    BETA_STEP,
    ICC_K_SAMPLES,
    ICC_N_PROMPTS,
    ICC_N_RESEEDS,
    ICC_SEED_BASE,
    PHASE1_ICC,
    PHASE4_DATA_DIR,
    PHASE4_REPORT_DIR,
    STAGE2_PER_PROMPT,
    STAGE2_SUMMARY,
)
from phase4.e1_advantage import (  # noqa: E402
    advantage_distribution,
    assert_records_consistent,
    load_records,
    variance_decomposition,
)

SAMPLES_JSONL = STAGE2_PER_PROMPT.parent / "stage2_samples.jsonl"


def run_advantage() -> None:
    PHASE4_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHASE4_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records(STAGE2_PER_PROMPT, SAMPLES_JSONL)
    print(f"Loaded {len(records)} per-prompt records from {STAGE2_PER_PROMPT}")

    # Bit-exactness guard: stored R_tilde must equal reward - BETA*k1_kl.
    assert_records_consistent(records, BETA)
    print(f"R_tilde consistency OK at beta={BETA:.17g}")

    adv = advantage_distribution(records)
    decomp = variance_decomposition(records, BETA)
    print(
        f"\nadvantage: std={adv['std']:.3f}  skew={adv['skewness']:.3f}  "
        f"excess_kurt={adv['excess_kurtosis']:.3f}  "
        f"min={adv['min']:.2f}  max={adv['max']:.2f}"
    )
    agg = decomp["aggregate"]
    print(
        f"variance decomposition (weighted shares): "
        f"reward={agg['weighted_share_reward']:.3f}  "
        f"KL={agg['weighted_share_kl']:.3f}  "
        f"cov={agg['weighted_share_cov']:.3f}"
    )
    print(f"reconstruction max abs err: {decomp['reconstruction_max_abs_err']:.3e}")
    ll = decomp["length_lens"]
    print(
        f"length lens: corr(len,r)={ll['pooled_corr_len_r']:+.3f}  "
        f"corr(len,KL)={ll['pooled_corr_len_kl']:+.3f}  "
        f"corr(len,R)={ll['pooled_corr_len_R']:+.3f}  rho2={ll['pooled_rho2_len_R']:.3f}"
    )

    out = {
        "beta": BETA,
        "beta_step": BETA_STEP,
        "n_prompts": len(records),
        "advantage": adv,
        "variance_decomposition": {
            "aggregate": decomp["aggregate"],
            "length_lens": decomp["length_lens"],
            "reconstruction_max_abs_err": decomp["reconstruction_max_abs_err"],
            "per_prompt": decomp["per_prompt"],
        },
    }
    out_path = PHASE4_DATA_DIR / "e1_advantage_stats.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Figures (imported lazily so --icc-ci needn't import matplotlib).
    from phase4.e1_plots import (
        plot_advantage_hist,
        plot_advantage_qq,
        plot_variance_decomposition,
    )

    plot_advantage_hist(records, PHASE4_REPORT_DIR / "e1_advantage_hist.png")
    plot_advantage_qq(records, PHASE4_REPORT_DIR / "e1_advantage_qq.png")
    plot_variance_decomposition(
        records, BETA, PHASE4_REPORT_DIR / "e1_variance_decomposition.png"
    )
    print(f"Wrote 3 figures to {PHASE4_REPORT_DIR}")


def run_icc_ci(device: str) -> None:
    from phase4.e1_icc_ci import resample_icc, select_icc_subset
    from phase4.models import load_models

    PHASE4_DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(STAGE2_PER_PROMPT, SAMPLES_JSONL)
    assert_records_consistent(records, BETA)

    # Anchor: confirm we point at the run that produced ICC=0.892.
    full_summary = json.loads(STAGE2_SUMMARY.read_text())
    full_icc = full_summary["icc_within_over_total"]
    if abs(full_icc - PHASE1_ICC) > 1e-9:
        raise ValueError(
            f"stage2_summary ICC {full_icc} != expected {PHASE1_ICC}; wrong run?"
        )

    subset_idx = select_icc_subset(records, ICC_N_PROMPTS)
    print(f"ICC subset ({len(subset_idx)} prompts), keys:")
    for i in subset_idx:
        print(f"  pos={i:>3}  key={records[i]['prompt_key']:>3}  "
              f"var_R={records[i]['var_R_tilde']:.3f}  "
              f"text={records[i]['prompt_text'][:50]!r}")

    print(f"\nLoading models on {device} ...")
    models = load_models(device, policy_requires_grad=False)

    print(
        f"\nResampling: {ICC_N_RESEEDS} reseeds × {ICC_N_PROMPTS} prompts × "
        f"{ICC_K_SAMPLES} responses, beta={BETA:.17g}"
    )
    result = resample_icc(
        records=records,
        subset_idx=subset_idx,
        models=models,
        device=device,
        beta=BETA,
        n_reseeds=ICC_N_RESEEDS,
        k=ICC_K_SAMPLES,
        seed_base=ICC_SEED_BASE,
    )
    result["icc_phase1_full_100"] = full_icc
    result["beta"] = BETA

    print(
        f"\nICC on full-100 (Phase 1): {full_icc:.4f}\n"
        f"ICC on subset (Phase 1 data): {result['icc_phase1_on_subset']:.4f}\n"
        f"ICC resample: mean={result['icc_resample_mean']:.4f}  "
        f"std={result['icc_resample_std']:.4f}  "
        f"range=[{result['icc_resample_min']:.4f}, {result['icc_resample_max']:.4f}]  "
        f"95%CI(mean)={result['icc_resample_mean_ci95']}"
    )

    out_path = PHASE4_DATA_DIR / "e1_icc_resample.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 E1")
    ap.add_argument("--advantage", action="store_true", help="advantage + decomposition (no model)")
    ap.add_argument("--icc-ci", action="store_true", help="ICC resample band (loads models)")
    args = ap.parse_args()

    if not (args.advantage or args.icc_ci):
        raise SystemExit("Pass --advantage and/or --icc-ci")

    if args.advantage:
        run_advantage()
    if args.icc_ci:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_icc_ci(device)


if __name__ == "__main__":
    main()
