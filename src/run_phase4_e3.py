"""Phase 4 E3 driver — the Delta error-budget (central experiment).

Pipeline:
  1. Sample 200 prompts x 16 on-policy rollouts at theta* (seed 42, in-dist).
     Compute R_tilde and same-prompt advantage A.
  2. For each target f in {f_seq, C1, C2, C3}: reuse cached p_f = F^-1 g_f.
  3. Matrix-free Delta~ p_f on the pool (per-prompt vectors), U-statistic
     ||Delta~ p_f||^2, thermometer ||Delta~ p_f|| / ||g_f||.
  4. Left-tail check: do the bottom-5%-A samples dominate Delta?
  5. Decision tree on the four thermometers:
       all < 0.1            -> transition; full corrected ranking (4 targets)
       any > 0.3            -> finish that target's ranking, STOP for review
       otherwise (0.1-0.3)  -> full corrected ranking (4 targets), report
  Corrected ranking reuses the production g_scaled_z cache (projection-only).

Outputs: data/phase4/e3_delta_budget.json, data/phase4/e3_onpolicy_pool.jsonl,
reports/phase4/e3_*.png (optional).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from phase4.config import BETA, EVAL_PROMPTS_PATH, PHASE4_DATA_DIR, SEED  # noqa: E402
from phase4.e2_stationarity import jackknife_ci, naive_quadratic  # noqa: E402
from phase4.e3_delta import (  # noqa: E402
    DeltaSample,
    corrected_scores_from_cache,
    delta_vp_per_prompt,
    project_eval_vector,
)
from phase4.grad_phi import get_phi_weight  # noqa: E402
from phase4.models import load_models  # noqa: E402
from phase4.sampling import sample_on_policy_pool  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EKFAC_RUN = REPO / "data" / "ekfac" / "influence" / "run_step0650"
CACHE_DIR = REPO / "data" / "ekfac" / "cache" / "g_scaled_step0650_layer9"

TARGETS = ["seq", "toxic_C1", "toxic_C2", "toxic_C3"]
N_PROMPTS = 200
K_PER_PROMPT = 16
N_ROLLOUTS = 20832

SMOKE = {"n_prompts": 4, "k": 4, "targets": ["seq"]}


def _pick_prompts(n: int, seed: int) -> list[str]:
    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    allp = ep["prompt_texts"]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(allp), size=n, replace=False).tolist())
    return [allp[i] for i in idx]


def _build_delta_samples(pool) -> list[DeltaSample]:
    """Attach same-prompt advantage A_i = R_tilde_i - mean_y R_tilde_x."""
    by_prompt: dict[int, list] = defaultdict(list)
    for s in pool:
        by_prompt[s.prompt_idx].append(s)
    out: list[DeltaSample] = []
    for samples in by_prompt.values():
        Rs = np.array([s.R_tilde for s in samples], dtype=np.float64)
        mean_R = float(Rs.mean())
        for s in samples:
            out.append(DeltaSample(
                prompt_idx=s.prompt_idx, prompt_ids=s.prompt_ids,
                response_ids=s.response_ids, A=float(s.R_tilde - mean_R),
            ))
    return out


def _left_tail_check(diag: list[dict], pool, target_k: int = 0) -> dict:
    """Do the bottom-5%-by-A samples carry a disproportionate share of the
    Delta contribution mass (sum of per-sample contrib norms)?"""
    A = np.array([d["A"] for d in diag])
    norms = np.array([d["contrib_norm"][target_k] for d in diag])
    n = len(A)
    k5 = max(1, int(round(0.05 * n)))
    order_A = np.argsort(A)                       # ascending: most negative first
    bottom = order_A[:k5]                          # bottom-5% by A (E1 left tail)
    order_absA = np.argsort(-np.abs(A))
    top_absA = order_absA[:k5]                     # largest |A|
    total = float(norms.sum())
    # Map diag index -> pool sample for text (diag is in pass order; pool order
    # differs, so match by (prompt_idx, A) is unreliable — reuse response via a
    # parallel list built alongside diag in the driver instead).
    return {
        "n_samples": n,
        "frac_5pct": float(k5 / n),
        "bottom5_A_share_of_contrib": float(norms[bottom].sum() / total) if total > 0 else float("nan"),
        "top5_absA_share_of_contrib": float(norms[top_absA].sum() / total) if total > 0 else float("nan"),
        "bottom5_A_threshold": float(A[order_A[k5 - 1]]),
        "uniform_share_ref": float(k5 / n),
        "bottom5_indices": bottom.tolist(),
        "top5_absA_indices": top_absA.tolist(),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 E3")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force-full-ranking", action="store_true",
                    help="compute corrected ranking regardless of thermometers")
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else {"n_prompts": N_PROMPTS, "k": K_PER_PROMPT, "targets": TARGETS}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PHASE4_DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    prompts = _pick_prompts(cfg["n_prompts"], seed=SEED)
    print(f"E3{'(smoke)' if args.smoke else ''}: {cfg['n_prompts']} prompts x {cfg['k']} "
          f"on-policy rollouts at theta*, beta={BETA:.17g}; targets={cfg['targets']}")

    print("Loading models (eager attn for double-backward) ...")
    models = load_models(device, policy_requires_grad=False, policy_attn_implementation="eager")
    weight = get_phi_weight(models.policy)
    weight.requires_grad_(True)

    print("Sampling on-policy pool ...")
    pool = sample_on_policy_pool(prompts, cfg["k"], seed_base=SEED, models=models, beta=BETA, device=device)
    samples = _build_delta_samples(pool)
    print(f"pool: {len(pool)} samples, {cfg['n_prompts']} prompts")

    with open(PHASE4_DATA_DIR / f"e3_onpolicy_pool{suffix}.jsonl", "w") as f:
        for s, ds in zip(pool, samples):
            f.write(json.dumps({
                "prompt_idx": s.prompt_idx, "prompt_text": s.prompt_text,
                "response_text": models.tokenizer.decode(s.response_ids, skip_special_tokens=True),
                "reward": s.reward, "k1_kl": s.k1_kl, "R_tilde": s.R_tilde, "A": ds.A,
            }) + "\n")
    # Parallel list of (A, response_text) in *sample* order for the tail report.
    sample_meta = [{"A": ds.A, "prompt_idx": s.prompt_idx,
                    "prompt_text": s.prompt_text,
                    "response_text": models.tokenizer.decode(s.response_ids, skip_special_tokens=True)}
                   for s, ds in zip(pool, samples)]

    # Cached EK-FAC factors + p_f.
    print("Loading EK-FAC factors and p_f ...")
    Q_A = torch.load(EKFAC_RUN / "Q_A.pt", map_location=device).to(torch.float64)
    Q_S = torch.load(EKFAC_RUN / "Q_S.pt", map_location=device).to(torch.float64)
    tgt_files = {"seq": "seq", "toxic_C1": "toxic_C1", "toxic_C2": "toxic_C2", "toxic_C3": "toxic_C3"}
    p_f = {t: torch.load(EKFAC_RUN / f"p_{tgt_files[t]}.pt", map_location=device).to(torch.float64)
           for t in cfg["targets"]}
    g_f = {t: torch.load(EKFAC_RUN / f"g_{tgt_files[t]}.pt", map_location=device).to(torch.float64)
           for t in cfg["targets"]}
    g_f_norm = {t: float(g_f[t].norm().item()) for t in cfg["targets"]}

    vs = [p_f[t] for t in cfg["targets"]]
    print("Computing Delta~ p_f (matrix-free, double-backward) ...")
    d_p, order, diag = delta_vp_per_prompt(
        samples, models.policy, weight, BETA, vs, device, track_contribs=True
    )

    # Per-target thermometer.
    results = {"_meta": {"n_prompts": cfg["n_prompts"], "k_per_prompt": cfg["k"],
                         "beta": BETA, "targets": cfg["targets"], "seed": SEED},
               "targets": {}}
    thermometers = {}
    delta_p_proj = {}
    for k, t in enumerate(cfg["targets"]):
        g = d_p[k]
        norm_sq = jackknife_ci(g, None)             # U-stat ||Delta~ p||^2 + jackknife SE
        norm_sq_naive = naive_quadratic(g, None)
        dp_norm = float(np.sqrt(max(norm_sq["point"], 0.0)))
        therm = dp_norm / g_f_norm[t]
        thermometers[t] = therm
        # Two-half convergence: cosine of Delta~_A p and Delta~_B p over disjoint
        # prompt halves; low cosine => not converged (need more N,K).
        Np = g.shape[0]
        gh = g.reshape(Np, -1).to(torch.float64)
        half = Np // 2
        a = gh[:half].mean(0); b = gh[half:].mean(0)
        cos_half = float((a @ b / (a.norm() * b.norm() + 1e-30)).item())
        # Projected Delta~ p (for corrected scoring against cache).
        delta_p_total = g.to(torch.float64).mean(0)   # (d_out, d_in) = Delta~ p_f
        delta_p_proj[t] = project_eval_vector(delta_p_total, Q_A, Q_S)
        results["targets"][t] = {
            "g_f_norm": g_f_norm[t],
            "delta_p_norm_sq_ustat": norm_sq["point"],
            "delta_p_norm_sq_jackknife_se": norm_sq["jackknife_se"],
            "delta_p_norm_sq_naive": norm_sq_naive,
            "delta_p_norm": dp_norm,
            "thermometer_ratio": therm,
            "two_half_cosine": cos_half,
        }
        print(f"  [{t}] ||Delta~ p||={dp_norm:.4e} (naive {np.sqrt(max(norm_sq_naive,0)):.4e}), "
              f"||g_f||={g_f_norm[t]:.4e}  ->  THERMOMETER = {therm:.4f}  "
              f"(two-half cos={cos_half:.3f}, jackSE(||.||^2)={norm_sq['jackknife_se']:.2e})")

    # Left-tail dominance check (using f_seq contributions, target index 0).
    tail = _left_tail_check(diag, pool, target_k=0)
    # attach the heavy-tail samples' A + text
    tail_samples = []
    for idx in tail["bottom5_indices"][:15]:
        m = sample_meta[idx]
        tail_samples.append({"A": round(m["A"], 3), "contrib_norm_seq": round(diag[idx]["contrib_norm"][0], 4),
                             "prompt": m["prompt_text"][:50], "response": m["response_text"][:70]})
    results["left_tail_check"] = {**{k: v for k, v in tail.items()
                                      if k not in ("bottom5_indices", "top5_absA_indices")},
                                  "bottom5_A_samples": tail_samples}
    print(f"\n  left-tail: bottom-5%-by-A carry {tail['bottom5_A_share_of_contrib']*100:.1f}% of "
          f"Delta-contrib mass (uniform ref {tail['uniform_share_ref']*100:.1f}%); "
          f"top-5%-by-|A| carry {tail['top5_absA_share_of_contrib']*100:.1f}%")

    results["thermometers"] = thermometers
    max_therm = max(thermometers.values())
    min_therm = min(thermometers.values())
    # Decision tree.
    if args.smoke:
        decision = "smoke"
    elif max_therm > 0.3:
        decision = "kill_boundary_stop"   # any > 0.3
    elif max_therm < 0.1:
        decision = "transition_full_ranking"
    else:
        decision = "midband_full_ranking"
    results["decision"] = decision
    print(f"\n  THERMOMETERS: " + ", ".join(f"{t}={thermometers[t]:.4f}" for t in cfg['targets']))
    print(f"  decision: {decision} (max={max_therm:.4f}, min={min_therm:.4f})")

    out_path = PHASE4_DATA_DIR / f"e3_delta_budget{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")

    # Corrected ranking.
    do_ranking = (not args.smoke) and (
        args.force_full_ranking or decision in ("transition_full_ranking", "midband_full_ranking")
    )
    ranking_targets = cfg["targets"]
    if (not args.smoke) and decision == "kill_boundary_stop":
        # Only the offending target(s).
        ranking_targets = [t for t in cfg["targets"] if thermometers[t] > 0.3]
        do_ranking = True

    if do_ranking:
        print(f"\nCorrected ranking for {ranking_targets} (streaming {N_ROLLOUTS} cached g_scaled_z) ...")
        I_cached = {t: torch.load(EKFAC_RUN / f"I_{tgt_files[t]}.pt", map_location="cpu").numpy()
                    for t in ranking_targets}
        for t in ranking_targets:
            gfp = project_eval_vector(g_f[t], Q_A, Q_S)
            baseline, correction = corrected_scores_from_cache(
                gfp, delta_p_proj[t], CACHE_DIR, N_ROLLOUTS, device
            )
            I_corr = baseline + correction
            I_base = I_cached[t]
            # baseline sanity: our recomputed baseline must match cached I_*.
            base_match = float(np.max(np.abs(baseline - I_base)))
            rel_corr = np.abs(correction) / (np.abs(I_base) + 1e-12)
            res = {
                "spearman_base_vs_corr": _spearman(I_base, I_corr),
                "pearson_base_vs_corr": float(np.corrcoef(I_base, I_corr)[0, 1]),
                "jaccard_top10": _jaccard_topk(I_base, I_corr, 10),
                "jaccard_top50": _jaccard_topk(I_base, I_corr, 50),
                "jaccard_top100": _jaccard_topk(I_base, I_corr, 100),
                "correction_rel_magnitude_median": float(np.median(rel_corr)),
                "correction_rel_magnitude_p90": float(np.percentile(rel_corr, 90)),
                "baseline_vs_cached_max_abs_diff": base_match,
            }
            results["targets"][t]["ranking"] = res
            print(f"  [{t}] Spearman={res['spearman_base_vs_corr']:.4f}  "
                  f"Pearson={res['pearson_base_vs_corr']:.4f}  "
                  f"J@10={res['jaccard_top10']:.2f} J@50={res['jaccard_top50']:.2f} "
                  f"J@100={res['jaccard_top100']:.2f}  "
                  f"|corr|/|base| med={res['correction_rel_magnitude_median']:.3f}  "
                  f"(baseline≈cached: max diff {base_match:.2e})")
        out_path.write_text(json.dumps(results, indent=2))
        print(f"updated {out_path}")

    print("\n=== E3 done ===  decision:", decision)


if __name__ == "__main__":
    main()
