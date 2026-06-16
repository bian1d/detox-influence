"""INDEPENDENT AUDIT 4b — REAL-SCALE damage of the Stage-1 pseudo-label
off-by-one on the production f_seq / f_toxic influence RANKING.

Audit 3b proved the off-by-one mechanism and measured ~0.07 rel magnitude /
0.998 Spearman damage at toy scale. This script measures it on the real
GPT-Neo-125M θ* and the production eval gradients, the quantity the thesis
actually rests on.

Method (apples-to-apples on the SAME rollout subset):
  1. Sample a subset of M production rollouts spanning steps 0..650.
  2. Accumulate EK-FAC factors TWICE over that subset:
       (buggy)     project accumulate_AS + fit_lambda  (pseudo[P:])
       (corrected) my reimplementation with pseudo[P-1:T-1] row-aligned labels
     Everything else identical (same seeds, same per-token mean, same eigdecomp,
     same two-level damping). The ONLY difference is the label alignment.
  3. Load the cached production g_seq, g_toxic_C1. Compute p = F^-1 g both ways.
  4. Score a large sample of rollouts (fresh per-sample grads) under both
     p's; compare I_buggy vs I_corrected: Spearman, Pearson, top-k Jaccard,
     per-rollout relative correction magnitude.
  5. Also report factor-level deltas: ||S_bug - S_corr||/||S_corr||,
     ||Lambda_bug - Lambda_corr||/||Lambda_corr||, ||p_bug - p_corr||/||p_corr||,
     and the thermometer-relevant ||p|| change.

Subset-vs-full guard: also load the FULL production factors and report
||S_subset_bug - S_full||/||S_full|| so we know the M-rollout subset is a
faithful stand-in for the 20,832-rollout production factors.

Writes JSON to reports/notes/audit4b_offbyone_damage.json and prints a summary.

Run (base env, GPU):
  python3 tests/audit_4b_offbyone_damage.py --n-rollouts 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout, build_input_and_labels, load_rollout_file  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda, inverse_hvp  # noqa: E402
from ekfac.factors import accumulate_AS, sample_pseudo_labels  # noqa: E402
from ekfac.hooks import capture_c_proj  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402

RUN = Path("data/ekfac/influence/run_step0650")
CKPT = Path("data/ppo_checkpoints/step_0650")
ROLL = Path("data/rollouts")
OUT = Path("reports/notes/audit4b_offbyone_damage.json")


def corrected_accumulate_AS(model, layer, rollouts, cfg, device, seed):
    """Stage 1A with ROW-ALIGNED pseudo labels: row t (predicting position
    t+1) gets a label sampled from softmax(logits[t]) — i.e. response_labels
    = pseudo[P-1:T-1] instead of pseudo[P:]. Otherwise identical to
    factors.accumulate_AS (per-token mean, fp64, symmetrised)."""
    d_out, d_in = layer.weight.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    A_sum = torch.zeros(d_in, d_in, dtype=torch.float64, device=device)
    S_sum = torch.zeros(d_out, d_out, dtype=torch.float64, device=device)
    n_tok = 0
    model.eval()
    for roll in rollouts:
        P, R = roll.prompt_len, roll.response_len
        T = P + R
        ids = torch.cat([roll.prompt_ids, roll.response_ids]).to(device).unsqueeze(0)
        for p_ in model.parameters():
            if p_.grad is not None:
                p_.grad = None
        with capture_c_proj(layer) as cache:
            logits = model(ids).logits
            pseudo = sample_pseudo_labels(logits[0], generator=gen)        # (T,)
            # ROW-ALIGNED: labels[P:T] = pseudo[P-1:T-1]
            _, labels = build_input_and_labels(
                roll.prompt_ids.to(device), roll.response_ids.to(device),
                response_labels=pseudo[P - 1 : T - 1], ignore_index=cfg.ignore_index,
            )
            loss = TF.cross_entropy(logits[0, :-1].float(), labels[1:],
                                    ignore_index=cfg.ignore_index, reduction="sum")
            loss.backward()
        m_resp = cache.m[0, P - 1 : T - 1, :].double()
        d_resp = cache.delta[0, P - 1 : T - 1, :].double()
        A_sum += m_resp.T @ m_resp
        S_sum += d_resp.T @ d_resp
        n_tok += R
    A = A_sum / n_tok
    S = S_sum / n_tok
    return 0.5 * (A + A.T), 0.5 * (S + S.T), n_tok


def corrected_fit_lambda(model, layer, rollouts, cfg, Q_A, Q_S, device, seed):
    """Stage 1B Lambda with the same row-aligned pseudo labels."""
    d_in, d_out = Q_A.shape[0], Q_S.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    Lam = torch.zeros(d_in, d_out, dtype=torch.float64, device=device)
    QAe, QSe = Q_A.double(), Q_S.double()
    n_tok = 0
    model.eval()
    for roll in rollouts:
        P, R = roll.prompt_len, roll.response_len
        T = P + R
        ids = torch.cat([roll.prompt_ids, roll.response_ids]).to(device).unsqueeze(0)
        for p_ in model.parameters():
            if p_.grad is not None:
                p_.grad = None
        with capture_c_proj(layer) as cache:
            logits = model(ids).logits
            pseudo = sample_pseudo_labels(logits[0], generator=gen)
            _, labels = build_input_and_labels(
                roll.prompt_ids.to(device), roll.response_ids.to(device),
                response_labels=pseudo[P - 1 : T - 1], ignore_index=cfg.ignore_index,
            )
            loss = TF.cross_entropy(logits[0, :-1].float(), labels[1:],
                                    ignore_index=cfg.ignore_index, reduction="sum")
            loss.backward()
        m_resp = cache.m[0, P - 1 : T - 1, :].double()
        d_resp = cache.delta[0, P - 1 : T - 1, :].double()
        a_proj = m_resp @ QAe
        d_proj = d_resp @ QSe
        Lam += a_proj.pow(2).T @ d_proj.pow(2)
        n_tok += R
    return Lam / n_tok, n_tok


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def jaccard_topk(a, b, k, *, by_abs=True):
    fa = np.abs(a) if by_abs else a
    fb = np.abs(b) if by_abs else b
    ta = set(np.argsort(-fa)[:k].tolist())
    tb = set(np.argsort(-fb)[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rollouts", type=int, default=3000,
                    help="subset size for factor accumulation")
    ap.add_argument("--n-score", type=int, default=2000,
                    help="rollouts to score for ranking comparison")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EKFACConfig(layer_name="transformer.h.9.mlp.c_proj")
    print(f"AUDIT 4b: real-scale off-by-one damage  (torch {torch.__version__}, "
          f"numpy {np.__version__}, dev {dev})", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(CKPT.as_posix()).to(dev).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    layer = model.get_submodule(cfg.layer_name)
    layer.weight.requires_grad_(True)

    # ---- subset of production rollouts spanning the full range ----
    steps = list(range(0, 651))
    rng = np.random.default_rng(42)
    per_step = max(1, args.n_rollouts // len(steps))
    subset: list[Rollout] = []
    for s in steps:
        rolls = load_rollout_file(ROLL / f"step_{s:04d}.pt")
        take = min(per_step, len(rolls))
        idx = rng.choice(len(rolls), size=take, replace=False)
        subset.extend(rolls[i] for i in idx)
        if len(subset) >= args.n_rollouts:
            break
    subset = subset[: args.n_rollouts]
    print(f"  subset: {len(subset)} rollouts", flush=True)

    # ---- buggy factors (project code) ----
    print("  accumulating BUGGY factors (project accumulate_AS) ...", flush=True)
    A_b, S_b, ntb = accumulate_AS(model, layer, subset, cfg, device=dev)
    QAb, _, QSb, _ = eigendecompose(A_b, S_b)
    Lam_b, _ = fit_lambda(model, layer, subset, cfg, Q_A=QAb, Q_S=QSb, device=dev)
    print(f"    done ({time.time()-t0:.0f}s), n_tok={ntb}", flush=True)

    # ---- corrected factors (row-aligned labels) ----
    print("  accumulating CORRECTED factors (row-aligned labels) ...", flush=True)
    A_c, S_c, ntc = corrected_accumulate_AS(model, layer, subset, cfg, dev, cfg.seed)
    QAc, _, QSc, _ = eigendecompose(A_c, S_c)
    Lam_c, _ = corrected_fit_lambda(model, layer, subset, cfg, QAc, QSc, dev, cfg.seed + 1)
    print(f"    done ({time.time()-t0:.0f}s), n_tok={ntc}", flush=True)

    # ---- factor-level deltas ----
    relA = (A_b - A_c).norm().item() / A_c.norm().item()
    relS = (S_b - S_c).norm().item() / S_c.norm().item()
    # A is label-free, so relA should be ~0 (sanity that only S/Lambda shift).
    print(f"  ||A_bug-A_corr||/||A_corr|| = {relA:.4f} (expect ~0, A is label-free)", flush=True)
    print(f"  ||S_bug-S_corr||/||S_corr|| = {relS:.4f}", flush=True)

    # subset representativeness vs FULL production factors
    rep = {}
    try:
        S_full = torch.load(RUN / "S.pt", map_location=dev).double()
        rep["S_subsetbug_vs_full"] = (S_b - S_full).norm().item() / S_full.norm().item()
        print(f"  ||S_subset_bug - S_full||/||S_full|| = {rep['S_subsetbug_vs_full']:.4f} "
              f"(subset representativeness)", flush=True)
    except FileNotFoundError:
        pass

    # ---- p = F^-1 g for production eval grads ----
    floor, alpha = cfg.damping_floor, cfg.damping_alpha
    out: dict = {"n_rollouts_factor": len(subset), "n_tok": int(ntb),
                 "relA": relA, "relS": relS, "subset_rep": rep, "targets": {}}
    for tag in ["seq", "toxic_C1"]:
        gp = RUN / f"g_{tag}.pt"
        if not gp.exists():
            continue
        g = torch.load(gp, map_location=dev).double()
        p_b = inverse_hvp(g, QAb.double(), QSb.double(), Lam_b.double(),
                          damping_floor=floor, damping_alpha=alpha)
        p_c = inverse_hvp(g, QAc.double(), QSc.double(), Lam_c.double(),
                          damping_floor=floor, damping_alpha=alpha)
        rel_p = (p_b - p_c).norm().item() / p_c.norm().item()
        rel_pnorm = abs(p_b.norm().item() - p_c.norm().item()) / p_c.norm().item()
        cos_p = float((p_b.flatten() @ p_c.flatten()) /
                      (p_b.norm() * p_c.norm())).real if False else float(
            (p_b.flatten() @ p_c.flatten() / (p_b.norm() * p_c.norm())).item())
        out["targets"][tag] = {"rel_p": rel_p, "rel_pnorm": rel_pnorm, "cos_p": cos_p}
        print(f"  [{tag}] ||p_bug-p_corr||/||p_corr|| = {rel_p:.4f}, "
              f"||p|| change = {rel_pnorm:.4f}, cos(p_bug,p_corr) = {cos_p:.5f}", flush=True)

    # ---- ranking comparison on a scoring sample ----
    rng2 = np.random.default_rng(7)
    score_steps = rng2.choice(steps, size=min(args.n_score, len(steps)), replace=True)
    score_rolls: list[Rollout] = []
    file_cache: dict[int, list[Rollout]] = {}
    for s in sorted(set(score_steps.tolist())):
        file_cache[s] = load_rollout_file(ROLL / f"step_{s:04d}.pt")
    rng3 = np.random.default_rng(8)
    while len(score_rolls) < args.n_score:
        s = int(rng3.choice(steps))
        rolls = file_cache.get(s)
        if rolls is None:
            rolls = file_cache[s] = load_rollout_file(ROLL / f"step_{s:04d}.pt")
        score_rolls.append(rolls[int(rng3.choice(len(rolls)))])
    print(f"  scoring {len(score_rolls)} rollouts under both factor sets ...", flush=True)

    p_pairs = {tag: (inverse_hvp(torch.load(RUN / f"g_{tag}.pt", map_location=dev).double(),
                                 QAb.double(), QSb.double(), Lam_b.double(),
                                 damping_floor=floor, damping_alpha=alpha),
                     inverse_hvp(torch.load(RUN / f"g_{tag}.pt", map_location=dev).double(),
                                 QAc.double(), QSc.double(), Lam_c.double(),
                                 damping_floor=floor, damping_alpha=alpha))
               for tag in out["targets"]}
    scores = {tag: ([], []) for tag in p_pairs}
    for i, roll in enumerate(score_rolls):
        s_m = compute_per_sample_grad(model, layer, roll, cfg, device=dev).double()
        for tag, (p_b, p_c) in p_pairs.items():
            scores[tag][0].append(influence_score_klrl(s_m, p_b))
            scores[tag][1].append(influence_score_klrl(s_m, p_c))
        if (i + 1) % 500 == 0:
            print(f"    scored {i+1}/{len(score_rolls)} ({time.time()-t0:.0f}s)", flush=True)

    for tag, (Ib, Ic) in scores.items():
        Ib, Ic = np.array(Ib), np.array(Ic)
        rho = spearman(Ib, Ic)
        pear = float(np.corrcoef(Ib, Ic)[0, 1])
        rel_corr = np.abs(Ib - Ic) / (np.abs(Ic) + 1e-12)
        res = {
            "spearman_bug_vs_corr": rho,
            "pearson_bug_vs_corr": pear,
            "jaccard_top10": jaccard_topk(Ib, Ic, 10),
            "jaccard_top50": jaccard_topk(Ib, Ic, 50),
            "jaccard_top100": jaccard_topk(Ib, Ic, 100),
            "rel_correction_median": float(np.median(rel_corr)),
            "rel_correction_p90": float(np.percentile(rel_corr, 90)),
        }
        out["targets"][tag].update(res)
        print(f"  [{tag}] RANKING bug-vs-corrected: Spearman={rho:.4f} Pearson={pear:.4f} "
              f"J@10={res['jaccard_top10']:.2f} J@50={res['jaccard_top50']:.2f} "
              f"J@100={res['jaccard_top100']:.2f} |Δ|/|I| med={res['rel_correction_median']:.3f} "
              f"p90={res['rel_correction_p90']:.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}  (total {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
