"""Block C4 — top-10% rollout analysis (layer-9, fixed factors, per-token Delta).

Computes, over ALL 20,832 training rollouts (steps 0..650), the layer-9 influence
for the robust toxicity direction and f_seq, both BASELINE (paper IF, -p^T s_m)
and DELTA-CORRECTED (first-order, -(p+q)^T s_m, q = F^-1 (Delta_tok~ p)). Then
the top-10% analysis the user asked for (judgement A in the plan):

  * stability: Jaccard(top-10% baseline, top-10% corrected) per target
    -> does the (kill-sized) Delta reorder the *coarse* top-10% membership, or
       only the noisy middle?
  * target agreement: Jaccard(top-10% toxic, top-10% f_seq)
    -> do two different eval targets point at the same rollouts?
  * profile: reward (RoBERTa nothate logit; LOWER = more toxic), PPO step,
    response length — for top-10% vs bottom-10% vs random.  Tests the user's
    hypothesis "the top-10% are the most toxic / most extreme".
  * dumps the top-10% prompt/response TEXTS to a file for the user to read
    (the human-in-the-loop stop point — stats are reported, texts held).

Reuses the gate-2 driver's validated helpers (factors, f_seq, robust direction,
per-token Delta). NO cache. Checkpoints the influence arrays so the analysis can
re-run without re-scoring.

Run (base env, GPU):  python3 src/run_gptneo_c4_top10.py [--n-prompts 200] [--k 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import inverse_hvp  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402
from phase4.config import BETA, EVAL_PROMPTS_PATH, SEED  # noqa: E402
from phase4.e3_delta import DeltaSample, delta_vp_per_prompt_tok  # noqa: E402
from phase4.models import load_models  # noqa: E402
from phase4.sampling import sample_on_policy_pool  # noqa: E402
from phase5.toxicity_direction import build_g_f, load_lee_pairs  # noqa: E402

# reuse the validated f_seq helper from the gate-2 driver
from run_gptneo_gate2_killcheck import compute_g_seq, REWARD_MODEL  # noqa: E402

LAYER = "transformer.h.9.mlp.c_proj"
FACT = Path("data/ekfac/influence/run_step0650_refix")     # gate-2 saved factors here
OUT = Path("data/ekfac/influence/c4_top10")
ROLL = Path("data/rollouts")
REPORT = Path("reports/notes/c4_top10.json")
TEXTS = Path("reports/notes/c4_top10_texts.json")          # held for the user
ROLLOUT_STEPS = range(0, 651)


def load_all_rollouts(max_rollouts: int | None = None):
    rolls, index, meta = [], [], []
    for step in ROLLOUT_STEPS:
        p = ROLL / f"step_{step:04d}.pt"
        raw = torch.load(p, map_location="cpu", weights_only=False)
        for i, r in enumerate(raw):
            rolls.append(Rollout(prompt_ids=r["prompt_token_ids"].long(),
                                 response_ids=r["response_token_ids"].long(),
                                 reward=float(r["reward"]), step=int(r["step"])))
            index.append((step, i))
            meta.append({"step": int(r["step"]), "reward": float(r["reward"]),
                         "prompt": r.get("prompt_text", ""), "response": r.get("response_text", ""),
                         "resp_len": int(r["response_token_ids"].numel())})
        if max_rollouts is not None and len(rolls) >= max_rollouts:
            break
    return rolls, index, meta


def jaccard_topk(a, b, k, by_abs=False):
    fa, fb = (np.abs(a), np.abs(b)) if by_abs else (a, b)
    ta = set(np.argsort(-fa)[:k].tolist())
    tb = set(np.argsort(-fb)[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def profile(idx, meta):
    rew = np.array([meta[i]["reward"] for i in idx])
    stp = np.array([meta[i]["step"] for i in idx])
    ln = np.array([meta[i]["resp_len"] for i in idx])
    return {"reward_mean": float(rew.mean()), "reward_median": float(np.median(rew)),
            "step_mean": float(stp.mean()), "step_median": float(np.median(stp)),
            "len_mean": float(ln.mean()), "n": len(idx)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-seq-prompts", type=int, default=100)
    ap.add_argument("--n-lee", type=int, default=100)
    ap.add_argument("--resume-scores", action="store_true", help="load saved I arrays, skip scoring")
    ap.add_argument("--max-rollouts", type=int, default=None, help="cap rollouts scored (smoke)")
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EKFACConfig(layer_name=LAYER)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"C4 top-10% analysis (layer-9, fixed factors, per-token Delta). dev {dev}")

    models = load_models(dev, policy_requires_grad=False, policy_attn_implementation="eager")
    layer = models.policy.get_submodule(LAYER)
    layer.weight.requires_grad_(True)
    weight = layer.weight

    # ---- factors (reuse gate-2's on-policy fixed factors) ----
    if (FACT / "Q_A.pt").exists():
        print("  loading on-policy fixed factors from gate-2 run ...")
        Q_A = torch.load(FACT / "Q_A.pt", map_location=dev).double()
        Q_S = torch.load(FACT / "Q_S.pt", map_location=dev).double()
        Lam = torch.load(FACT / "Lambda.pt", map_location=dev).double()
    else:
        raise FileNotFoundError(f"{FACT}/Q_A.pt not found — run gate-2 driver first")

    targets_order = ["toxic_subtracted", "seq", "toxic_only"]

    if args.resume_scores and (OUT / "I_baseline.pt").exists():
        print("  resuming saved influence arrays ...")
        I_base = torch.load(OUT / "I_baseline.pt")
        I_corr = torch.load(OUT / "I_corrected.pt")
        index = torch.load(OUT / "rollout_index.pt").tolist()
        meta = json.loads((OUT / "meta.json").read_text())
    else:
        # ---- directions ----
        print("\nEval directions ...")
        from trl import AutoModelForCausalLMWithValueHead
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        ep = json.loads(EVAL_PROMPTS_PATH.read_text())
        all_prompts = ep["prompt_texts"]
        model_vh = AutoModelForCausalLMWithValueHead.from_pretrained(
            "data/ppo_checkpoints/step_0650").to(dev).eval()
        rob_tok = AutoTokenizer.from_pretrained(REWARD_MODEL)
        rob = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).to(dev).eval()
        g_seq, _ = compute_g_seq(models.policy, model_vh, models.tokenizer, rob, rob_tok,
                                 all_prompts[: args.n_seq_prompts], layer, cfg, dev, K=8, seed=SEED)
        del model_vh, rob, rob_tok
        torch.cuda.empty_cache()
        gf = build_g_f(models.policy, weight, models.tokenizer, load_lee_pairs(args.n_lee), dev)
        g_dir = {"toxic_subtracted": gf.g_f_subtracted.double().to(dev),
                 "seq": g_seq.double(),
                 "toxic_only": gf.g_f_toxic_only.double().to(dev)}
        p_f = {t: inverse_hvp(g, Q_A, Q_S, Lam, damping_floor=cfg.damping_floor,
                              damping_alpha=cfg.damping_alpha) for t, g in g_dir.items()}

        # ---- Delta_tok~ p (per-token, layer-9) -> q = F^-1 (Delta~ p) ----
        print("\nOn-policy pool + per-token Delta -> q ...")
        rng = np.random.RandomState(SEED)
        pick = sorted(rng.choice(len(all_prompts), size=args.n_prompts, replace=False).tolist())
        pool = sample_on_policy_pool([all_prompts[i] for i in pick], args.k, seed_base=SEED,
                                     models=models, beta=BETA, device=dev)
        by_p = defaultdict(list)
        for s in pool:
            by_p[s.prompt_idx].append(s)
        samples = []
        for sl in by_p.values():
            mR = float(np.mean([s.R_tilde for s in sl]))
            for s in sl:
                samples.append(DeltaSample(prompt_idx=s.prompt_idx, prompt_ids=s.prompt_ids,
                                           response_ids=s.response_ids, A=float(s.R_tilde - mR)))
        vs = [p_f[t] for t in targets_order]
        d_p, _, N_tok, _ = delta_vp_per_prompt_tok(samples, models.policy, layer, BETA, vs, dev,
                                                   store_device="cpu")
        q_f = {}
        for i, t in enumerate(targets_order):
            delta_p = d_p[i].to(dev).double().mean(0)          # Delta_tok~ p_f
            q_f[t] = inverse_hvp(delta_p, Q_A, Q_S, Lam, damping_floor=cfg.damping_floor,
                                 damping_alpha=cfg.damping_alpha)
        print(f"  q built ({time.time()-t0:.0f}s)")

        # ---- stream baseline + corrected over all rollouts ----
        print("\nLoading all rollouts ...")
        rolls, index, meta = load_all_rollouts(args.max_rollouts)
        N = len(rolls)
        print(f"  {N} rollouts. Scoring (one backward each) ...")
        I_base = {t: torch.zeros(N, dtype=torch.float64) for t in targets_order}
        I_corr = {t: torch.zeros(N, dtype=torch.float64) for t in targets_order}
        pq = {t: (p_f[t] + q_f[t]) for t in targets_order}
        for m, r in enumerate(rolls):
            s_m = compute_per_sample_grad(models.policy, layer, r, cfg, device=dev).double()
            for t in targets_order:
                I_base[t][m] = influence_score_klrl(s_m, p_f[t])
                I_corr[t][m] = influence_score_klrl(s_m, pq[t])
            if (m + 1) % 4000 == 0:
                print(f"    scored {m+1}/{N} ({time.time()-t0:.0f}s)")
        torch.save(I_base, OUT / "I_baseline.pt")
        torch.save(I_corr, OUT / "I_corrected.pt")
        torch.save(torch.tensor(index, dtype=torch.int32), OUT / "rollout_index.pt")
        (OUT / "meta.json").write_text(json.dumps(meta))
        print(f"  scored + saved ({time.time()-t0:.0f}s)")

    # ---- top-10% analysis ----
    N = len(meta)
    k10 = N // 10
    res = {"n_rollouts": N, "k_top10pct": k10, "targets": {}}
    rng = np.random.RandomState(0)
    rand_idx = rng.choice(N, size=k10, replace=False).tolist()
    res["random_profile"] = profile(rand_idx, meta)
    res["all_profile"] = profile(list(range(N)), meta)

    top10 = {}
    for t in targets_order:
        b = np.asarray(I_base[t]); c = np.asarray(I_corr[t])
        from scipy.stats import spearmanr
        top_b = np.argsort(-b)[:k10]; top_c = np.argsort(-c)[:k10]
        top_b_abs = np.argsort(-np.abs(b))[:k10]
        bot_b = np.argsort(b)[:k10]
        top10[t] = {"signed": top_b, "abs": top_b_abs}
        res["targets"][t] = {
            "spearman_base_vs_corr": float(spearmanr(b, c).statistic),
            "jaccard_top10_base_vs_corr_signed": jaccard_topk(b, c, k10),
            "jaccard_top10_base_vs_corr_abs": jaccard_topk(b, c, k10, by_abs=True),
            "profile_top10_signed": profile(top_b.tolist(), meta),
            "profile_top10_abs": profile(top_b_abs.tolist(), meta),
            "profile_bottom10_signed": profile(bot_b.tolist(), meta),
        }
    # target agreement (toxic_subtracted vs seq), top-10% by |I|
    res["jaccard_toxic_vs_seq_top10_abs"] = jaccard_topk(
        np.abs(np.asarray(I_base["toxic_subtracted"])), np.abs(np.asarray(I_base["seq"])), k10)
    res["jaccard_toxic_vs_seq_top10_signed"] = jaccard_topk(
        np.asarray(I_base["toxic_subtracted"]), np.asarray(I_base["seq"]), k10)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(res, indent=2))

    # ---- dump top-10% TEXTS (held for the user) ----
    texts = {}
    for t in targets_order:
        for cut, idxs in [("top10_signed", top10[t]["signed"]), ("top10_abs", top10[t]["abs"])]:
            rows = []
            for i in idxs[:40]:
                mi = meta[int(i)]
                rows.append({"I_base": float(np.asarray(I_base[t])[int(i)]),
                             "step": mi["step"], "reward": round(mi["reward"], 3),
                             "prompt": mi["prompt"][:120], "response": mi["response"][:160]})
            texts[f"{t}__{cut}"] = rows
    TEXTS.write_text(json.dumps(texts, indent=2, ensure_ascii=False))

    # ---- console summary ----
    print("\n" + "=" * 70)
    print(f"all rollouts: reward_mean={res['all_profile']['reward_mean']:.3f}  "
          f"step_mean={res['all_profile']['step_mean']:.0f}")
    print(f"random 10%:   reward_mean={res['random_profile']['reward_mean']:.3f}  "
          f"step_mean={res['random_profile']['step_mean']:.0f}")
    for t in targets_order:
        r = res["targets"][t]
        ps, pa = r["profile_top10_signed"], r["profile_top10_abs"]
        print(f"\n[{t}]  Spearman(base,corr)={r['spearman_base_vs_corr']:.3f}  "
              f"J@top10 base-vs-corr signed={r['jaccard_top10_base_vs_corr_signed']:.2f} "
              f"abs={r['jaccard_top10_base_vs_corr_abs']:.2f}")
        print(f"   top10%(signed) reward_mean={ps['reward_mean']:.3f} step_mean={ps['step_mean']:.0f} "
              f"len={ps['len_mean']:.1f}")
        print(f"   top10%(|I|)    reward_mean={pa['reward_mean']:.3f} step_mean={pa['step_mean']:.0f} "
              f"len={pa['len_mean']:.1f}")
    print(f"\nJaccard(toxic_subtracted vs seq) top10% |I|={res['jaccard_toxic_vs_seq_top10_abs']:.2f} "
          f"signed={res['jaccard_toxic_vs_seq_top10_signed']:.2f}")
    print(f"\nwrote {REPORT}  +  {TEXTS} (texts held for user)  total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
