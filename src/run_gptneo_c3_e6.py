"""Block C3 / E6 — all-MLP-layer influence vs single-layer-9 (representativeness).

Builds EK-FAC factors for ALL MLP layers (12 blocks x {c_fc=W_1, c_proj=W_2})
on-policy with the fixed code, computes per-layer eval directions (f_seq + robust
baseline-subtracted toxicity), per-layer p_f = F^-1 g_f, then streams the
BASELINE influence over all 20,832 rollouts in two reductions:
  * all-MLP  = sum over all 24 layers of -<p_f[layer], s_m[layer]>
  * layer-9  = -<p_f[layer9 c_proj], s_m[layer9 c_proj]>  (the primary subspace)
and reports Spearman / top-k Jaccard between them. High agreement => the
single-layer-9 attribution is representative of the full MLP subspace (phase4
E6). Attention QKVO excluded (softmax breaks Kronecker; Grosse 2023).

NO cache. Checkpoints factors / directions / influence arrays.

Run (base env, GPU):  python3 src/run_gptneo_c3_e6.py [--budget N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import eigendecompose, inverse_hvp  # noqa: E402
from ekfac.multilayer import (  # noqa: E402
    accumulate_AS_multi, fit_lambda_multi, gptneo_mlp_layer_names,
    mlp_layer_dict, per_sample_grads_multi,
)
from eval_directions import build_g_f_multi  # noqa: E402
from phase4.config import EVAL_PROMPTS_PATH, SEED  # noqa: E402
from phase4.models import load_models  # noqa: E402
from phase5.sample_rollouts import generate_factor_rollouts  # noqa: E402
from phase5.toxicity_direction import load_lee_pairs  # noqa: E402
from run_gptneo_gate2_killcheck import _roberta_nothate_logit, _value_last, REWARD_MODEL  # noqa: E402

L9 = "transformer.h.9.mlp.c_proj"
FACT = Path("data/ekfac/influence/c3_multilayer")
OUT = Path("data/ekfac/influence/c3_e6")
ROLL = Path("data/rollouts")
REPORT = Path("reports/notes/c3_e6.json")


def compute_g_seq_multi(models, model_vh, rob, rob_tok, prompts, layers, weights, cfg, dev, *, K, seed):
    """Per-layer f_seq direction: g_seq[layer] = mean_{x,y~theta*} A * s(x,y)[layer]."""
    torch.manual_seed(seed)
    g = {nm: torch.zeros_like(w, dtype=torch.float64) for nm, w in weights.items()}
    n = 0
    tok = models.tokenizer
    pretrained = model_vh.pretrained_model
    pretrained.eval()
    for pi, ptext in enumerate(prompts):
        pid = tok(ptext, truncation=True, max_length=64, add_special_tokens=False,
                  return_tensors="pt").input_ids[0].to(dev)
        inp = pid.unsqueeze(0).expand(K, -1).contiguous()
        with torch.no_grad():
            gen = pretrained.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True,
                                      top_k=0, top_p=1.0, temperature=1.0, max_new_tokens=30,
                                      min_new_tokens=10, pad_token_id=tok.eos_token_id)
        P = pid.shape[0]
        for k in range(K):
            y = gen[k, P:].cpu()
            if tok.eos_token_id in y.tolist():
                y = y[: y.tolist().index(tok.eos_token_id)]
            if y.numel() < 1:
                continue
            text = tok.decode(torch.cat([pid.cpu(), y]).tolist())
            A = _roberta_nothate_logit(rob, rob_tok, text, dev) - _value_last(model_vh, pid.cpu(), y, dev)
            r = Rollout(prompt_ids=pid.cpu(), response_ids=y, reward=0.0, step=0)
            s = per_sample_grads_multi(models.policy, weights, r, cfg, device=dev)
            for nm in weights:
                g[nm] += A * s[nm].double()
            n += 1
        if (pi + 1) % 25 == 0:
            print(f"    [g_seq] {pi+1}/{len(prompts)} prompts, {n} pairs")
    return {nm: g[nm] / n for nm in weights}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--n-seq-prompts", type=int, default=100)
    ap.add_argument("--n-lee", type=int, default=100)
    ap.add_argument("--max-rollouts", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda"
    cfg = EKFACConfig(layer_name=L9)
    FACT.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    print(f"C3/E6 all-MLP vs layer-9. torch {torch.__version__}")

    models = load_models(dev, policy_requires_grad=False, policy_attn_implementation="eager")
    names = gptneo_mlp_layer_names(12)
    layers = mlp_layer_dict(models.policy, names)
    for p in models.policy.parameters():
        p.requires_grad_(False)
    weights = {}
    for nm, lyr in layers.items():
        lyr.weight.requires_grad_(True)
        weights[nm] = lyr.weight
    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    all_prompts = ep["prompt_texts"]

    # ---- multi-layer factors (on-policy, fixed) ----
    if (FACT / "QA_0.pt").exists():
        print("  loading saved multi-layer factors ...")
        QLam = torch.load(FACT / "QLam.pt", map_location=dev)
    else:
        print("  on-policy rollouts + multi-layer EK-FAC ...")
        rolls, ntok = generate_factor_rollouts(models.policy, models.tokenizer, all_prompts,
                                               k_per_prompt=24, token_budget=args.budget,
                                               device=dev, seed_base=SEED)
        print(f"    {len(rolls)} rollouts, {ntok} tok ({time.time()-t0:.0f}s)")
        AS = accumulate_AS_multi(models.policy, layers, rolls, cfg, device=dev)
        Qs = {nm: (eigendecompose(A, S)[0], eigendecompose(A, S)[2]) for nm, (A, S, _) in AS.items()}
        Lam = fit_lambda_multi(models.policy, layers, rolls, cfg, Qs=Qs, device=dev)
        QLam = {nm: (Qs[nm][0].double(), Qs[nm][1].double(), Lam[nm][0].double()) for nm in names}
        torch.save(QLam, FACT / "QLam.pt")
        torch.save(torch.tensor([1]), FACT / "QA_0.pt")  # marker
        print(f"    factors built ({time.time()-t0:.0f}s)")

    # ---- per-layer directions ----
    print("\n  eval directions (per layer) ...")
    from trl import AutoModelForCausalLMWithValueHead
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_vh = AutoModelForCausalLMWithValueHead.from_pretrained("data/ppo_checkpoints/step_0650").to(dev).eval()
    rob_tok = AutoTokenizer.from_pretrained(REWARD_MODEL)
    rob = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).to(dev).eval()
    g_seq = compute_g_seq_multi(models, model_vh, rob, rob_tok, all_prompts[:args.n_seq_prompts],
                                layers, weights, cfg, dev, K=8, seed=SEED)
    del model_vh, rob, rob_tok
    torch.cuda.empty_cache()
    gf = build_g_f_multi(models.policy, weights, models.tokenizer, load_lee_pairs(args.n_lee), cfg, dev)
    g_dir = {"seq": g_seq, "toxic_subtracted": gf.g_subtracted}
    p_f = {t: {nm: inverse_hvp(g_dir[t][nm].to(dev), QLam[nm][0], QLam[nm][1], QLam[nm][2],
                               damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha)
               for nm in names} for t in g_dir}
    print(f"    directions + p_f built ({time.time()-t0:.0f}s)")

    # ---- stream baseline influence: all-MLP sum + layer-9 ----
    print("\n  scoring all rollouts (all-MLP + layer-9) ...")
    rolls, meta = [], []
    for step in range(0, 651):
        raw = torch.load(ROLL / f"step_{step:04d}.pt", map_location="cpu", weights_only=False)
        for r in raw:
            rolls.append(Rollout(prompt_ids=r["prompt_token_ids"].long(),
                                 response_ids=r["response_token_ids"].long(),
                                 reward=float(r["reward"]), step=int(r["step"])))
        if args.max_rollouts and len(rolls) >= args.max_rollouts:
            break
    N = len(rolls)
    tgts = list(g_dir)
    I_all = {t: torch.zeros(N, dtype=torch.float64) for t in tgts}
    I_l9 = {t: torch.zeros(N, dtype=torch.float64) for t in tgts}
    for m, r in enumerate(rolls):
        s_m = per_sample_grads_multi(models.policy, weights, r, cfg, device=dev)
        s64 = {nm: s_m[nm].double() for nm in names}
        for t in tgts:
            acc = 0.0
            for nm in names:
                acc += -float((p_f[t][nm] * s64[nm]).sum().item())
            I_all[t][m] = acc
            I_l9[t][m] = -float((p_f[t][L9] * s64[L9]).sum().item())
        if (m + 1) % 4000 == 0:
            print(f"    scored {m+1}/{N} ({time.time()-t0:.0f}s)")
    torch.save({"I_all": I_all, "I_l9": I_l9}, OUT / "influence_allmlp_vs_l9.pt")

    # ---- E6 comparison ----
    from scipy.stats import spearmanr
    def jac(a, b, k, by_abs=True):
        fa, fb = (np.abs(a), np.abs(b)) if by_abs else (a, b)
        return len(set(np.argsort(-fa)[:k]) & set(np.argsort(-fb)[:k])) / len(set(np.argsort(-fa)[:k]) | set(np.argsort(-fb)[:k]))
    k10 = N // 10
    res = {"n_rollouts": N, "n_layers": len(names), "targets": {}}
    for t in tgts:
        a = np.asarray(I_all[t]); b = np.asarray(I_l9[t])
        res["targets"][t] = {
            "spearman_allmlp_vs_l9": float(spearmanr(a, b).statistic),
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "jaccard_top10pct_abs": jac(a, b, k10),
            "jaccard_top100_abs": jac(a, b, 100),
        }
        print(f"  [{t}] Spearman(allMLP,L9)={res['targets'][t]['spearman_allmlp_vs_l9']:.3f} "
              f"J@top10%={res['targets'][t]['jaccard_top10pct_abs']:.2f} "
              f"J@top100={res['targets'][t]['jaccard_top100_abs']:.2f}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {REPORT} (total {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
