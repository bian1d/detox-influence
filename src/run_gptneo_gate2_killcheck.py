"""Block C, gate 2 — does the KILL survive the Fisher fix, on-policy factors?

C1: rebuild layer-9 EK-FAC factors with the FIXED (row-aligned) code, on FRESH
    on-policy rollouts at theta* (the metric-consistent choice the user picked:
    F and Delta now share the same y ~ pi_theta* measure).
C2: recompute the thermometer ||Delta~ p_f|| / ||g_f|| for f_seq and the robust
    baseline-subtracted toxicity direction, plus the per-token-vs-per-sample
    metric ratio c, and report BOTH the raw thermometer and the metric-honest
    estimate (~ c x raw). Compare to the old buggy-era 34/12/4/12 and confirm
    the kill (>> 0.3) survives.

This reuses the Phase-4-validated Delta machinery (delta_vp_per_prompt double
backward, U-stat ||.||^2, two-half cosine) UNCHANGED — only the factors (fixed,
on-policy) and the eval directions (robust) differ. NO disk cache.

Run (base env, GPU):
  python3 src/run_gptneo_gate2_killcheck.py [--budget N] [--n-prompts N] [--k N]
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda, inverse_hvp  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.hooks import capture_c_proj  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402
from phase4.config import BETA, EVAL_PROMPTS_PATH, SEED  # noqa: E402
from phase4.e3_delta import DeltaSample, delta_vp_per_prompt_tok  # noqa: E402
from phase4.models import load_models  # noqa: E402
from phase4.sampling import sample_on_policy_pool  # noqa: E402
from phase5.sample_rollouts import generate_factor_rollouts  # noqa: E402
from phase5.toxicity_direction import build_g_f, load_lee_pairs  # noqa: E402

LAYER = "transformer.h.9.mlp.c_proj"
OUT = Path("data/ekfac/influence/run_step0650_refix")
REPORT = Path("reports/notes/gate2_killcheck_tok.json")
OLD_THERMO = {"seq": 34.0, "toxic_C1": 12.0, "toxic_C2": 4.0, "toxic_C3": 12.0}
REWARD_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"


# ---------------------------------------------------------------------------
# f_seq direction (Hu et al. 2025): g_seq = mean over (x,y~theta*) of A * s(x,y)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _roberta_nothate_logit(rob, rob_tok, text, device):
    inp = rob_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    return float(rob(**inp).logits[0, 0].item())


@torch.no_grad()
def _value_last(model_vh, prompt_ids, y_ids, device):
    full = torch.cat([prompt_ids, y_ids]).unsqueeze(0).to(device)
    out = model_vh(full)
    values = out[2] if isinstance(out, tuple) and len(out) >= 3 else getattr(out, "value", None)
    return float(values[0, -1].item())


def compute_g_seq(model, model_vh, tokenizer, rob, rob_tok, prompts, layer, cfg, device,
                  *, K, seed):
    torch.manual_seed(seed)
    g = torch.zeros_like(layer.weight, dtype=torch.float64)
    n = 0
    advs = []
    pretrained = model_vh.pretrained_model
    pretrained.eval()
    for pi, ptext in enumerate(prompts):
        pid = tokenizer(ptext, truncation=True, max_length=64, add_special_tokens=False,
                        return_tensors="pt").input_ids[0].to(device)
        inp = pid.unsqueeze(0).expand(K, -1).contiguous()
        attn = torch.ones_like(inp)
        with torch.no_grad():
            gen = pretrained.generate(inp, attention_mask=attn, do_sample=True, top_k=0,
                                      top_p=1.0, temperature=1.0, max_new_tokens=30,
                                      min_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        P = pid.shape[0]
        for k in range(K):
            y = gen[k, P:].cpu()
            if tokenizer.eos_token_id in y.tolist():
                y = y[: y.tolist().index(tokenizer.eos_token_id)]
            if y.numel() < 1:
                continue
            text = tokenizer.decode(torch.cat([pid.cpu(), y]).tolist())
            reward = _roberta_nothate_logit(rob, rob_tok, text, device)
            v = _value_last(model_vh, pid.cpu(), y, device)
            A = reward - v
            r_obj = Rollout(prompt_ids=pid.cpu(), response_ids=y, reward=reward, step=0)
            s = compute_per_sample_grad(model, layer, r_obj, cfg, device=device).double()
            g.add_(A * s)
            advs.append(A)
            n += 1
        if (pi + 1) % 20 == 0:
            print(f"    [g_seq] {pi+1}/{len(prompts)} prompts, {n} pairs")
    return g / n, {"n_pairs": n, "adv_mean": float(np.mean(advs)),
                   "adv_min": float(np.min(advs)), "adv_max": float(np.max(advs))}


# ---------------------------------------------------------------------------
# per-token vs per-sample Fisher metric ratio c (audit-5 method) for a direction
# ---------------------------------------------------------------------------
def metric_ratio(model, layer, pool_rolls, v, device):
    """c = q_tok(v)/q_samp(v): how much larger the per-token EK-FAC Fisher is
    than the per-sample-score Fisher in direction v. v is (d_out,d_in) fp64."""
    V = v.to(device).double()
    sum_tok = 0.0
    sum_samp = 0.0
    n_tok = 0
    n_samp = 0
    model.eval()
    for r in pool_rolls:
        P, R = r.prompt_len, r.response_len
        T = P + R
        ids = torch.cat([r.prompt_ids, r.response_ids]).to(device).unsqueeze(0)
        labels = torch.full((T,), -100, dtype=torch.long, device=device)
        labels[P:] = r.response_ids.to(device)
        for p_ in model.parameters():
            if p_.grad is not None:
                p_.grad = None
        with capture_c_proj(layer) as cache:
            logits = model(ids).logits
            loss = F.cross_entropy(logits[0, :-1].to(torch.promote_types(logits.dtype, torch.float32)),
                                   labels[1:], ignore_index=-100, reduction="sum")
            loss.backward()
        m_resp = cache.m[0, P - 1:T - 1].double()
        d_resp = cache.delta[0, P - 1:T - 1].double()
        gtv = (d_resp * (m_resp @ V.T)).sum(dim=1)   # g_t . v per token
        sum_tok += float((gtv ** 2).sum().item())
        sum_samp += (float(gtv.sum().item()) / R) ** 2
        n_tok += R
        n_samp += 1
    q_tok = sum_tok / n_tok
    q_samp = sum_samp / n_samp
    return q_tok / q_samp if q_samp > 0 else float("nan")


def _streaming_norm_sq(d_p: torch.Tensor) -> dict:
    """U-stat ||mean d_p||^2 + two-half cosine, streaming (no (N,D) matrix)."""
    N = d_p.shape[0]
    half = N // 2
    a = torch.zeros(d_p.shape[1:], dtype=torch.float64)
    b = torch.zeros_like(a)
    for p in range(N):
        (a if p < half else b).add_(d_p[p].to(torch.float64))
    S = a + b
    cos = float(((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item())
    diag = 0.0
    for p in range(N):
        gp = d_p[p].to(torch.float64)
        diag += float((gp * gp).sum().item())
    cross = float((S * S).sum().item())
    U = (cross - diag) / (N * (N - 1))
    return {"point": U, "two_half_cosine": cos, "naive": cross / (N * N)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200_000, help="on-policy factor token budget")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-seq-prompts", type=int, default=100)
    ap.add_argument("--n-lee", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EKFACConfig(layer_name=LAYER)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"GATE 2 kill-check (fixed code, on-policy factors). torch {torch.__version__}, dev {dev}")

    # eager attn for double-backward; policy grads on layer-9 only.
    models = load_models(dev, policy_requires_grad=False, policy_attn_implementation="eager")
    layer = models.policy.get_submodule(LAYER)
    layer.weight.requires_grad_(True)
    weight = layer.weight

    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    all_prompts = ep["prompt_texts"]

    # ---- C1: on-policy factors, fixed code ----
    print("\nC1: on-policy factor rollouts + fixed EK-FAC ...")
    rolls, n_tok = generate_factor_rollouts(models.policy, models.tokenizer, all_prompts,
                                            k_per_prompt=24, token_budget=args.budget,
                                            device=dev, seed_base=SEED)
    print(f"  {len(rolls)} on-policy rollouts, {n_tok} tokens ({time.time()-t0:.0f}s)")
    A, S, n_as = accumulate_AS(models.policy, layer, rolls, cfg, device=dev)
    Q_A, _, Q_S, _ = eigendecompose(A, S)
    Lambda, _ = fit_lambda(models.policy, layer, rolls, cfg, Q_A=Q_A, Q_S=Q_S, device=dev)
    torch.save(S.cpu(), OUT / "S.pt")
    torch.save(Q_A.cpu(), OUT / "Q_A.pt")
    torch.save(Q_S.cpu(), OUT / "Q_S.pt")
    torch.save(Lambda.cpu(), OUT / "Lambda.pt")
    print(f"  factors built ({time.time()-t0:.0f}s); Lambda range "
          f"[{float(Lambda.min()):.2e},{float(Lambda.max()):.2e}]")
    Q_A_d, Q_S_d, Lam_d = Q_A.double(), Q_S.double(), Lambda.double()

    # ---- eval directions ----
    print("\nEval directions: f_seq (value head) + robust toxicity (Lee pairs) ...")
    from trl import AutoModelForCausalLMWithValueHead
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_vh = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(Path("data/ppo_checkpoints/step_0650"))).to(dev).eval()
    rob_tok = AutoTokenizer.from_pretrained(REWARD_MODEL)
    rob = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).to(dev).eval()
    seq_prompts = all_prompts[: args.n_seq_prompts]
    g_seq, seq_stats = compute_g_seq(models.policy, model_vh, models.tokenizer, rob, rob_tok,
                                     seq_prompts, layer, cfg, dev, K=8, seed=SEED)
    del model_vh, rob, rob_tok
    torch.cuda.empty_cache()
    pairs = load_lee_pairs(args.n_lee)
    gf = build_g_f(models.policy, weight, models.tokenizer, pairs, dev)
    print(f"  g_f robust: ||subtracted||={gf.norms['norm_subtracted']:.3e} "
          f"||toxic||={gf.norms['norm_mean_toxic']:.3e} cos={gf.norms['cos_tox_nontox']:.3f}")

    targets = {
        "seq": g_seq.double(),
        "toxic_subtracted": gf.g_f_subtracted.double().to(dev),
        "toxic_only": gf.g_f_toxic_only.double().to(dev),
    }
    p_f = {t: inverse_hvp(g, Q_A_d, Q_S_d, Lam_d,
                          damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha)
           for t, g in targets.items()}
    g_norm = {t: float(g.norm().item()) for t, g in targets.items()}

    # ---- on-policy pool for Delta ----
    print(f"\nC2: on-policy pool {args.n_prompts}x{args.k} for Delta ...")
    rng = np.random.RandomState(SEED)
    pick = sorted(rng.choice(len(all_prompts), size=args.n_prompts, replace=False).tolist())
    prompts = [all_prompts[i] for i in pick]
    pool = sample_on_policy_pool(prompts, args.k, seed_base=SEED, models=models, beta=BETA, device=dev)
    by_p = defaultdict(list)
    for s in pool:
        by_p[s.prompt_idx].append(s)
    samples = []
    for sl in by_p.values():
        mR = float(np.mean([s.R_tilde for s in sl]))
        for s in sl:
            samples.append(DeltaSample(prompt_idx=s.prompt_idx, prompt_ids=s.prompt_ids,
                                       response_ids=s.response_ids, A=float(s.R_tilde - mR)))
    print(f"  pool {len(pool)} samples ({time.time()-t0:.0f}s)")

    # ---- Delta_tok~ p_f (PER-TOKEN granularity, matches F_tok), thermometer ----
    # Granularity-consistent: Delta is now the per-token operator (diagonal score
    # term + sum-HVP, /total-tokens) matching the per-token EK-FAC Fisher. The
    # thermometer ||Delta_tok~ p|| / ||g_f|| is a same-granularity ratio — NO c
    # correction (see reports/metric_granularity_derivation.md).
    print("\nComputing Delta_tok~ p_f (per-token) + thermometer ...")
    vs = [p_f[t] for t in targets]
    d_p, order, N_tok_delta, _ = delta_vp_per_prompt_tok(
        samples, models.policy, layer, BETA, vs, dev, store_device="cpu")
    results = {}
    for i, t in enumerate(targets):
        st = _streaming_norm_sq(d_p[i])
        dp_norm = float(np.sqrt(max(st["point"], 0.0)))
        therm = dp_norm / g_norm[t]
        results[t] = {
            "thermometer": therm,
            "delta_p_norm": dp_norm,
            "g_f_norm": g_norm[t],
            "two_half_cosine": st["two_half_cosine"],
            "old_thermometer_persample": OLD_THERMO.get(t if t == "seq" else "toxic_C1"),
        }
        print(f"  [{t:17s}] thermometer(per-token) = {therm:8.4f}   "
              f"two-half cos={st['two_half_cosine']:.3f}")

    max_t = max(r["thermometer"] for r in results.values())
    verdict = ("KILL (>0.3)" if max_t > 0.3 else
               "TRANSITION (<0.1)" if max_t < 0.1 else "MIDBAND (0.1-0.3)")
    out = {
        "n_factor_rollouts": len(rolls), "n_tok_factors": int(n_as),
        "n_tok_delta_pool": int(N_tok_delta),
        "beta": BETA, "seq_stats": seq_stats, "gf_robust_norms": gf.norms,
        "n_pool": len(pool), "n_prompts": args.n_prompts, "k": args.k,
        "targets": results, "max_thermometer": max_t, "verdict": verdict,
        "note": ("PER-TOKEN-granularity Delta (diagonal score term + sum-HVP, /total-tokens) "
                 "matching the per-token EK-FAC Fisher F_tok. Thermometer is a same-granularity "
                 "ratio, NO c correction. Compare to the OLD per-sample-mean Delta + c-estimate "
                 "(gate2_killcheck.json): seq/toxic raw were 70.7/8.5/42.1 with c~13-22."),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\n  VERDICT: {verdict} (max per-token thermometer = {max_t:.3f})")
    print(f"  wrote {REPORT}  (total {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
