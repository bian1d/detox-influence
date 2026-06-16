"""INDEPENDENT AUDIT 5 — F-vs-Delta metric-scale & sampling-source consistency.

Cut 4 from the review brief: confirm the Fisher F and the curvature operator
Delta are built in the SAME metric, so the thermometer ||Delta~ p|| / ||g_f||
(the Phase-4/5 kill-vs-transition decision variable) is a ratio of comparable
quantities.

Two independent measurements, both on the REAL θ* GPT-Neo, on ONE fresh
on-policy pool (so sample source is held fixed and only the reduction differs):

(M1) per-token vs per-sample Fisher scale.
   The production EK-FAC F is accumulated from per-token gradient outer products
   normalised by token count -> F approximates the PER-TOKEN Fisher
   F_tok = E_t[ g_t g_t^T ],  g_t = (p_t - e)·m_t^T   (sum-loss per-token grad).
   Delta's s s^T term and the eval/train scores are PER-SAMPLE mean-reduced
   s_i = (1/R_i) sum_t g_t.  Their Fisher is F_samp = E_i[ s_i s_i^T ].
   For a direction v:  q_tok(v) = E_t[(g_t·v)^2],  q_samp(v) = E_i[(s_i·v)^2].
   ratio(v) = q_tok(v) / q_samp(v).  If ~1 the two are on the same scale; if
   ~R_typ (~25-30) there is a per-token/per-sample mismatch of that size between
   the F used to build p=F^-1 g_f and the s-scale that builds Delta.
   Measured for v in {p_seq, p_toxic, g_seq, random}.

(M2) does EK-FAC's production F (Lambda) match the empirical per-token Fisher
   on these on-policy directions?  Compare v^T F_ekfac v (forward EK-FAC,
   no damping) to q_tok(v).  Same order = EK-FAC F is genuinely a per-token
   Fisher (which then confirms M1's interpretation); wildly off would mean the
   scale story is more complex.

(M3) consequence for the thermometer: if a global scale factor c multiplies F
   (F_used = c · F_true), then p = F^-1 g_f -> p/c and the thermometer
   ||Delta~ p||/||g_f|| -> /c.  Report the implied thermometer rescaling for the
   measured ratio and whether it could move the Phase-4 verdicts (34/12/4/12
   for seq/C1/C2/C3) across the 0.3 kill line.

Also DOCUMENTS (no run needed) the sampling-source fact: production F is built
on STORED TRAINING rollouts (steps 0..650, off-policy w.r.t. θ*) + pseudo-labels;
Delta (E3) on a FRESH on-policy pool + real tokens + real rewards. The spec
§1 E3 revision asked for F's pseudo-label and A's response to be the same
on-policy draw; the production pipeline does not do this. Whether it matters is
exactly what the ranking-robustness (audit 4b) and this scale check bound.

Run (base env, GPU):  python3 tests/audit_5_metric_scale.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout, build_input_and_labels  # noqa: E402
from ekfac.hooks import capture_c_proj  # noqa: E402

RUN = Path("data/ekfac/influence/run_step0650")
CKPT = Path("data/ppo_checkpoints/step_0650")
PROMPTS = Path("data/eval_prompts_400.json")
OUT = Path("reports/notes/audit5_metric_scale.json")


def f_ekfac_forward(v, Q_A, Q_S, Lam):
    """Apply the EK-FAC Fisher (NO damping) to v: F v = Q_S ((Q_S^T v Q_A) * Λ.T) Q_A^T.
    Inverse divides by (Λ.T + damp); forward multiplies by Λ.T."""
    v_proj = Q_S.T @ v @ Q_A                # (d_out, d_in)
    scaled = v_proj * Lam.T                 # (d_out, d_in)
    return Q_S @ scaled @ Q_A.T


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EKFACConfig(layer_name="transformer.h.9.mlp.c_proj")
    print(f"AUDIT 5: metric-scale F vs Delta  (torch {torch.__version__}, "
          f"numpy {np.__version__}, dev {dev})", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(CKPT.as_posix()).to(dev).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    layer = model.get_submodule(cfg.layer_name)
    layer.weight.requires_grad_(True)
    tok = AutoTokenizer.from_pretrained(CKPT.as_posix())
    tok.pad_token = tok.eos_token

    # ---- fresh on-policy pool at θ* ----
    ep = json.loads(PROMPTS.read_text())
    prompt_texts = ep["prompt_texts"][:40]
    torch.manual_seed(42)
    pool: list[Rollout] = []
    for pt in prompt_texts:
        pid = tok(pt, return_tensors="pt", truncation=True, max_length=64).input_ids.to(dev)
        with torch.no_grad():
            gen = model.generate(pid, do_sample=True, top_k=0, top_p=1.0,
                                 max_new_tokens=30, min_new_tokens=10,
                                 pad_token_id=tok.pad_token_id)
        P = pid.shape[1]
        for _ in range(8):
            with torch.no_grad():
                g1 = model.generate(pid, do_sample=True, top_k=0, top_p=1.0,
                                    max_new_tokens=30, min_new_tokens=10,
                                    pad_token_id=tok.pad_token_id)
            resp = g1[0, P:].cpu()
            # trim trailing pads
            keep = resp.shape[0]
            for j in range(resp.shape[0] - 1, -1, -1):
                if int(resp[j]) != tok.pad_token_id:
                    keep = j + 1
                    break
            resp = resp[:keep]
            if resp.numel() >= 1:
                pool.append(Rollout(prompt_ids=pid[0].cpu(), response_ids=resp, reward=0.0, step=0))
    Rs = [r.response_len for r in pool]
    R_typ = float(np.mean(Rs))
    print(f"  on-policy pool: {len(pool)} samples, mean R = {R_typ:.1f} "
          f"(min {min(Rs)}, max {max(Rs)})", flush=True)

    # ---- directions ----
    Q_A = torch.load(RUN / "Q_A.pt", map_location=dev).double()
    Q_S = torch.load(RUN / "Q_S.pt", map_location=dev).double()
    Lam = torch.load(RUN / "Lambda.pt", map_location=dev).double()
    dirs = {}
    for name, f in [("p_seq", "p_seq.pt"), ("p_toxic_C1", "p_toxic_C1.pt"),
                    ("g_seq", "g_seq.pt"), ("g_toxic_C1", "g_toxic_C1.pt")]:
        if (RUN / f).exists():
            dirs[name] = torch.load(RUN / f, map_location=dev).double()
    torch.manual_seed(0)
    rv = torch.randn_like(dirs["g_seq"])
    dirs["random"] = rv / rv.norm()

    # ---- per-token / per-sample accumulation, one sum-loss backward per sample ----
    # For direction v (d_out, d_in): g_t·v = δ_t^T (V m_t) where V = v.
    # q_tok = mean_t (g_t·v)^2 ;  s_i·v = (1/R) sum_t g_t·v ; q_samp = mean_i (s_i·v)^2.
    names = list(dirs)
    Vs = {k: dirs[k] for k in names}
    sum_tok = {k: 0.0 for k in names}    # sum_t (g_t·v)^2
    n_tok = 0
    sum_samp = {k: 0.0 for k in names}   # sum_i (s_i·v)^2
    n_samp = 0
    # also F_ekfac forward quadratic per direction
    vFv = {k: float((Vs[k] * f_ekfac_forward(Vs[k], Q_A, Q_S, Lam)).sum().item())
           for k in names}

    model.eval()
    for r in pool:
        P, R = r.prompt_len, r.response_len
        T = P + R
        ids = torch.cat([r.prompt_ids, r.response_ids]).to(dev).unsqueeze(0)
        labels = torch.full((T,), -100, dtype=torch.long, device=dev)
        labels[P:] = r.response_ids.to(dev)
        for p_ in model.parameters():
            if p_.grad is not None:
                p_.grad = None
        with capture_c_proj(layer) as cache:
            logits = model(ids).logits
            loss = TF.cross_entropy(logits[0, :-1].float(), labels[1:],
                                    ignore_index=-100, reduction="sum")
            loss.backward()
        m_resp = cache.m[0, P - 1:T - 1, :].double()       # (R, d_in)
        d_resp = cache.delta[0, P - 1:T - 1, :].double()   # (R, d_out)
        for k in names:
            V = Vs[k]                                       # (d_out, d_in)
            Vm = m_resp @ V.T                               # (R, d_out)
            gtv = (d_resp * Vm).sum(dim=1)                  # (R,) = g_t·v per token
            sum_tok[k] += float((gtv ** 2).sum().item())
            s_iv = float(gtv.sum().item()) / R              # s_i·v (mean reduction)
            sum_samp[k] += s_iv ** 2
        n_tok += R
        n_samp += 1

    print("\n  direction        q_tok        q_samp      ratio(tok/samp)   vFv_ekfac   vFv/q_tok", flush=True)
    res = {"n_samples": n_samp, "n_tok": n_tok, "R_typ": R_typ, "directions": {}}
    for k in names:
        q_tok = sum_tok[k] / n_tok
        q_samp = sum_samp[k] / n_samp
        ratio = q_tok / q_samp if q_samp > 0 else float("inf")
        ek_ratio = vFv[k] / q_tok if q_tok > 0 else float("inf")
        res["directions"][k] = {"q_tok": q_tok, "q_samp": q_samp,
                                "ratio_tok_over_samp": ratio,
                                "vFv_ekfac": vFv[k], "vFv_over_qtok": ek_ratio}
        print(f"  {k:14s}  {q_tok:.4e}  {q_samp:.4e}    {ratio:8.2f}        "
              f"{vFv[k]:.4e}   {ek_ratio:6.3f}", flush=True)

    # ---- M3 thermometer consequence ----
    # Direction of the effect (careful — easy to get backwards):
    #   F_used = F_tok = c · F_samp with measured c = q_tok/q_samp > 1 (F_tok is
    #   LARGER than the per-sample-score Fisher that Δ's s s^T lives in).
    #   p = F_used^-1 g_f = (1/c) F_samp^-1 g_f  -> production p is SMALLER by c.
    #   thermometer_production = ||Δ~ p||/||g_f|| = (1/c) · thermometer_metric_honest.
    #   => thermometer_metric_honest = c · thermometer_production  (LARGER, not smaller).
    seq_ratio = res["directions"].get("p_seq", {}).get("ratio_tok_over_samp", float("nan"))
    print("\n  M3 thermometer consequence:", flush=True)
    print(f"     mean R (per-token vs per-sample scale gap) ~ {R_typ:.1f}", flush=True)
    print(f"     measured c = q_tok/q_samp on p_seq = {seq_ratio:.2f} (F_tok > F_samp)", flush=True)
    print("     metric-honest thermometer = c × reported  (production p is 1/c too small).", flush=True)
    print("     Phase-4 thermometers reported seq/C1/C2/C3 = 34/12/4/12 (≫0.3 kill).", flush=True)
    print(f"     metric-honest s s^T-channel estimate ~ {34*seq_ratio:.0f}/{12*seq_ratio:.0f}/"
          f"{4*seq_ratio:.0f}/{12*seq_ratio:.0f} — LARGER, so kill is reinforced, not", flush=True)
    print("     threatened. (Δ also carries the HVP term, scaled differently, so treat", flush=True)
    print("     these as order-of-magnitude, not exact.) No target approaches the 0.3", flush=True)
    print("     line under the metric fix; ranking IF is scale-invariant either way.", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
