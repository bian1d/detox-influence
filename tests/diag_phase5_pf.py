"""Diagnose the Phase 5 thermometer blow-up: is p_f = F^-1 g_f exploding because
OLMo's down_proj Fisher is near-singular (tiny eigenvalues / weak damping)?

Reports, for both g_f versions:
  - ||g_f||, ||p_f||, amplification ||p_f||/||g_f||
  - Lambda spectrum (min/mean/max, fraction below damping floor and below mean)
  - the two-level denom min/max and what the IHVP cap (1/denom_min) is
  - share of ||p_f||^2 that lives in the smallest-denominator 1% of directions
and a few per-sample Delta contribution norms (A*(s(s^Tp)+HVP(p))) so we see
whether Delta~ p is genuinely huge or the smoke's U-stat was just 4-prompt noise.

    /root/miniconda3/envs/olmo/bin/python tests/diag_phase5_pf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.eigen import inverse_hvp  # noqa: E402
from phase4.grad_phi import get_phi_weight  # noqa: E402
from phase4.hvp_logpi import score_and_hvps  # noqa: E402
from phase5.config import BETA_NOMINAL, PHI_LAYER, OUT_DIR  # noqa: E402
from phase5.models_olmo import load_olmo_models  # noqa: E402
from phase5.toxicity_direction import build_g_f, load_lee_pairs  # noqa: E402

FACTOR_DIR = OUT_DIR / "factors"


def main() -> int:
    device = "cuda"
    cfg = EKFACConfig(layer_name=PHI_LAYER)
    Q_A = torch.load(FACTOR_DIR / "Q_A.pt", map_location=device).to(torch.float64)
    Q_S = torch.load(FACTOR_DIR / "Q_S.pt", map_location=device).to(torch.float64)
    Lambda = torch.load(FACTOR_DIR / "Lambda.pt", map_location=device).to(torch.float64)

    lam = Lambda.flatten()
    print(f"Lambda: shape {tuple(Lambda.shape)} min {lam.min():.3e} mean {lam.mean():.3e} "
          f"max {lam.max():.3e}")
    print(f"  frac < floor(1e-5): {(lam < cfg.damping_floor).float().mean():.4f}  "
          f"frac < mean: {(lam < lam.mean()).float().mean():.4f}")
    denom = (Lambda + cfg.damping_alpha * Lambda.mean()).clamp(min=cfg.damping_floor)
    print(f"  two-level denom: min {denom.min():.3e} max {denom.max():.3e}  "
          f"=> IHVP gain cap 1/denom_min = {1.0/float(denom.min()):.3e}")

    print("\nloading OLMo (eager) + building g_f (20 Lee pairs) ...")
    models = load_olmo_models(device, policy_requires_grad=False, policy_attn_implementation="eager")
    weight = get_phi_weight(models.policy, PHI_LAYER)
    weight.requires_grad_(True)
    pairs = load_lee_pairs(20)
    gf = build_g_f(models.policy, weight, models.tokenizer, pairs, device)

    for name, g in [("subtracted", gf.g_f_subtracted.to(device)),
                    ("toxic_only", gf.g_f_toxic_only.to(device))]:
        p = inverse_hvp(g, Q_A, Q_S, Lambda,
                        damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha)
        gn, pn = float(g.norm()), float(p.norm())
        # share of ||p||^2 in the smallest-denom 1% of directions
        g_eig = Q_S.T @ g @ Q_A           # (d_out, d_in) in eigenbasis
        p_eig = g_eig / denom.T            # what inverse_hvp does, eigenbasis layout
        p2 = (p_eig ** 2)
        thr = torch.quantile(denom.flatten(), 0.01)
        small = (denom.T <= thr)
        share_small = float(p2[small].sum() / p2.sum())
        print(f"\n[{name}] ||g_f||={gn:.4e} ||p_f||={pn:.4e} amplification={pn/gn:.3e}")
        print(f"   share of ||p_f||^2 in smallest-denom 1% dirs: {share_small:.3f}")

    # a few per-sample Delta contributions to gauge ||Delta~ p|| scale directly
    print("\nper-sample Delta contributions A*(s(s^Tp)+HVP(p))/beta for p=p_subtracted:")
    p_sub = inverse_hvp(gf.g_f_subtracted.to(device), Q_A, Q_S, Lambda,
                        damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha)
    for i, rec in enumerate(pairs[:4]):
        p_ids = models.tokenizer(rec["prompt_text"], return_tensors="pt")["input_ids"][0][:32]
        r_ids = models.tokenizer(rec["pert_gen_text"], return_tensors="pt",
                                 add_special_tokens=False)["input_ids"][0][:48]
        s, (hvp,) = score_and_hvps(models.policy, weight, p_ids, r_ids, [p_sub], device)
        s64 = s.to(torch.float64)
        stp = float((s64 * p_sub).sum())
        contrib = (s64 * stp + hvp.to(torch.float64)) / BETA_NOMINAL  # A=1 here
        print(f"   sample {i}: |s|={float(s64.norm()):.3e} s^Tp={stp:.3e} "
              f"|HVP(p)|={float(hvp.norm()):.3e} ||contrib(A=1)||={float(contrib.norm()):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
