"""Phase 5 Stage 1 — EK-FAC factors on OLMo-2-1B-SFT's layer-12 down_proj.

Builds A, S, Q_A, Q_S, Lambda over ~600k response tokens of OLMo-SFT on-policy
rollouts (matched to GPT-Neo's Fisher token budget). These give F^-1 for the
thermometer's p_f = F^-1 g_f. Reuses the Phase-4-validated ekfac pipeline
(accumulate_AS -> eigendecompose -> fit_lambda) unchanged; only the model,
layer name, and rollout source differ.

    /root/miniconda3/envs/olmo/bin/python src/run_phase5_stage1_factors.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from phase5.config import (  # noqa: E402
    EVAL_PROMPTS_PATH, FACTOR_TOKEN_BUDGET, OUT_DIR, PHI_LAYER, SEED, THETA_STAR_ID,
)
from phase5.sample_rollouts import generate_factor_rollouts  # noqa: E402

FACTOR_DIR = OUT_DIR / "factors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small budget, ~5k tokens")
    args = ap.parse_args()
    device = "cuda"
    budget = 5_000 if args.smoke else FACTOR_TOKEN_BUDGET
    k_per_prompt = 8 if args.smoke else 52
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading OLMo-SFT (frozen except {PHI_LAYER}.weight) ...")
    tok = AutoTokenizer.from_pretrained(THETA_STAR_ID)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(THETA_STAR_ID, torch_dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer = model.get_submodule(PHI_LAYER)
    layer.weight.requires_grad_(True)              # the one leaf backward needs
    cfg = EKFACConfig(layer_name=PHI_LAYER)

    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    prompts = ep["prompt_texts"]                   # full RTP tox>0.3 set (~400)
    print(f"generating on-policy rollouts (budget {budget} tokens, k={k_per_prompt}/prompt) ...")
    t0 = time.time()
    rollouts, n_tok = generate_factor_rollouts(model, tok, prompts, k_per_prompt, budget, device, seed_base=SEED)
    t_gen = time.time() - t0
    print(f"  {len(rollouts)} rollouts, {n_tok} response tokens in {t_gen:.0f}s")

    print("Stage 1A: accumulate A, S ...")
    t0 = time.time()
    A, S, n_as = accumulate_AS(model, layer, rollouts, cfg, device=device)
    t_as = time.time() - t0
    print(f"  A {tuple(A.shape)} S {tuple(S.shape)} n_tok={n_as} in {t_as:.0f}s")

    print("Stage 1B: eigendecompose + fit_lambda ...")
    t0 = time.time()
    Q_A, Lambda_A, Q_S, Lambda_S = eigendecompose(A, S)
    Lambda, n_lam = fit_lambda(model, layer, rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=device)
    t_eig = time.time() - t0
    print(f"  eigendecompose+fit_lambda in {t_eig:.0f}s; Lambda {tuple(Lambda.shape)}")

    for name, t in [("A", A), ("S", S), ("Q_A", Q_A), ("Q_S", Q_S),
                    ("Lambda_A", Lambda_A), ("Lambda_S", Lambda_S), ("Lambda", Lambda)]:
        torch.save(t.cpu(), FACTOR_DIR / f"{name}.pt")
    meta = {
        "model": THETA_STAR_ID, "layer": PHI_LAYER, "phi_shape": list(layer.weight.shape),
        "n_rollouts": len(rollouts), "n_response_tokens_AS": int(n_as),
        "n_tok_lambda": int(n_lam), "token_budget": budget, "k_per_prompt": k_per_prompt,
        "damping_floor": cfg.damping_floor, "damping_alpha": cfg.damping_alpha, "seed": SEED,
        "timings_s": {"gen": t_gen, "accumulate_AS": t_as, "eigen_lambda": t_eig},
    }
    (FACTOR_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved factors to {FACTOR_DIR}")
    print(f"  Lambda range [{float(Lambda.min()):.3e}, {float(Lambda.max()):.3e}], "
          f"mean {float(Lambda.mean()):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
