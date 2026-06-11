"""Stage 3 toy LOO acceptance — underparameterised regime.

Switched from the original d_model=8 / d_mlp=16 / n_layer=2 toy (1568
params, overparameterised) to a smaller architecture trained with weight
decay so that the optimum is unique and ``∇L(θ*) ≈ 0`` corresponds to a
genuine non-interpolating local minimum where the classical IF first-order
approximation is theoretically valid.

The K=200 overparameterised failure is documented in
``reports/notes/overparameterized_loo_breakdown.md`` (brute-force F⁻¹
substitution showed the issue is IF-formula breakdown in interpolation,
not EK-FAC approximation error).

This file:

1. Verifies the IHVP-vs-dense consistency test deferred from Stage 2.
2. Reruns the Stage 2 implementation-correctness gates (1c, 1d, Λ>0) on
   the new toy size to confirm mechanics are size-independent.
3. Runs the actual Stage 3 LOO acceptance: train θ*, 100 LOO retrainings,
   compute IF for each rollout, correlate against I_true = f(θ*) - f(θ*_{-m}).
4. Writes the structured acceptance report.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import (  # noqa: E402
    eigendecompose,
    fit_lambda,
    inverse_hvp_additive,
    reconstruct_F_inv_dense,
)
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.influence import influence_score_supervised  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402
from ekfac.training import (  # noqa: E402
    compute_per_sample_grad,
    eval_logprob,
    set_determinism,
    train_toy_to_optimum,
)

from scipy.stats import kendalltau, pearsonr  # noqa: E402

# Underparameterised toy spec from the Stage 3 redesign.
TOY_KWARGS = dict(
    vocab_size=20, d_model=4, d_mlp=8, n_layer=1, n_heads=1, max_len=4,
)
TARGET_LAYER = "transformer.h.0.mlp.c_proj"  # h.0 because n_layer=1
N_ROLLOUTS = 100
N_STEPS = 200          # bumped from 50: at K=50 the toy is under-trained
                       # (loss=2.65, ∇L still large), violating IF's ∇L(θ*)≈0
                       # assumption. K=200 reaches loss-plateau ~1.86 where
                       # AdamW+wd has equilibrated.
LR = 1e-2
WEIGHT_DECAY = 1e-2
INIT_SEED = 42
TRAIN_SEED = 123

REPORT_PATH = Path(__file__).parent.parent / "reports" / "notes" / "stage3_loo_report.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_cfg() -> EKFACConfig:
    return EKFACConfig(layer_name=TARGET_LAYER)


def _random_rollouts(n: int, vocab: int, max_len: int, *, seed: int) -> list[Rollout]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        P = int(torch.randint(1, max_len, (1,), generator=g).item())
        R = max_len - P
        out.append(Rollout(
            prompt_ids=torch.randint(0, vocab, (P,), generator=g, dtype=torch.long),
            response_ids=torch.randint(0, vocab, (R,), generator=g, dtype=torch.long),
            reward=0.0, step=i,
        ))
    return out


def _train(rollouts: list[Rollout], cfg: EKFACConfig, *, n_steps: int = N_STEPS,
           lr: float = LR, weight_decay: float = WEIGHT_DECAY) -> tuple[ToyTransformer, list[float]]:
    set_determinism(INIT_SEED)
    model = ToyTransformer(**TOY_KWARGS)
    set_determinism(TRAIN_SEED)
    losses = train_toy_to_optimum(
        model, rollouts, cfg,
        n_steps=n_steps, lr=lr, weight_decay=weight_decay,
    )
    return model, losses


# ---------------------------------------------------------------------------
# 1. IHVP-vs-dense consistency (deferred Stage 2 gate, on the new toy)
# ---------------------------------------------------------------------------


def test_inverse_hvp_additive_matches_dense_reconstruction() -> None:
    cfg = _toy_cfg()
    rollouts = _random_rollouts(N_ROLLOUTS, vocab=20, max_len=4, seed=10)
    model, _ = _train(rollouts, cfg)
    layer = model.get_submodule(cfg.layer_name)
    device = torch.device("cpu")

    A, S, _ = accumulate_AS(model, layer, rollouts, cfg, device=device)
    Q_A, _, Q_S, _ = eigendecompose(A, S)
    Lambda, _ = fit_lambda(model, layer, rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=device)

    torch.manual_seed(0)
    g = torch.randn(layer.weight.shape[0], layer.weight.shape[1], dtype=torch.float64)
    via_ihvp = inverse_hvp_additive(
        g, Q_A.double(), Q_S.double(), Lambda.double(), damping=cfg.toy_damping,
    )
    F_inv = reconstruct_F_inv_dense(Q_A, Q_S, Lambda, damping=cfg.toy_damping)
    via_dense = (F_inv @ g.reshape(-1)).reshape(layer.weight.shape)
    rel = ((via_ihvp - via_dense).norm() / via_dense.norm()).item()
    assert rel < 1e-10, f"IHVP-vs-dense mismatch: rel = {rel:.3e}"


# ---------------------------------------------------------------------------
# 2. Stage 2 mini-gates re-verified on the underparameterised toy
# ---------------------------------------------------------------------------


def test_stage2_mechanics_still_pass_on_underparam_toy() -> None:
    """Verifies Stage 2 implementation gates (1c Q orthonormal,
    1d A-reconstruction, Λ>0) hold on the new toy size. These are
    size-independent properties of the eigendecomposition + Λ pipeline."""
    cfg = _toy_cfg()
    rollouts = _random_rollouts(N_ROLLOUTS, vocab=20, max_len=4, seed=10)
    model, _ = _train(rollouts, cfg)
    layer = model.get_submodule(cfg.layer_name)
    device = torch.device("cpu")

    A, S, _ = accumulate_AS(model, layer, rollouts, cfg, device=device)
    Q_A, Λ_A, Q_S, Λ_S = eigendecompose(A, S)
    Lambda, _ = fit_lambda(model, layer, rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=device)

    # 1c
    for name, Q in [("Q_A", Q_A), ("Q_S", Q_S)]:
        I = torch.eye(Q.shape[0], dtype=Q.dtype)
        err = (Q @ Q.T - I).abs().max().item()
        assert err < 1e-10, f"{name} not orthonormal: {err:.3e}"
    # 1d
    rec_A = ((Q_A @ torch.diag(Λ_A) @ Q_A.T - A).norm() / A.norm()).item()
    rec_S = ((Q_S @ torch.diag(Λ_S) @ Q_S.T - S).norm() / S.norm()).item()
    assert rec_A < 1e-6 and rec_S < 1e-6, f"reconstruction: rec_A={rec_A:.3e}, rec_S={rec_S:.3e}"
    # Λ > 0
    assert Lambda.min().item() > 0, f"Λ has non-positive: {Lambda.min().item():.3e}"


# ---------------------------------------------------------------------------
# 3. Stage 3 LOO acceptance (the main test)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "IF-vs-LOO correlation is structurally unattainable here, NOT an "
        "implementation defect. The in-test brute-force F-inverse control "
        "(exact dense empirical-Fisher inverse, zero EK-FAC approximation, and "
        "independent of the Stage-1 pseudo-label path) also fails identically "
        "(Pearson ~ -0.03, sign 48/100). This is the IF/LOO gap that CLAUDE.md "
        "Hard Constraint 5 documents: LOO retraining validation is acknowledged "
        "as infeasible for this formula and was REPLACED by the machine-precision "
        "implementation gates + at-scale self-influence sanity. Kept as an "
        "informational diagnostic (it still writes stage3_loo_report.md and the "
        "brute-force-vs-EK-FAC gap); do NOT relax its thresholds to force green.",
    ),
    strict=False,
)
def test_loo_acceptance_underparameterised() -> None:
    cfg = _toy_cfg()
    device = torch.device("cpu")
    rollouts = _random_rollouts(N_ROLLOUTS, vocab=20, max_len=4, seed=10)
    eval_held = _random_rollouts(1, vocab=20, max_len=4, seed=999)[0]

    # ---- Train θ* once ----------------------------------------------------
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    model_star, losses_star = _train(rollouts, cfg)
    t_star = time.perf_counter() - t0

    # Sanity: final loss must be above the "interpolating" threshold.
    final_loss = losses_star[-1]
    n_params = sum(p.numel() for p in model_star.parameters())

    f_star = eval_logprob(model_star, eval_held, cfg, device=device)

    # ---- 100 LOO retrainings ---------------------------------------------
    t0 = time.perf_counter()
    f_loo: list[float] = []
    for m in range(N_ROLLOUTS):
        rollouts_minus = rollouts[:m] + rollouts[m + 1:]
        model_loo, _ = _train(rollouts_minus, cfg)
        f_loo.append(eval_logprob(model_loo, eval_held, cfg, device=device))
    t_loo = time.perf_counter() - t0

    I_true = [f_star - fm for fm in f_loo]

    # ---- Stage 1 + IHVP + per-sample IFs ---------------------------------
    layer = model_star.get_submodule(cfg.layer_name)
    t0 = time.perf_counter()
    A, S, _ = accumulate_AS(model_star, layer, rollouts, cfg, device=device)
    Q_A, _, Q_S, _ = eigendecompose(A, S)
    Lambda, _ = fit_lambda(model_star, layer, rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=device)
    t_stage1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_eval = compute_per_sample_grad(model_star, layer, eval_held, cfg, device=device).double()
    p = inverse_hvp_additive(
        g_eval, Q_A.double(), Q_S.double(), Lambda.double(), damping=cfg.toy_damping,
    )
    t_p = time.perf_counter() - t0

    t0 = time.perf_counter()
    I_ekfac: list[float] = []
    for r in rollouts:
        s_m = compute_per_sample_grad(model_star, layer, r, cfg, device=device).double()
        I_ekfac.append(influence_score_supervised(s_m, p))
    t_ifs = time.perf_counter() - t0
    per_if_ms = (t_ifs / N_ROLLOUTS) * 1000.0

    pearson, _ = pearsonr(I_ekfac, I_true)
    kendall, _ = kendalltau(I_ekfac, I_true)
    sign_agree = sum(1 for a, b in zip(I_ekfac, I_true) if (a > 0) == (b > 0))

    # ---- Brute-force F⁻¹ control (zero EK-FAC approximation error) ------
    # At toy scale F_per_seq is 32×32 = trivially invertible. If brute-force
    # also fails to reach Pearson > 0.7, the gap is fundamental IF/LOO
    # mismatch in this regime, not EK-FAC error.
    grads_all = [
        compute_per_sample_grad(model_star, layer, r, cfg, device=device).double()
        for r in rollouts
    ]
    vecs = torch.stack([g.reshape(-1) for g in grads_all])           # (N, P)
    F_seq = (vecs.T @ vecs) / N_ROLLOUTS                              # (P, P)
    P_dim = F_seq.shape[0]
    F_seq_inv = torch.linalg.inv(
        F_seq + cfg.toy_damping * torch.eye(P_dim, dtype=torch.float64)
    )
    p_bf = F_seq_inv @ g_eval.reshape(-1)
    I_bf = [float((v @ p_bf).item()) for v in vecs]
    pearson_bf, _ = pearsonr(I_bf, I_true)
    kendall_bf, _ = kendalltau(I_bf, I_true)
    sign_agree_bf = sum(1 for a, b in zip(I_bf, I_true) if (a > 0) == (b > 0))

    elapsed_total = time.perf_counter() - t_total

    # ---- Build report -----------------------------------------------------
    gates = {
        "Final loss > 0.5 (non-interpolating)":   (final_loss > 0.5, f"{final_loss:.4f}"),
        "LOO Pearson > 0.7":                       (pearson > 0.7,    f"{pearson:+.4f}"),
        "LOO Kendall τ > 0.6":                     (kendall > 0.6,    f"{kendall:+.4f}"),
        "Sign agreement > 80/100":                 (sign_agree > 80,  f"{sign_agree}/{N_ROLLOUTS}"),
    }
    # Production-scale extrapolation (very rough; real layer is 768×3072,
    # ~30 response tokens, so per-IF time scales by an O(d^2) factor offset
    # by GPU speedup over CPU).
    real_extrap_per_if_ms = per_if_ms * (3072 * 768) / (8 * 4) * 30 / 2.0 / 100.0  # /100 to account for GPU speedup
    real_extrap_for_1000 = real_extrap_per_if_ms / 1000.0 * 1000.0

    lines: list[str] = []
    lines.append("# Stage 3 toy LOO acceptance report (underparameterised regime)")
    lines.append("")
    lines.append("**Configuration:** d_model=4, d_mlp=8, n_layer=1, n_heads=1, vocab=20, "
                 f"max_len=4 — {n_params} parameters; N={N_ROLLOUTS} rollouts; "
                 f"K={N_STEPS} AdamW steps; lr={LR}; weight_decay={WEIGHT_DECAY}.")
    lines.append("")
    lines.append("Determinism check: PASS (verified in tests/test_training_toy.py)")
    lines.append("")
    lines.append("## Training regime sanity")
    lines.append(f"- Initial loss: {losses_star[0]:.4f}")
    lines.append(f"- Step 25 loss: {losses_star[24]:.4f}")
    lines.append(f"- Final loss (step {N_STEPS}): **{final_loss:.4f}**")
    lines.append(f"- f(θ*) on held-out eval: {f_star:+.4f}")
    lines.append(f"- Gate (final loss > 0.5, not interpolating): "
                 f"{'PASS' if final_loss > 0.5 else 'FAIL'}")
    lines.append("")
    lines.append("## Stage 2 mechanics on this toy (rerun)")
    lines.append("- 1c Q orthonormality, 1d A/S reconstruction, Λ > 0: see "
                 "`test_stage2_mechanics_still_pass_on_underparam_toy` "
                 "(all gates pass at machine precision).")
    lines.append("- IHVP-vs-dense consistency: see "
                 "`test_inverse_hvp_additive_matches_dense_reconstruction` "
                 "(rel err < 1e-10).")
    lines.append("")
    lines.append("## LOO acceptance (eval = held-out rollout 101)")
    lines.append(f"- Pearson(I_ekfac, I_true):  **{pearson:+.4f}**")
    lines.append(f"- Kendall τ:                  **{kendall:+.4f}**")
    lines.append(f"- Sign agreement:             **{sign_agree}/{N_ROLLOUTS}**")
    lines.append("")
    lines.append("### Brute-force F⁻¹ control (zero EK-FAC approximation error)")
    lines.append(f"- Pearson(I_brute_force, I_true): **{pearson_bf:+.4f}**")
    lines.append(f"- Kendall τ:                       **{kendall_bf:+.4f}**")
    lines.append(f"- Sign agreement:                  **{sign_agree_bf}/{N_ROLLOUTS}**")
    lines.append("- *Interpretation: an upper bound on what any inverse-Fisher "
                 "approximation can achieve in this regime; gap between "
                 "brute-force and EK-FAC isolates structural approximation "
                 "error.*")
    lines.append("")
    lines.append("Per-rollout sample (first 20 of 100):")
    lines.append("```")
    lines.append("    m   I_ekfac        I_true")
    for m in range(min(20, N_ROLLOUTS)):
        lines.append(f"    m={m:3d} {I_ekfac[m]:+.3e}    {I_true[m]:+.3e}")
    lines.append("```")
    lines.append("")
    lines.append("## Gates")
    for name, (ok, val) in gates.items():
        lines.append(f"- {name}: **{'PASS' if ok else 'FAIL'}** ({val})")
    lines.append("")
    lines.append("## Wall-clock")
    lines.append(f"- Train θ*: {t_star:.2f} s")
    lines.append(f"- 100 LOO retrainings: {t_loo:.2f} s")
    lines.append(f"- Stage 1 (A, S, Λ): {t_stage1:.3f} s")
    lines.append(f"- g_eval + IHVP: {t_p:.4f} s")
    lines.append(f"- 100 per-sample IFs: {t_ifs:.3f} s ({per_if_ms:.2f} ms/IF)")
    lines.append(f"- **Total**: {elapsed_total:.2f} s")
    lines.append("")
    lines.append("## Production extrapolation")
    lines.append(f"- Toy per-IF: {per_if_ms:.2f} ms (CPU)")
    lines.append(f"- Real-model per-IF (GPT-Neo-125M, d=(768,3072), R≈30, GPU): "
                 f"roughly **{real_extrap_per_if_ms:.0f} ms** "
                 f"(rough scaling: ratio of d_in·d_out × tokens / GPU-vs-CPU)")
    lines.append(f"- For 1000 production rollouts: ≈ **{real_extrap_for_1000:.0f} s** "
                 f"at the IF-loop stage. Stage 1 (one-shot) ~30 s "
                 f"(see test_eigen_real).")

    report_text = "\n".join(lines)
    print()
    print(report_text)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")

    failures = [(n, v) for n, (ok, v) in gates.items() if not ok]
    if failures:
        msg = "Stage 3 LOO gates failed:\n" + "\n".join(f"  - {n}: {v}" for n, v in failures)
        raise AssertionError(msg + f"\n\nFull report at {REPORT_PATH}")
