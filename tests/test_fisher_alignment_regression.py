"""REGRESSION GATE — Stage-1 pseudo-label row alignment of the EK-FAC Fisher.

Polarity: PASS == code is CORRECT (row-aligned). This file exercises the
PROJECT pipeline (`accumulate_AS`, `fit_lambda`, `inverse_hvp_additive`,
`influence_score_klrl`) and compares its output against an INDEPENDENT
closed-form ground truth computed with numpy from the flat softmax model's
analytic per-token Fisher — the truth NEVER reuses the pipeline's label logic.

This is the red->green sentinel for the off-by-one fix:
  * on the UN-fixed code (factors.py/eigen.py use response_labels=pseudo[P:])
    both gates FAIL — the pipeline Fisher converges to the off-by-one biased
    limit, ~25% off in S and ~42% off in the end-to-end influence.
  * on the FIXED code (response_labels=pseudo[P-1:T-1]) both gates PASS — the
    pipeline reproduces the brute-force closed-form influence to a few percent
    and the ranking Spearman hits ~1.0.

Why this catches the bug when the old toy gate did not: the old gate
(test_eigen_toy.test_lambda_matches_kronecker_diagonal) compared fit_lambda
against `compute_per_token_fisher_beta`, which SHARES the same
sample_pseudo_labels + pseudo[plen:] slice — so it was bug-to-bug self-
consistent and passed regardless. Here the truth is the analytic Fisher of the
flat model (no sampling, no pipeline code), so it cannot share the bug.

Run standalone (prints red/green numbers):
    python3 tests/test_fisher_alignment_regression.py
Run under pytest:
    pytest tests/test_fisher_alignment_regression.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda, inverse_hvp_additive  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402

# Independent closed-form references (numpy, no pipeline label logic).
from audit_3_known_answer_e2e import (  # noqa: E402
    FlatSoftmaxModel,
    build_rollouts_v1,
    closed_form_grad,
    exact_fisher,
)

DEV = torch.device("cpu")
S_REL_TOL = 0.05          # buggy ~0.25, fixed ~0.01-0.02
INF_REL_TOL = 0.06        # buggy ~0.42, fixed ~0.03
INF_SPEARMAN_TOL = 0.99   # buggy ~0.95, fixed ~1.0


def _flat_model():
    torch.manual_seed(0)
    return FlatSoftmaxModel(vocab=12, d_in=6, d_out=5, seed=11).eval()


def measure_S_alignment(n_fisher: int = 8000) -> float:
    """rel Frobenius between the pipeline's S factor and the analytic
    row-aligned per-token Fisher S (closed form). Constant-context rollouts so
    the truth is a single deterministic Sigma."""
    cfg = EKFACConfig(layer_name="transformer.h.0.mlp.c_proj")
    model = _flat_model()
    layer = model.get_submodule(cfg.layer_name)
    rolls = build_rollouts_v1(n_fisher, vocab=12, ctx_token=3, seed=1)
    _, S_pipe, _ = accumulate_AS(model, layer, rolls, cfg, device=DEV)
    emb, head = model.emb.numpy(), model.head.numpy()
    W = layer.weight.detach().numpy()
    _, _, S_correct, _ = exact_fisher(rolls, emb, head, W)   # analytic, row-aligned
    return float(np.linalg.norm(S_pipe.numpy() - S_correct) / np.linalg.norm(S_correct))


def measure_influence_alignment(n_fisher: int = 6000) -> tuple[float, float]:
    """End-to-end: pipeline influence vs brute-force closed-form influence
    -g_eval^T inv(F_exact + damp) s_m on the flat model."""
    cfg = EKFACConfig(layer_name="transformer.h.0.mlp.c_proj")
    damp = cfg.toy_damping
    model = _flat_model()
    layer = model.get_submodule(cfg.layer_name)
    emb, head = model.emb.numpy(), model.head.numpy()
    W = layer.weight.detach().numpy()
    dim = W.size

    rolls = build_rollouts_v1(n_fisher, vocab=12, ctx_token=3, seed=2)
    F_exact, _, _, _ = exact_fisher(rolls, emb, head, W)
    Finv = np.linalg.inv(F_exact + damp * np.eye(dim))
    ev = build_rollouts_v1(3, vocab=12, ctx_token=3, seed=501)
    tr = build_rollouts_v1(80, vocab=12, ctx_token=3, seed=500)
    g_true = [closed_form_grad(r, emb, head, W) for r in ev]
    s_true = [closed_form_grad(r, emb, head, W) for r in tr]
    I_true = np.array([[-g.reshape(-1) @ Finv @ s.reshape(-1) for s in s_true]
                       for g in g_true])

    A, S, _ = accumulate_AS(model, layer, rolls, cfg, device=DEV)
    Q_A, _, Q_S, _ = eigendecompose(A, S)
    Lam, _ = fit_lambda(model, layer, rolls, cfg, Q_A=Q_A, Q_S=Q_S, device=DEV)
    g_pipe = [compute_per_sample_grad(model, layer, r, cfg, device=DEV).double() for r in ev]
    s_pipe = [compute_per_sample_grad(model, layer, r, cfg, device=DEV).double() for r in tr]
    I_pipe = np.empty_like(I_true)
    for a, g in enumerate(g_pipe):
        p = inverse_hvp_additive(g, Q_A, Q_S, Lam, damping=damp)
        for b, s in enumerate(s_pipe):
            I_pipe[a, b] = influence_score_klrl(s, p)

    from scipy.stats import spearmanr
    rms = np.sqrt((I_true ** 2).mean())
    rel = float(np.sqrt(((I_pipe - I_true) ** 2).mean()) / rms)
    rho = float(np.mean([spearmanr(I_pipe[a], I_true[a]).statistic
                         for a in range(I_true.shape[0])]))
    return rel, rho


# --------------------------- pytest entry points --------------------------- #
def test_accumulate_AS_S_is_row_aligned_fisher() -> None:
    rel = measure_S_alignment()
    assert rel < S_REL_TOL, (
        f"pipeline S vs analytic row-aligned Fisher rel err = {rel:.4f} >= {S_REL_TOL} "
        f"-> Stage-1 pseudo-labels are off-by-one (response_labels should be "
        f"pseudo[P-1:T-1], not pseudo[P:])"
    )


def test_end_to_end_influence_matches_bruteforce() -> None:
    rel, rho = measure_influence_alignment()
    assert rel < INF_REL_TOL and rho > INF_SPEARMAN_TOL, (
        f"end-to-end influence vs brute-force truth: rel err = {rel:.4f} "
        f"(tol {INF_REL_TOL}), Spearman = {rho:.4f} (tol {INF_SPEARMAN_TOL}) "
        f"-> off-by-one pseudo-label bias"
    )


def main() -> int:
    print("REGRESSION: Stage-1 pseudo-label row alignment (PASS = fixed)")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    rel_S = measure_S_alignment()
    ok_S = rel_S < S_REL_TOL
    print(f"  [{'PASS' if ok_S else 'FAIL'}] S factor vs analytic row-aligned Fisher: "
          f"rel err = {rel_S:.4f} (tol {S_REL_TOL})")
    rel_I, rho = measure_influence_alignment()
    ok_I = rel_I < INF_REL_TOL and rho > INF_SPEARMAN_TOL
    print(f"  [{'PASS' if ok_I else 'FAIL'}] end-to-end influence vs brute-force truth: "
          f"rel err = {rel_I:.4f} (tol {INF_REL_TOL}), Spearman = {rho:.4f} (tol {INF_SPEARMAN_TOL})")
    n_fail = (not ok_S) + (not ok_I)
    print(f"\n{'ALL PASS (code is row-aligned/correct)' if n_fail == 0 else f'{n_fail} FAIL (off-by-one present)'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
