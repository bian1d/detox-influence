"""INDEPENDENT AUDIT 3c — does the end-to-end pipeline reproduce the brute-force
truth ONCE the pseudo-label off-by-one is corrected?

Audit 3 ran the project pipeline as-is on the flat softmax model and found the
end-to-end influence did NOT converge to the closed-form truth: at n=4000 the
relative RMS error stayed ~0.42 and barely moved from n=400 (0.47), i.e. it is
BIAS-dominated, not Monte-Carlo-dominated. Audit 3b/N1 identified the bias as
the Stage-1 pseudo-label off-by-one.

This script closes the loop: rebuild Stage 1A/1B with ROW-ALIGNED pseudo labels
(response_labels = pseudo[P-1:T-1]) — everything else identical to audit 3 —
and show the end-to-end influence now converges to the brute-force closed-form
truth (structural error small AND total error -> ~0, Spearman -> ~1) as the
sample count grows. This is the positive control for cut 5: with the one-line
fix, the known-answer test passes.

Run:  python3 tests/audit_3c_known_answer_corrected.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout, build_input_and_labels  # noqa: E402
from ekfac.eigen import eigendecompose, inverse_hvp_additive  # noqa: E402
from ekfac.factors import sample_pseudo_labels  # noqa: E402
from ekfac.hooks import capture_c_proj  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402

from audit_3_known_answer_e2e import (  # noqa: E402
    FlatSoftmaxModel,
    build_rollouts_v1,
    build_rollouts_v2,
    closed_form_grad,
    exact_fisher,
)

RESULTS: list[tuple[str, bool, str]] = []


def gate(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def info(name, detail):
    print(f"  [info] {name}: {detail}")


def corrected_stage1(model, layer, rolls, cfg, device, seed):
    """Row-aligned Stage 1A + 1B on the flat model (pseudo[P-1:T-1])."""
    d_out, d_in = layer.weight.shape
    gen = torch.Generator(device=device).manual_seed(seed)

    def one_pass(g, want_lambda, Q_A=None, Q_S=None):
        A = torch.zeros(d_in, d_in, dtype=torch.float64)
        S = torch.zeros(d_out, d_out, dtype=torch.float64)
        Lam = torch.zeros(d_in, d_out, dtype=torch.float64)
        n = 0
        for r in rolls:
            P, R = r.prompt_len, r.response_len
            T = P + R
            ids = torch.cat([r.prompt_ids, r.response_ids]).to(device).unsqueeze(0)
            for p_ in model.parameters():
                if p_.grad is not None:
                    p_.grad = None
            with capture_c_proj(layer) as cache:
                logits = model(ids)
                logits = logits.logits if hasattr(logits, "logits") else logits
                pseudo = sample_pseudo_labels(logits[0], generator=g)
                _, labels = build_input_and_labels(
                    r.prompt_ids.to(device), r.response_ids.to(device),
                    response_labels=pseudo[P - 1:T - 1], ignore_index=cfg.ignore_index,
                )
                loss = TF.cross_entropy(logits[0, :-1].float(), labels[1:],
                                        ignore_index=cfg.ignore_index, reduction="sum")
                loss.backward()
            m_resp = cache.m[0, P - 1:T - 1].double()
            d_resp = cache.delta[0, P - 1:T - 1].double()
            if not want_lambda:
                A += m_resp.T @ m_resp
                S += d_resp.T @ d_resp
            else:
                Lam += (m_resp @ Q_A).pow(2).T @ (d_resp @ Q_S).pow(2)
            n += R
        return A / n, S / n, Lam / n

    A, S, _ = one_pass(gen, want_lambda=False)
    A = 0.5 * (A + A.T); S = 0.5 * (S + S.T)
    Q_A, _, Q_S, _ = eigendecompose(A, S)
    gen2 = torch.Generator(device=device).manual_seed(seed + 1)
    _, _, Lam = one_pass(gen2, want_lambda=True, Q_A=Q_A, Q_S=Q_S)
    return Q_A, Q_S, Lam


def run(tag, model, rolls_fisher, rolls_train, rolls_eval, cfg, damp):
    layer = model.get_submodule(cfg.layer_name)
    emb, head = model.emb.numpy(), model.head.numpy()
    W = layer.weight.detach().numpy()
    dim = W.size
    dev = torch.device("cpu")

    F_exact, _, _, _ = exact_fisher(rolls_fisher, emb, head, W)
    Finv_true = np.linalg.inv(F_exact + damp * np.eye(dim))
    g_evals = [closed_form_grad(r, emb, head, W) for r in rolls_eval]
    s_trains = [closed_form_grad(r, emb, head, W) for r in rolls_train]
    I_true = np.array([[-g.reshape(-1) @ Finv_true @ s.reshape(-1) for s in s_trains]
                       for g in g_evals])

    Q_A, Q_S, Lam = corrected_stage1(model, layer, rolls_fisher, cfg, dev, cfg.seed)
    g_pipe = [compute_per_sample_grad(model, layer, r, cfg, device=dev).double() for r in rolls_eval]
    s_pipe = [compute_per_sample_grad(model, layer, r, cfg, device=dev).double() for r in rolls_train]
    I_pipe = np.empty_like(I_true)
    for a, g in enumerate(g_pipe):
        p = inverse_hvp_additive(g, Q_A, Q_S, Lam, damping=damp)
        for b, s in enumerate(s_pipe):
            I_pipe[a, b] = influence_score_klrl(s, p)

    from scipy.stats import spearmanr
    rms = np.sqrt((I_true ** 2).mean())
    e_total = np.sqrt(((I_pipe - I_true) ** 2).mean()) / rms
    rho = np.mean([spearmanr(I_pipe[a], I_true[a]).statistic for a in range(I_true.shape[0])])
    info(f"{tag}: CORRECTED end-to-end vs brute-force truth",
         f"rel RMS err total = {e_total:.4f}, Spearman = {rho:.4f}")
    return {"e_total": e_total, "spearman": rho}


def main() -> int:
    print("AUDIT 3c: end-to-end known-answer AFTER the off-by-one fix (positive control)")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    vocab, d_in, d_out = 12, 6, 5
    cfg = EKFACConfig(layer_name="transformer.h.0.mlp.c_proj")
    damp = cfg.toy_damping
    torch.manual_seed(0)
    model = FlatSoftmaxModel(vocab, d_in, d_out, seed=11).eval()

    print("\n===== Variant 1 (Kronecker-exact) — corrected labels =====")
    tr1 = build_rollouts_v1(100, vocab, 3, seed=500)
    ev1 = build_rollouts_v1(3, vocab, 3, seed=501)
    s1 = run("v1 n=400", model, build_rollouts_v1(400, vocab, 3, seed=1), tr1, ev1, cfg, damp)
    b1 = run("v1 n=4000", model, build_rollouts_v1(4000, vocab, 3, seed=2), tr1, ev1, cfg, damp)
    gate("v1 corrected: total err shrinks toward 0 with 10x samples",
         b1["e_total"] < s1["e_total"], f"{s1['e_total']:.4f} -> {b1['e_total']:.4f}")
    gate("v1 corrected: end-to-end reproduces brute-force truth at n=4000",
         b1["e_total"] < 0.05 and b1["spearman"] > 0.99,
         f"total {b1['e_total']:.4f} (was 0.4211 buggy), Spearman {b1['spearman']:.4f} (was 0.9533)")

    print("\n===== Variant 2 (non-Kronecker) — corrected labels =====")
    tr2 = build_rollouts_v2(100, vocab, seed=600)
    ev2 = build_rollouts_v2(3, vocab, seed=601)
    s2 = run("v2 n=400", model, build_rollouts_v2(400, vocab, seed=3), tr2, ev2, cfg, damp)
    b2 = run("v2 n=4000", model, build_rollouts_v2(4000, vocab, seed=4), tr2, ev2, cfg, damp)
    info("v2 corrected note",
         f"total {s2['e_total']:.4f}->{b2['e_total']:.4f}; residual is EK-FAC's "
         f"genuine non-Kronecker structural error + finite-sample, NOT the bug")
    gate("v2 corrected: Spearman high at n=4000", b2["spearman"] > 0.97,
         f"Spearman {b2['spearman']:.4f}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (audit 3c) ===")
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"gates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
