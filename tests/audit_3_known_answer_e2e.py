"""INDEPENDENT AUDIT 3 — end-to-end known-answer test for the full IF pipeline.

Construction: a flat softmax model with NO attention and NO cross-position
flow — input token -> frozen embedding -> c_proj (the ONLY trainable layer)
-> frozen readout -> logits. On this model every per-position label
distribution is independent, so the token-level Fisher the pipeline targets
has an EXACT closed form (enumerate the label expectation analytically — no
Monte Carlo on the ground-truth side):

    Sigma_t = head^T (diag(p_t) - p_t p_t^T) head        (exact, per position)
    F_exact = mean_t kron(Sigma_t, outer(m_t, m_t))      (dense, fp64)

and the per-sample mean-reduced gradient of log-likelihood is also closed
form. Ground-truth influence is then the "stupidest possible" computation:

    I_true(m) = - vec(g_eval)^T  inv(F_exact + damp*I)  vec(s_m)

computed entirely in numpy with my own conventions. The project pipeline
(accumulate_AS -> eigendecompose -> fit_lambda -> inverse_hvp_additive ->
influence_score_klrl) is then run END TO END as a black box on the same
rollouts and its final influence numbers are compared to I_true.

Error decomposition via a same-draw dense empirical Fisher (passive hooks):
    |I_pipeline - I_emp|   = EK-FAC structural error (Kronecker + diag refit)
    |I_emp     - I_true|   = pseudo-label Monte-Carlo error (shrinks ~1/sqrt(n))

Variant 1 (constant context token): F_exact is EXACTLY Kronecker, so the
structural error must vanish asymptotically — any persistent gap is a bug.
Variant 2 (random tokens): F_exact is genuinely non-Kronecker; the gap is
the honest end-to-end approximation error of the production method.

Run:  python3 tests/audit_3_known_answer_e2e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda, inverse_hvp_additive  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def info(name: str, detail: str) -> None:
    print(f"  [info] {name}: {detail}")


# --------------------------------------------------------------------------- #
# Flat model: token -> emb (frozen) -> c_proj (trainable) -> head (frozen)
# --------------------------------------------------------------------------- #


class FlatSoftmaxModel(nn.Module):
    def __init__(self, vocab: int, d_in: int, d_out: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("emb", torch.randn(vocab, d_in, generator=g, dtype=torch.float64))
        self.register_buffer(
            "head", torch.randn(vocab, d_out, generator=g, dtype=torch.float64) / d_out**0.5
        )
        c_proj = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
        self.transformer = nn.ModuleDict(
            {"h": nn.ModuleList([nn.ModuleDict({"mlp": nn.ModuleDict({"c_proj": c_proj})})])}
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.emb[input_ids]                                    # (B, T, d_in)
        out = self.transformer["h"][0]["mlp"]["c_proj"](m)         # (B, T, d_out)
        return out @ self.head.T                                   # (B, T, V)


class PassiveCapture:
    """Independent hooks recording (m, delta) per forward/backward on `layer`."""

    def __init__(self, layer: nn.Module) -> None:
        self.ms: list[torch.Tensor] = []
        self.ds: list[torch.Tensor] = []
        self._h1 = layer.register_forward_hook(
            lambda _m, args, _o: self.ms.append(args[0].detach().clone().double())
        )
        self._h2 = layer.register_full_backward_hook(
            lambda _m, _gi, go: self.ds.append(go[0].detach().clone().double())
        )

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()


# --------------------------------------------------------------------------- #
# My closed-form reference math (numpy fp64, independent conventions)
# --------------------------------------------------------------------------- #


def softmax_np(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def window_terms(roll: Rollout, emb: np.ndarray, head: np.ndarray, W: np.ndarray):
    """Per loss row t in [P-1, T-2]: (m_t, p_t, label_t). My own derivation of
    the response window: shift-CE row t predicts token t+1; rows with labels
    are exactly P-1 .. T-2."""
    ids = np.concatenate([roll.prompt_ids.numpy(), roll.response_ids.numpy()])
    P, T = roll.prompt_len, roll.prompt_len + roll.response_len
    out = []
    for t in range(P - 1, T - 1):
        m_t = emb[ids[t]]
        p_t = softmax_np(head @ (W @ m_t))
        out.append((m_t, p_t, int(ids[t + 1])))
    return out


def exact_fisher(rolls, emb, head, W) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """F_exact (dense, label expectation enumerated), A_exact, S_exact."""
    d_in, d_out = emb.shape[1], head.shape[1]
    dim = d_in * d_out
    F = np.zeros((dim, dim))
    A = np.zeros((d_in, d_in))
    S = np.zeros((d_out, d_out))
    n = 0
    for r in rolls:
        for m_t, p_t, _ in window_terms(r, emb, head, W):
            Sigma = head.T @ (np.diag(p_t) - np.outer(p_t, p_t)) @ head
            F += np.kron(Sigma, np.outer(m_t, m_t))
            A += np.outer(m_t, m_t)
            S += Sigma
            n += 1
    return F / n, A / n, S / n, n


def closed_form_grad(roll, emb, head, W) -> np.ndarray:
    """Mean-reduced grad of log-likelihood of recorded labels, (d_out, d_in)."""
    terms = window_terms(roll, emb, head, W)
    g = np.zeros((head.shape[1], emb.shape[1]))
    for m_t, p_t, y in terms:
        e_y = np.zeros(p_t.shape[0])
        e_y[y] = 1.0
        g += np.outer(head.T @ (e_y - p_t), m_t)
    return g / len(terms)


def build_rollouts_v1(n: int, vocab: int, ctx_token: int, seed: int) -> list[Rollout]:
    """Constant-context variant: prompt [c, c], one free response token."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        y = torch.randint(0, vocab, (1,), generator=g)
        out.append(Rollout(
            prompt_ids=torch.tensor([ctx_token, ctx_token]),
            response_ids=y, reward=0.0, step=0,
        ))
    return out


def build_rollouts_v2(n: int, vocab: int, seed: int, T: int = 6) -> list[Rollout]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        P = int(torch.randint(1, T - 1, (1,), generator=g).item())  # R >= 2
        ids = torch.randint(0, vocab, (T,), generator=g)
        out.append(Rollout(prompt_ids=ids[:P], response_ids=ids[P:], reward=0.0, step=0))
    return out


# --------------------------------------------------------------------------- #
# One full experiment: pipeline vs closed-form truth on a rollout set
# --------------------------------------------------------------------------- #


def run_experiment(tag: str, model, rolls_fisher, rolls_train, rolls_eval, cfg, damp):
    layer = model.get_submodule(cfg.layer_name)
    emb = model.emb.numpy()
    head = model.head.numpy()
    W = layer.weight.detach().numpy()
    d_out, d_in = W.shape
    dim = d_in * d_out
    device = torch.device("cpu")

    # ---- ground truth (all numpy) ----
    F_exact, A_exact, S_exact, n_tok_mine = exact_fisher(rolls_fisher, emb, head, W)
    Finv_true = np.linalg.inv(F_exact + damp * np.eye(dim))
    g_evals_true = [closed_form_grad(r, emb, head, W) for r in rolls_eval]
    s_train_true = [closed_form_grad(r, emb, head, W) for r in rolls_train]
    I_true = np.array([
        [-g.reshape(-1) @ Finv_true @ s.reshape(-1) for s in s_train_true]
        for g in g_evals_true
    ])  # (n_eval, n_train)

    # ---- project pipeline, end to end, with passive same-draw capture ----
    cap1 = PassiveCapture(layer)
    A_t, S_t, n_tok = accumulate_AS(model, layer, rolls_fisher, cfg, device=device)
    cap1.remove()
    Q_A, _, Q_S, _ = eigendecompose(A_t, S_t)
    cap2 = PassiveCapture(layer)
    Lam, _ = fit_lambda(model, layer, rolls_fisher, cfg, Q_A=Q_A, Q_S=Q_S, device=device)
    cap2.remove()

    g_evals_pipe = [
        compute_per_sample_grad(model, layer, r, cfg, device=device).double()
        for r in rolls_eval
    ]
    s_train_pipe = [
        compute_per_sample_grad(model, layer, r, cfg, device=device).double()
        for r in rolls_train
    ]
    I_pipe = np.empty_like(I_true)
    for a, g in enumerate(g_evals_pipe):
        p = inverse_hvp_additive(g, Q_A, Q_S, Lam, damping=damp)
        for b, s in enumerate(s_train_pipe):
            I_pipe[a, b] = influence_score_klrl(s, p)

    # ---- same-draw dense empirical Fisher (label-MC isolated) ----
    ms, ds = [], []
    for r, m, d in zip(rolls_fisher, cap2.ms, cap2.ds):
        P, T = r.prompt_len, r.prompt_len + r.response_len
        ms.append(m[0, P - 1 : T - 1].numpy())
        ds.append(d[0, P - 1 : T - 1].numpy())
    ms, ds = np.concatenate(ms), np.concatenate(ds)
    F_emp = np.zeros((dim, dim))
    for t in range(ms.shape[0]):
        v = np.kron(ds[t], ms[t])
        F_emp += np.outer(v, v)
    F_emp /= ms.shape[0]
    Finv_emp = np.linalg.inv(F_emp + damp * np.eye(dim))
    I_emp = np.array([
        [-g.reshape(-1) @ Finv_emp @ s.reshape(-1) for s in s_train_true]
        for g in g_evals_true
    ])

    # ---- gates and reports ----
    print(f"\n--- experiment [{tag}]  n_fisher={len(rolls_fisher)}  n_tok={n_tok} ---")
    gate(f"{tag}: token bookkeeping", n_tok == n_tok_mine,
         f"pipeline n_tok={n_tok}, closed-form n_tok={n_tok_mine}")

    eA = np.abs(A_t.numpy() - A_exact).max() / np.abs(A_exact).max()
    gate(f"{tag}: pipeline A == closed-form A (label-free, exact)", eA < 1e-12,
         f"max rel diff = {eA:.2e}")

    eS = np.linalg.norm(S_t.numpy() - S_exact) / np.linalg.norm(S_exact)
    info(f"{tag}: pipeline S vs exact label-expectation S",
         f"rel Frobenius = {eS:.4f} (pure pseudo-label MC, expect ~1/sqrt(n_tok))")

    worst = 0.0
    for g_p, g_t in zip(g_evals_pipe + s_train_pipe[:5], g_evals_true + s_train_true[:5]):
        rel = np.linalg.norm(g_p.numpy() - g_t) / np.linalg.norm(g_t)
        worst = max(worst, rel)
    gate(f"{tag}: compute_per_sample_grad == closed-form gradient", worst < 1e-5,
         f"worst rel err = {worst:.2e} (fp32 CE cast limits this)")

    rms = np.sqrt((I_true**2).mean())
    e_total = np.sqrt(((I_pipe - I_true) ** 2).mean()) / rms
    e_struct = np.sqrt(((I_pipe - I_emp) ** 2).mean()) / rms
    e_mc = np.sqrt(((I_emp - I_true) ** 2).mean()) / rms
    from scipy.stats import spearmanr
    rho = np.mean([spearmanr(I_pipe[a], I_true[a]).statistic for a in range(I_true.shape[0])])
    pear = np.mean([np.corrcoef(I_pipe[a], I_true[a])[0, 1] for a in range(I_true.shape[0])])
    info(f"{tag}: END-TO-END influence vs brute-force truth",
         f"rel RMS err total = {e_total:.4f} "
         f"[structural = {e_struct:.4f}, label-MC = {e_mc:.4f}], "
         f"Spearman = {rho:.4f}, Pearson = {pear:.4f}")
    return {"e_total": e_total, "e_struct": e_struct, "e_mc": e_mc, "spearman": rho}


def main() -> int:
    print("AUDIT 3: end-to-end known-answer (flat softmax model, closed-form truth)")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    vocab, d_in, d_out = 12, 6, 5
    cfg = EKFACConfig(layer_name="transformer.h.0.mlp.c_proj")
    damp = cfg.toy_damping
    torch.manual_seed(0)

    model = FlatSoftmaxModel(vocab, d_in, d_out, seed=11)
    model.eval()

    # ----- Variant 1: constant context (F exactly Kronecker) -----
    print("\n===== Variant 1: constant-context (Kronecker-exact Fisher) =====")
    rolls_train = build_rollouts_v1(100, vocab, ctx_token=3, seed=500)
    rolls_eval = build_rollouts_v1(3, vocab, ctx_token=3, seed=501)
    r_small = run_experiment(
        "v1 n=400", model, build_rollouts_v1(400, vocab, 3, seed=1),
        rolls_train, rolls_eval, cfg, damp,
    )
    r_big = run_experiment(
        "v1 n=4000", model, build_rollouts_v1(4000, vocab, 3, seed=2),
        rolls_train, rolls_eval, cfg, damp,
    )
    gate("v1 convergence: total err shrinks with 10x samples",
         r_big["e_total"] < r_small["e_total"],
         f"{r_small['e_total']:.4f} -> {r_big['e_total']:.4f}")
    gate("v1 final accuracy: pipeline reproduces brute-force I at n=4000",
         r_big["e_total"] < 0.05 and r_big["spearman"] > 0.99,
         f"rel RMS err = {r_big['e_total']:.4f}, Spearman = {r_big['spearman']:.4f}")

    # ----- Variant 2: random tokens (genuinely non-Kronecker Fisher) -----
    print("\n===== Variant 2: random tokens (non-Kronecker Fisher) =====")
    rolls_train2 = build_rollouts_v2(100, vocab, seed=600)
    rolls_eval2 = build_rollouts_v2(3, vocab, seed=601)
    r2_small = run_experiment(
        "v2 n=400", model, build_rollouts_v2(400, vocab, seed=3),
        rolls_train2, rolls_eval2, cfg, damp,
    )
    r2_big = run_experiment(
        "v2 n=4000", model, build_rollouts_v2(4000, vocab, seed=4),
        rolls_train2, rolls_eval2, cfg, damp,
    )
    info("v2 structural error persistence",
         f"structural component {r2_small['e_struct']:.4f} -> {r2_big['e_struct']:.4f} "
         f"(this is EK-FAC's irreducible approximation on non-Kronecker data)")
    gate("v2 sanity: label-MC component shrinks with 10x samples",
         r2_big["e_mc"] < r2_small["e_mc"],
         f"{r2_small['e_mc']:.4f} -> {r2_big['e_mc']:.4f}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY ===")
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"hard gates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
