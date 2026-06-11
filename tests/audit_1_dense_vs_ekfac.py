"""INDEPENDENT AUDIT 1 — dense brute-force Fisher inverse vs EK-FAC.

Written as an external review of the EK-FAC pipeline (src/ekfac/). The point
is NOT to re-run existing tests: every ground-truth quantity here is computed
from scratch with numpy (np.kron / np.linalg.inv / np.linalg.eigh) and with
independently re-derived conventions (vec ordering, response-window slicing,
Lambda layout). Project code is treated as a black box under test.

Rung A — pure linear-algebra oracle (no model):
  synthetic per-token samples (m, delta) with strong cross-correlation so the
  true Fisher is NOT Kronecker. Validates, at machine precision:
    A1. project eigendecompose output diagonalises my A_emp/S_emp
    A2. my dense diag(U^T F_emp U) layout identity (self-consistency)
    A3. inverse_hvp_additive == numpy oracle U diag(1/(lam+damp)) U^T
    A4. reconstruct_F_inv_dense == same oracle, dense form
    A5. two-level damping probe: eigen-direction in == 1/max(L+a*mean, floor) out
    A6. projection round-trip g -> eigenbasis -> back == identity
  plus informational: EK-FAC inverse vs np.linalg.inv(F_emp + damp I) on
  correlated data (approximation error visible) and on independent data
  (should be near-exact in population).

Rung B — real toy transformer, same-draw dense comparison:
  run the project pipeline (accumulate_AS -> eigendecompose -> fit_lambda)
  with my OWN passive hooks capturing every (m, delta) the pipeline actually
  used. From those captures, with my own window slicing, build:
    B1. A_mine/S_mine == project A/S          (machine precision; same draws)
    B2. dense diag(U^T F_emp U) == project Lambda (machine precision; same draws)
    B3. hook semantics: sum_t delta_t outer m_t over ALL positions == autograd
        grad of layer.weight (machine precision)
    B4. informational: np.linalg.inv(F_emp + damp I) vs EK-FAC reconstructed
        inverse — Frobenius rel err + action rel err + off-diagonal Kronecker
        mass. This is the "is the approximation in an explainable range" cut.

Run:  python3 tests/audit_1_dense_vs_ekfac.py
Exit: 0 if all hard gates pass, 1 otherwise. Informational numbers always print.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import (  # noqa: E402
    eigendecompose,
    fit_lambda,
    inverse_hvp,
    inverse_hvp_additive,
    reconstruct_F_inv_dense,
)
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def info(name: str, detail: str) -> None:
    print(f"  [info] {name}: {detail}")


# --------------------------------------------------------------------------- #
# My own (independent) reference math, numpy only.
# Convention derived from first principles:
#   layer weight W has shape (d_out, d_in); per-token gradient contribution is
#   g_t = outer(delta_t, m_t), shape (d_out, d_in). Row-major flatten of g_t
#   gives vec(g_t) = np.kron(delta_t, m_t). Hence the dense token-level Fisher
#   F = E_t[vec(g_t) vec(g_t)^T] and the Kronecker model is S (x) A with
#   U = np.kron(Q_S, Q_A): column index j*d_in + i <-> eigvec kron(qS_j, qA_i),
#   whose fitted eigenvalue is Lambda[i, j] = E[(qA_i.m)^2 (qS_j.delta)^2].
# --------------------------------------------------------------------------- #


def dense_fisher_from_pairs(ms: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """F_emp = mean_t kron(d_t, m_t) kron(d_t, m_t)^T, fp64."""
    n = ms.shape[0]
    dim = ms.shape[1] * ds.shape[1]
    F = np.zeros((dim, dim), dtype=np.float64)
    for t in range(n):
        v = np.kron(ds[t], ms[t])
        F += np.outer(v, v)
    return F / n


def my_lambda_from_dense(F_emp: np.ndarray, Q_A: np.ndarray, Q_S: np.ndarray) -> np.ndarray:
    """diag(U^T F U) re-laid-out to (d_in, d_out) per MY index derivation."""
    d_in = Q_A.shape[0]
    d_out = Q_S.shape[0]
    U = np.kron(Q_S, Q_A)
    diag = np.einsum("ki,kl,li->i", U, F_emp, U)  # diag of U^T F U
    # flat index k = j*d_in + i  ->  Lambda[i, j]
    return diag.reshape(d_out, d_in).T.copy()


def my_ekfac_inverse_dense(
    Q_A: np.ndarray, Q_S: np.ndarray, Lam: np.ndarray, damp: float
) -> np.ndarray:
    """Oracle dense EK-FAC inverse from my own layout derivation."""
    d_in, d_out = Lam.shape
    U = np.kron(Q_S, Q_A)
    lam_flat = Lam.T.reshape(-1)  # index j*d_in + i -> Lam[i, j]
    return U @ np.diag(1.0 / (lam_flat + damp)) @ U.T


def vec_row(g: np.ndarray) -> np.ndarray:
    """Row-major flatten of (d_out, d_in)."""
    return g.reshape(-1)


# --------------------------------------------------------------------------- #
# Rung A
# --------------------------------------------------------------------------- #


def rung_A() -> None:
    print("\n=== Rung A: pure linear-algebra oracle (synthetic, correlated) ===")
    rng = np.random.default_rng(7)
    d_in, d_out, n = 7, 5, 6000
    damp = 0.01

    # Correlated, non-Gaussian samples: delta depends nonlinearly on m so the
    # population Fisher is genuinely non-Kronecker.
    B = rng.standard_normal((d_out, d_in))
    ms = rng.standard_normal((n, d_in)) @ rng.standard_normal((d_in, d_in)) * 0.7
    ds = np.tanh(ms @ B.T) + 0.3 * rng.standard_normal((n, d_out)) + 0.2 * (ms[:, :d_out] ** 2)

    A_emp = ms.T @ ms / n
    S_emp = ds.T @ ds / n
    F_emp = dense_fisher_from_pairs(ms, ds)

    # Project eigendecompose as black box.
    Q_A_t, lam_A_t, Q_S_t, lam_S_t = eigendecompose(
        torch.from_numpy(A_emp), torch.from_numpy(S_emp)
    )
    Q_A = Q_A_t.numpy()
    Q_S = Q_S_t.numpy()

    # A1: their eigenpairs diagonalise my matrices.
    rec_A = Q_A @ np.diag(lam_A_t.numpy()) @ Q_A.T
    rec_S = Q_S @ np.diag(lam_S_t.numpy()) @ Q_S.T
    eA = np.linalg.norm(rec_A - A_emp) / np.linalg.norm(A_emp)
    eS = np.linalg.norm(rec_S - S_emp) / np.linalg.norm(S_emp)
    gate("A1 eigendecompose reconstructs A,S", eA < 1e-12 and eS < 1e-12,
         f"rel err A={eA:.2e}, S={eS:.2e}")

    # A2: my Lambda layout self-consistency: dense diag vs direct formula.
    Lam_dense = my_lambda_from_dense(F_emp, Q_A, Q_S)
    a_proj = ms @ Q_A          # (n, d_in)
    d_proj = ds @ Q_S          # (n, d_out)
    Lam_direct = (a_proj**2).T @ (d_proj**2) / n
    e = np.abs(Lam_dense - Lam_direct).max() / Lam_direct.max()
    gate("A2 Lambda layout identity (dense diag == direct formula)", e < 1e-10,
         f"max rel diff = {e:.2e}")

    # A3: project inverse_hvp_additive vs my numpy oracle on random g.
    Lam_t = torch.from_numpy(Lam_direct)
    F_inv_mine = my_ekfac_inverse_dense(Q_A, Q_S, Lam_direct, damp)
    ok3, worst3 = True, 0.0
    for k in range(5):
        g = rng.standard_normal((d_out, d_in))
        x_theirs = inverse_hvp_additive(
            torch.from_numpy(g), Q_A_t, Q_S_t, Lam_t, damping=damp
        ).numpy()
        x_mine = (F_inv_mine @ vec_row(g)).reshape(d_out, d_in)
        rel = np.linalg.norm(x_theirs - x_mine) / np.linalg.norm(x_mine)
        worst3 = max(worst3, rel)
        ok3 &= rel < 1e-10
    gate("A3 inverse_hvp_additive == numpy oracle", ok3, f"worst rel err = {worst3:.2e}")

    # A4: their dense reconstruction vs my oracle matrix.
    F_inv_theirs = reconstruct_F_inv_dense(Q_A_t, Q_S_t, Lam_t, damping=damp).numpy()
    e4 = np.linalg.norm(F_inv_theirs - F_inv_mine) / np.linalg.norm(F_inv_mine)
    gate("A4 reconstruct_F_inv_dense == numpy oracle", e4 < 1e-10, f"rel err = {e4:.2e}")

    # A5: two-level damping probe. Eigen-direction g = outer(qS_j, qA_i) must
    # come back scaled by exactly 1/max(Lam[i,j] + alpha*mean(Lam), floor).
    # A double (or missing) damping application fails this loudly.
    alpha, floor = 0.1, 1e-5
    lam_mean = Lam_direct.mean()
    # Force some entries below the floor to exercise the clamp branch.
    Lam_floor = Lam_direct.copy()
    Lam_floor[-1, -1] = 0.0
    Lam_floor[-2, -1] = 1e-9
    Lam_floor_t = torch.from_numpy(Lam_floor)
    lam_mean_f = Lam_floor.mean()
    ok5, worst5 = True, 0.0
    probes = [(0, 0), (d_in - 1, d_out - 1), (d_in - 2, d_out - 1), (2, 3)]
    for (i, j) in probes:
        g = np.outer(Q_S[:, j], Q_A[:, i])
        x = inverse_hvp(
            torch.from_numpy(g), Q_A_t, Q_S_t, Lam_floor_t,
            damping_floor=floor, damping_alpha=alpha,
        ).numpy()
        denom_hand = max(Lam_floor[i, j] + alpha * lam_mean_f, floor)
        rel = np.linalg.norm(x - g / denom_hand) / np.linalg.norm(g / denom_hand)
        worst5 = max(worst5, rel)
        ok5 &= rel < 1e-10
    gate("A5 two-level damping applied exactly once (incl. floor branch)", ok5,
         f"worst rel err = {worst5:.2e} over probes {probes}")

    # A6: projection round-trip is the identity.
    g = rng.standard_normal((d_out, d_in))
    g_rt = Q_S @ (Q_S.T @ g @ Q_A) @ Q_A.T
    e6 = np.linalg.norm(g_rt - g) / np.linalg.norm(g)
    gate("A6 eigenbasis round-trip identity", e6 < 1e-12, f"rel err = {e6:.2e}")

    # Informational: EK-FAC inverse vs brute-force dense inverse.
    F_dense_inv = np.linalg.inv(F_emp + damp * np.eye(d_in * d_out))
    e_frob = np.linalg.norm(F_inv_mine - F_dense_inv) / np.linalg.norm(F_dense_inv)
    U = np.kron(Q_S, Q_A)
    Fk = U.T @ F_emp @ U
    off = np.linalg.norm(Fk - np.diag(np.diag(Fk))) / np.linalg.norm(Fk)
    info("A-corr: EK-FAC inv vs np.linalg.inv (correlated data)",
         f"Frobenius rel err = {e_frob:.3f}, off-diag Kronecker mass = {off:.3f}")

    # Control: independent m, delta -> population Fisher IS Kronecker; EK-FAC
    # should approach the dense inverse (residual = finite-sample only).
    msI = rng.standard_normal((n, d_in)) @ rng.standard_normal((d_in, d_in)) * 0.5
    dsI = rng.standard_normal((n, d_out)) @ rng.standard_normal((d_out, d_out)) * 0.5
    A_I, S_I = msI.T @ msI / n, dsI.T @ dsI / n
    F_I = dense_fisher_from_pairs(msI, dsI)
    QA_t, _, QS_t, _ = eigendecompose(torch.from_numpy(A_I), torch.from_numpy(S_I))
    LamI = my_lambda_from_dense(F_I, QA_t.numpy(), QS_t.numpy())
    FinvI_ek = my_ekfac_inverse_dense(QA_t.numpy(), QS_t.numpy(), LamI, damp)
    FinvI_dense = np.linalg.inv(F_I + damp * np.eye(d_in * d_out))
    eI = np.linalg.norm(FinvI_ek - FinvI_dense) / np.linalg.norm(FinvI_dense)
    UI = np.kron(QS_t.numpy(), QA_t.numpy())
    FkI = UI.T @ F_I @ UI
    offI = np.linalg.norm(FkI - np.diag(np.diag(FkI))) / np.linalg.norm(FkI)
    info("A-indep control: EK-FAC inv vs dense inv (independent data)",
         f"Frobenius rel err = {eI:.4f}, off-diag mass = {offI:.4f} "
         f"(should be small; residual is finite-sample noise)")


# --------------------------------------------------------------------------- #
# Rung B — real toy transformer, passive same-draw capture
# --------------------------------------------------------------------------- #


class PassiveCapture:
    """My own hooks (independent of ekfac.hooks): record every (m, delta) pair
    the pipeline's forward/backward passes actually produce on `layer`."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.ms: list[torch.Tensor] = []
        self.ds: list[torch.Tensor] = []
        self._h1 = layer.register_forward_hook(self._fwd)
        self._h2 = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _mod, args, _out) -> None:
        self.ms.append(args[0].detach().clone().double())

    def _bwd(self, _mod, _gi, go) -> None:
        self.ds.append(go[0].detach().clone().double())

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()


def build_rollouts(n: int, vocab: int, max_len: int, seed: int) -> list[Rollout]:
    """My own rollout builder (mirrors the shapes the pipeline expects)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        T = max_len
        P = int(torch.randint(1, T, (1,), generator=g).item())  # 1..T-1, R>=1
        ids = torch.randint(0, vocab, (T,), generator=g, dtype=torch.long)
        out.append(Rollout(prompt_ids=ids[:P], response_ids=ids[P:], reward=0.0, step=0))
    return out


def rung_B() -> None:
    print("\n=== Rung B: real toy transformer, same-draw dense comparison ===")
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer()
    model.eval()
    model.double()  # fp64 end to end so machine-precision gates are meaningful
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    d_out, d_in = layer.weight.shape
    device = torch.device("cpu")
    rollouts = build_rollouts(200, model.vocab_size, model.max_len, seed=123)

    # --- B3 first: hook-semantics identity on one rollout, MY code only ---
    r0 = rollouts[0]
    cap = PassiveCapture(layer)
    ids = torch.cat([r0.prompt_ids, r0.response_ids]).unsqueeze(0)
    T = ids.shape[1]
    P = r0.prompt_len
    labels = torch.full((T,), -100, dtype=torch.long)
    labels[P:] = ids[0, P:]
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[0, :-1], labels[1:], ignore_index=-100, reduction="sum"
    )
    layer.weight.grad = None
    loss.backward()
    cap.remove()
    m_full, d_full = cap.ms[0][0], cap.ds[0][0]  # (T, d_in), (T, d_out)
    grad_rebuilt = torch.einsum("ti,tj->ij", d_full, m_full)  # sum over ALL positions
    e3 = (grad_rebuilt - layer.weight.grad).abs().max().item() / layer.weight.grad.abs().max().item()
    gate("B3 hook semantics: sum_t delta x m over ALL positions == autograd grad",
         e3 < 1e-10, f"rel err = {e3:.2e}")

    # --- run project pipeline with passive capture ---
    cap1 = PassiveCapture(layer)
    A_t, S_t, n_tok = accumulate_AS(model, layer, rollouts, cfg, device=device)
    cap1.remove()
    Q_A_t, lam_A_t, Q_S_t, lam_S_t = eigendecompose(A_t, S_t)
    cap2 = PassiveCapture(layer)
    Lam_t, n_tok_lam = fit_lambda(
        model, layer, rollouts, cfg, Q_A=Q_A_t, Q_S=Q_S_t, device=device
    )
    cap2.remove()
    assert len(cap1.ms) == len(rollouts) and len(cap2.ms) == len(rollouts)

    # --- my own window slicing: loss rows are shift positions P-1 .. T-2 ---
    def windowed(caps: PassiveCapture) -> tuple[np.ndarray, np.ndarray]:
        ms, ds = [], []
        for r, m, d in zip(rollouts, caps.ms, caps.ds):
            P, T = r.prompt_len, r.prompt_len + r.response_len
            ms.append(m[0, P - 1 : T - 1].numpy())
            ds.append(d[0, P - 1 : T - 1].numpy())
        return np.concatenate(ms), np.concatenate(ds)

    ms1, ds1 = windowed(cap1)
    gate("B0 token count bookkeeping", ms1.shape[0] == n_tok == n_tok_lam,
         f"my window tokens = {ms1.shape[0]}, their n_tok = {n_tok}")

    # B1: A, S from my capture+slice == project A, S (same draws, label-free m).
    A_mine = ms1.T @ ms1 / ms1.shape[0]
    S_mine = ds1.T @ ds1 / ds1.shape[0]
    eA = np.abs(A_mine - A_t.numpy()).max() / np.abs(A_t.numpy()).max()
    eS = np.abs(S_mine - S_t.numpy()).max() / np.abs(S_t.numpy()).max()
    gate("B1 A,S == my dense accumulation (same draws)", eA < 1e-10 and eS < 1e-10,
         f"rel err A={eA:.2e}, S={eS:.2e}")

    # B2: project Lambda == diag(U^T F_emp U) from the SAME fit_lambda draws.
    ms2, ds2 = windowed(cap2)
    F_emp2 = dense_fisher_from_pairs(ms2, ds2)
    Lam_mine = my_lambda_from_dense(F_emp2, Q_A_t.numpy(), Q_S_t.numpy())
    e2 = np.abs(Lam_mine - Lam_t.numpy()).max() / Lam_t.numpy().max()
    gate("B2 Lambda == dense diag(U^T F_emp U) (same draws)", e2 < 1e-10,
         f"max rel diff = {e2:.2e}")

    # B4 informational: brute-force dense inverse vs EK-FAC inverse.
    damp = cfg.toy_damping
    dim = d_in * d_out
    F_pool = dense_fisher_from_pairs(
        np.concatenate([ms1, ms2]), np.concatenate([ds1, ds2])
    )
    for tag, F_target in [("fit_lambda draws", F_emp2), ("pooled both passes", F_pool)]:
        F_dense_inv = np.linalg.inv(F_target + damp * np.eye(dim))
        F_ek_inv = reconstruct_F_inv_dense(Q_A_t, Q_S_t, Lam_t, damping=damp).numpy()
        e_frob = np.linalg.norm(F_ek_inv - F_dense_inv) / np.linalg.norm(F_dense_inv)
        # Action error on random probe gradients (what scoring actually uses).
        rng = np.random.default_rng(0)
        rels = []
        for _ in range(20):
            g = rng.standard_normal(dim)
            rels.append(
                np.linalg.norm((F_ek_inv - F_dense_inv) @ g)
                / np.linalg.norm(F_dense_inv @ g)
            )
        U = np.kron(Q_S_t.numpy(), Q_A_t.numpy())
        Fk = U.T @ F_target @ U
        off = np.linalg.norm(Fk - np.diag(np.diag(Fk))) / np.linalg.norm(Fk)
        info(f"B4 dense-vs-EKFAC inverse [{tag}]",
             f"Frobenius rel err = {e_frob:.3f}, action rel err median = "
             f"{np.median(rels):.3f} / max = {np.max(rels):.3f}, off-diag mass = {off:.3f}")

    # B5 informational: influence-score level agreement, dense vs EK-FAC.
    # Random "eval" and "train" gradient pairs through both inverses.
    rng = np.random.default_rng(1)
    F_dense_inv = np.linalg.inv(F_emp2 + damp * np.eye(dim))
    F_ek_inv = reconstruct_F_inv_dense(Q_A_t, Q_S_t, Lam_t, damping=damp).numpy()
    n_pairs = 200
    I_dense, I_ek = np.empty(n_pairs), np.empty(n_pairs)
    g_eval = rng.standard_normal(dim)
    for k in range(n_pairs):
        s = rng.standard_normal(dim)
        I_dense[k] = -g_eval @ F_dense_inv @ s
        I_ek[k] = -g_eval @ F_ek_inv @ s
    from scipy.stats import spearmanr
    rho = spearmanr(I_dense, I_ek).statistic
    pear = np.corrcoef(I_dense, I_ek)[0, 1]
    info("B5 influence ranking, dense inverse vs EK-FAC inverse (random grads)",
         f"Spearman = {rho:.4f}, Pearson = {pear:.4f} over {n_pairs} pairs")


def main() -> int:
    print("AUDIT 1: dense brute-force Fisher inverse vs EK-FAC (independent oracle)")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    rung_A()
    rung_B()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY ===")
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"hard gates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
