"""INDEPENDENT AUDIT 3b — pin down the pseudo-label off-by-one in Stage 1A/1B.

Suspicion (from audit 3): in factors._accumulate_one / eigen._fit_lambda_one,
``pseudo = sample_pseudo_labels(logits[0])`` draws pseudo[u] from
softmax(logits[u]); after the shift-by-one CE alignment, loss row t is paired
with label pseudo[t+1] ~ softmax(logits[t+1]). The Fisher definition requires
row t's label to be drawn from row t's OWN conditional softmax(logits[t]).
Net effect: every pseudo label comes from the NEXT position's conditional.

Three nails:

N1 (flat model, closed-form biased limit): with mismatched labels l ~ q_next,
    E[(q_row - e_l)(q_row - e_l)^T]
      = diag(q_next) + q_row q_row^T - q_row q_next^T - q_next q_row^T
    which differs from the correct diag(q_row) - q_row q_row^T. Predict the
    biased S limit in closed form and show the pipeline's S converges to the
    BIASED limit (within MC error) and NOT to the correct one.

N2 (real toy transformer): my own hooks + my own label sampling, two
    estimators on identical rollouts — (a) row-aligned labels from
    softmax(logits[t]), (b) next-row labels replicating the pipeline.
    The pipeline's S must statistically match (b) and differ from (a).

N3 (damage estimate at toy scale): run the full pipeline vs a corrected
    pipeline (row-aligned sampling, my own implementation of Stage 1A/1B
    feeding the project's eigendecompose/inverse), compare influence scores:
    relative RMS diff + Spearman. Toy-scale only; real-scale damage needs a
    factor re-run on GPT-Neo (see audit 5).

Run:  python3 tests/audit_3b_pseudolabel_offbyone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda, inverse_hvp_additive  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402

from audit_3_known_answer_e2e import (  # noqa: E402  (reuse my own reference tools)
    FlatSoftmaxModel,
    build_rollouts_v1,
    closed_form_grad,
    exact_fisher,
    softmax_np,
)

RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def info(name: str, detail: str) -> None:
    print(f"  [info] {name}: {detail}")


# --------------------------------------------------------------------------- #
# N1: flat model, closed-form biased limit
# --------------------------------------------------------------------------- #


def n1_flat_biased_limit() -> None:
    print("\n=== N1: pipeline S converges to the predicted BIASED limit (flat model) ===")
    vocab, d_in, d_out = 12, 6, 5
    cfg = EKFACConfig(layer_name="transformer.h.0.mlp.c_proj")
    torch.manual_seed(0)
    model = FlatSoftmaxModel(vocab, d_in, d_out, seed=11)
    model.eval()
    layer = model.get_submodule(cfg.layer_name)
    emb, head = model.emb.numpy(), model.head.numpy()
    W = layer.weight.detach().numpy()
    ctx = 3

    # v1 rollouts: prompt [c, c], response [y], y uniform over vocab.
    # Single loss row t = 1: q_row = p(ctx) fixed; label drawn (by the bug)
    # from softmax(logits[2]) where position 2 holds the response token y.
    n = 20000
    rolls = build_rollouts_v1(n, vocab, ctx_token=ctx, seed=21)
    q_row = softmax_np(head @ (W @ emb[ctx]))

    # Correct limit and biased limit, both closed form. y ~ uniform(vocab)
    # by construction of the rollouts (NOT model-dependent).
    S_correct = head.T @ (np.diag(q_row) - np.outer(q_row, q_row)) @ head
    S_biased = np.zeros((d_out, d_out))
    for y in range(vocab):
        q_next = softmax_np(head @ (W @ emb[y]))
        M = (
            np.diag(q_next)
            + np.outer(q_row, q_row)
            - np.outer(q_row, q_next)
            - np.outer(q_next, q_row)
        )
        S_biased += head.T @ M @ head
    S_biased /= vocab

    A_t, S_t, n_tok = accumulate_AS(model, layer, rolls, cfg, device=torch.device("cpu"))
    S_pipe = S_t.numpy()
    d_correct = np.linalg.norm(S_pipe - S_correct) / np.linalg.norm(S_correct)
    d_biased = np.linalg.norm(S_pipe - S_biased) / np.linalg.norm(S_biased)
    sep = np.linalg.norm(S_biased - S_correct) / np.linalg.norm(S_correct)
    info("N1 distances", f"||S_pipe - S_correct|| rel = {d_correct:.4f}, "
         f"||S_pipe - S_biased|| rel = {d_biased:.4f}, "
         f"||S_biased - S_correct|| rel = {sep:.4f}, n_tok = {n_tok}")
    gate("N1 pipeline S matches the predicted off-by-one biased limit",
         d_biased < 0.05 * sep and d_correct > 0.5 * sep,
         f"pipeline sits at the biased limit ({d_biased:.4f}), far from correct ({d_correct:.4f})")


# --------------------------------------------------------------------------- #
# N2: real toy transformer, my own two estimators vs the pipeline
# --------------------------------------------------------------------------- #


def my_S_estimator(model, layer, rollouts, *, aligned: bool, seed: int) -> np.ndarray:
    """My own Stage-1A S estimator with my own hooks and label sampling.

    aligned=True : row t's label ~ softmax(logits[t])   (correct Fisher draw)
    aligned=False: row t's label ~ softmax(logits[t+1]) (replicates pipeline)
    """
    g = torch.Generator().manual_seed(seed)
    captured: list[torch.Tensor] = []
    h = layer.register_full_backward_hook(
        lambda _m, _gi, go: captured.append(go[0].detach().clone().double())
    )
    S = np.zeros((layer.weight.shape[0], layer.weight.shape[0]))
    n_tok = 0
    try:
        for r in rollouts:
            ids = torch.cat([r.prompt_ids, r.response_ids]).unsqueeze(0)
            P, T = r.prompt_len, r.prompt_len + r.response_len
            logits = model(ids)[0]
            probs = torch.softmax(logits.double(), dim=-1)
            # label for loss row t in [P-1, T-2]
            rows = torch.arange(P - 1, T - 1)
            src = rows if aligned else rows + 1
            labels_rows = torch.stack(
                [torch.multinomial(probs[u], 1, generator=g).squeeze() for u in src]
            )
            loss = TF.cross_entropy(logits[rows].float(), labels_rows, reduction="sum")
            for p_ in model.parameters():
                p_.grad = None
            captured.clear()
            loss.backward()
            d_resp = captured[0][0, P - 1 : T - 1]
            S += (d_resp.T @ d_resp).numpy()
            n_tok += T - P
    finally:
        h.remove()
    return S / n_tok


def n2_toy_transformer() -> None:
    print("\n=== N2: real toy transformer — aligned vs next-row label sampling ===")
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer()
    model.eval()
    model.double()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    g = torch.Generator().manual_seed(77)
    rolls = []
    for _ in range(300):
        T = model.max_len
        P = int(torch.randint(1, T, (1,), generator=g).item())
        ids = torch.randint(0, model.vocab_size, (T,), generator=g)
        rolls.append(Rollout(prompt_ids=ids[:P], response_ids=ids[P:], reward=0.0, step=0))

    # Average the PIPELINE's S over several seeds too (it is one MC draw per
    # seed), so all three objects compared are seed-averaged means and the
    # residual MC noise is ~spread/sqrt(8).
    n_seeds = 8
    S_pipes = []
    for k in range(n_seeds):
        cfg_k = EKFACConfig(seed=3000 + k)
        _, S_t, _ = accumulate_AS(model, layer, rolls, cfg_k, device=torch.device("cpu"))
        S_pipes.append(S_t.numpy())
    S_pipe = np.mean(S_pipes, axis=0)

    def mean_and_spread(aligned: bool):
        Ss = [my_S_estimator(model, layer, rolls, aligned=aligned, seed=1000 + k)
              for k in range(n_seeds)]
        M = np.mean(Ss, axis=0)
        spread = np.mean([np.linalg.norm(s - M) for s in Ss]) / np.linalg.norm(M)
        return M, spread

    S_aligned, sp_a = mean_and_spread(True)
    S_next, sp_n = mean_and_spread(False)
    d_a = np.linalg.norm(S_pipe - S_aligned) / np.linalg.norm(S_aligned)
    d_n = np.linalg.norm(S_pipe - S_next) / np.linalg.norm(S_next)
    sep = np.linalg.norm(S_aligned - S_next) / np.linalg.norm(S_aligned)
    noise = sp_a / np.sqrt(n_seeds)
    info("N2 distances (all seed-averaged means)",
         f"||S_pipe - S_aligned|| rel = {d_a:.4f}, "
         f"||S_pipe - S_nextrow|| rel = {d_n:.4f}, separation = {sep:.4f}, "
         f"residual MC noise ~ {noise:.4f}")
    gate("N2 pipeline S matches next-row sampling, not row-aligned sampling",
         d_n < 3 * noise + 0.02 and d_a > 0.5 * sep,
         f"next-row dist {d_n:.4f} ~ noise; aligned dist {d_a:.4f} ~ separation {sep:.4f}")


# --------------------------------------------------------------------------- #
# N3: toy-scale damage estimate on final influence scores
# --------------------------------------------------------------------------- #


def corrected_AS_and_lambda(model, layer, rollouts, *, seed: int):
    """My corrected Stage 1A + 1B: row-aligned pseudo labels, same per-token
    normalisation, feeding the PROJECT's eigendecompose. fp64."""
    d_out, d_in = layer.weight.shape
    g = torch.Generator().manual_seed(seed)
    cap_m: list[torch.Tensor] = []
    cap_d: list[torch.Tensor] = []
    h1 = layer.register_forward_hook(
        lambda _m, args, _o: cap_m.append(args[0].detach().clone().double())
    )
    h2 = layer.register_full_backward_hook(
        lambda _m, _gi, go: cap_d.append(go[0].detach().clone().double())
    )
    A = torch.zeros(d_in, d_in, dtype=torch.float64)
    S = torch.zeros(d_out, d_out, dtype=torch.float64)
    pairs = []
    n_tok = 0
    try:
        for r in rollouts:
            ids = torch.cat([r.prompt_ids, r.response_ids]).unsqueeze(0)
            P, T = r.prompt_len, r.prompt_len + r.response_len
            cap_m.clear(); cap_d.clear()      # clear BEFORE forward fires the hook
            logits = model(ids)[0]
            probs = torch.softmax(logits.double(), dim=-1)
            rows = torch.arange(P - 1, T - 1)
            labels_rows = torch.stack(
                [torch.multinomial(probs[u], 1, generator=g).squeeze() for u in rows]
            )
            loss = TF.cross_entropy(logits[rows].float(), labels_rows, reduction="sum")
            for p_ in model.parameters():
                p_.grad = None
            loss.backward()
            m_resp = cap_m[0][0, P - 1 : T - 1]
            d_resp = cap_d[0][0, P - 1 : T - 1]
            A += m_resp.T @ m_resp
            S += d_resp.T @ d_resp
            pairs.append((m_resp, d_resp))
            n_tok += T - P
    finally:
        h1.remove(); h2.remove()
    A, S = A / n_tok, S / n_tok
    A = 0.5 * (A + A.T); S = 0.5 * (S + S.T)
    Q_A, _, Q_S, _ = eigendecompose(A, S)
    # second pass for Lambda would need fresh draws; reuse captured pairs with
    # fresh aligned draws for honesty: redraw labels, redo backward
    Lam = torch.zeros(d_in, d_out, dtype=torch.float64)
    g2 = torch.Generator().manual_seed(seed + 1)
    cap_m2: list[torch.Tensor] = []
    cap_d2: list[torch.Tensor] = []
    h1 = layer.register_forward_hook(
        lambda _m, args, _o: cap_m2.append(args[0].detach().clone().double())
    )
    h2 = layer.register_full_backward_hook(
        lambda _m, _gi, go: cap_d2.append(go[0].detach().clone().double())
    )
    try:
        for r in rollouts:
            ids = torch.cat([r.prompt_ids, r.response_ids]).unsqueeze(0)
            P, T = r.prompt_len, r.prompt_len + r.response_len
            cap_m2.clear(); cap_d2.clear()    # clear BEFORE forward fires the hook
            logits = model(ids)[0]
            probs = torch.softmax(logits.double(), dim=-1)
            rows = torch.arange(P - 1, T - 1)
            labels_rows = torch.stack(
                [torch.multinomial(probs[u], 1, generator=g2).squeeze() for u in rows]
            )
            loss = TF.cross_entropy(logits[rows].float(), labels_rows, reduction="sum")
            for p_ in model.parameters():
                p_.grad = None
            loss.backward()
            m_resp = cap_m2[0][0, P - 1 : T - 1]
            d_resp = cap_d2[0][0, P - 1 : T - 1]
            a_proj = m_resp @ Q_A
            d_proj = d_resp @ Q_S
            Lam += a_proj.pow(2).T @ d_proj.pow(2)
    finally:
        h1.remove(); h2.remove()
    return Q_A, Q_S, Lam / n_tok


def n3_damage_estimate() -> None:
    print("\n=== N3: toy-scale damage to final influence scores ===")
    cfg = EKFACConfig()
    torch.manual_seed(cfg.seed)
    model = ToyTransformer()
    model.eval()
    model.double()
    layer = model.get_submodule("transformer.h.1.mlp.c_proj")
    g = torch.Generator().manual_seed(88)
    rolls = []
    for _ in range(400):
        T = model.max_len
        P = int(torch.randint(1, T, (1,), generator=g).item())
        ids = torch.randint(0, model.vocab_size, (T,), generator=g)
        rolls.append(Rollout(prompt_ids=ids[:P], response_ids=ids[P:], reward=0.0, step=0))
    device = torch.device("cpu")
    damp = cfg.toy_damping

    # Pipeline as-is (off-by-one labels).
    A_t, S_t, _ = accumulate_AS(model, layer, rolls, cfg, device=device)
    Q_A0, _, Q_S0, _ = eigendecompose(A_t, S_t)
    Lam0, _ = fit_lambda(model, layer, rolls, cfg, Q_A=Q_A0, Q_S=Q_S0, device=device)

    # Corrected (row-aligned labels), my implementation.
    Q_A1, Q_S1, Lam1 = corrected_AS_and_lambda(model, layer, rolls, seed=cfg.seed)

    g_eval = compute_per_sample_grad(model, layer, rolls[0], cfg, device=device).double()
    s_list = [compute_per_sample_grad(model, layer, r, cfg, device=device).double()
              for r in rolls[1:151]]
    p0 = inverse_hvp_additive(g_eval, Q_A0, Q_S0, Lam0, damping=damp)
    p1 = inverse_hvp_additive(g_eval, Q_A1, Q_S1, Lam1, damping=damp)
    I0 = np.array([influence_score_klrl(s, p0) for s in s_list])
    I1 = np.array([influence_score_klrl(s, p1) for s in s_list])
    from scipy.stats import spearmanr
    rel = np.linalg.norm(I0 - I1) / np.linalg.norm(I1)
    rho = spearmanr(I0, I1).statistic
    info("N3 buggy vs corrected influence (toy scale)",
         f"rel RMS diff = {rel:.4f}, Spearman = {rho:.4f} over {len(I0)} rollouts "
         f"(toy only; real-model damage needs a GPT-Neo factor re-run)")


def main() -> int:
    print("AUDIT 3b: pseudo-label off-by-one — mechanism + damage")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    n1_flat_biased_limit()
    n2_toy_transformer()
    n3_damage_estimate()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY ===")
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"gates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed "
          f"(here PASS = bug mechanism CONFIRMED)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
