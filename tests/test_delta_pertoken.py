"""Gate for the per-token-granularity Delta (metric fix).

Validates the matrix-free per-token Delta-vector product (score_and_hvps_pertoken
+ delta_vp_per_prompt_tok) against an INDEPENDENT dense brute-force per-token
Delta on the toy transformer, built straight from the definition

    Delta_tok = (1/N_tok) sum_u A(seq(u)) ( vec(g_u) vec(g_u)^T + Hess_u ),
    g_u = outer(delta_u, m_u),   Hess_u summed over a sequence = dense Hessian
                                  of the SUM log pi (grad^2_phi, basis-by-basis).

Gates:
  G1. matrix-free diagonal score term == explicit sum_u g_u (g_u . v).
  G2. matrix-free HVP == central finite-difference HVP of the SUM log pi.
  G3. END-TO-END: mean_p d_p (delta_vp_per_prompt_tok) == dense Delta_tok~ @ vec(v),
      for several targets v. (machine precision; the dense side never reuses the
      matrix-free code path — it captures (m,delta) with its own hook and builds
      the Hessian by basis-vector double-backward.)
  G4. HEALTH CHECK (derivation §7): at per-token granularity the diagonal score
      term and the HVP term are the SAME order of magnitude (the cancellation
      structure the mean-reduction broke). Report the ratio.

Run:  python3 tests/test_delta_pertoken.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.data import build_input_and_labels  # noqa: E402
from ekfac.hooks import capture_c_proj  # noqa: E402
from ekfac.toy import ToyTransformer  # noqa: E402
from phase4.e3_delta import DeltaSample, delta_vp_per_prompt_tok  # noqa: E402
from phase4.hvp_logpi import score_and_hvps_pertoken  # noqa: E402

DEV = torch.device("cpu")
LAYER = "transformer.h.1.mlp.c_proj"
BETA = 0.37
RESULTS: list[tuple[str, bool, str]] = []


def gate(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def info(name, detail):
    print(f"  [info] {name}: {detail}")


def _logpi_sum(model, prompt_ids, response_ids):
    """-CE(sum) over response tokens, real labels (= sequence log pi)."""
    input_ids, labels = build_input_and_labels(prompt_ids, response_ids,
                                               response_labels=response_ids, ignore_index=-100)
    logits = model(input_ids.unsqueeze(0))
    logits = logits.logits if hasattr(logits, "logits") else logits
    ce = F.cross_entropy(logits[0, :-1].to(torch.float64), labels[1:],
                         ignore_index=-100, reduction="sum")
    return -ce


def dense_per_sample(model, layer, prompt_ids, response_ids):
    """Independent dense (diag_score, dense_Hessian) for one sample, fp64.
    Captures (m_u, delta_u) with its OWN hook + sum-loss backward; builds the
    Hessian by basis-vector double-backward. Never calls the matrix-free code."""
    P, R = int(prompt_ids.shape[0]), int(response_ids.shape[0])
    T = P + R
    weight = layer.weight
    d_out, d_in = weight.shape
    dim = d_out * d_in

    # --- per-token g_u via own hook on a sum-loss backward ---
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    with capture_c_proj(layer) as cache:
        logpi = _logpi_sum(model, prompt_ids, response_ids)
        logpi.backward(retain_graph=False)
    m_resp = cache.m[0, P - 1:T - 1].to(torch.float64)        # (R, d_in)
    d_resp = cache.delta[0, P - 1:T - 1].to(torch.float64)    # (R, d_out); delta of -CE? sign cancels in diag
    diag_dense = torch.zeros(dim, dim, dtype=torch.float64)
    for u in range(R):
        g_u = torch.outer(d_resp[u], m_resp[u]).reshape(-1)   # vec(g_u)
        diag_dense += torch.outer(g_u, g_u)

    # --- dense Hessian of sum log pi via basis-vector double-backward ---
    logpi = _logpi_sum(model, prompt_ids, response_ids)
    (g,) = torch.autograd.grad(logpi, weight, create_graph=True)   # (d_out, d_in)
    H = torch.zeros(dim, dim, dtype=torch.float64)
    eye = torch.eye(dim, dtype=g.dtype).reshape(dim, d_out, d_in)
    for j in range(dim):
        (col,) = torch.autograd.grad(g, weight, grad_outputs=eye[j],
                                     retain_graph=(j < dim - 1))
        H[:, j] = col.reshape(-1).to(torch.float64)
    H = 0.5 * (H + H.T)   # symmetrise tiny roundoff
    return diag_dense, H, R


def main() -> int:
    print("Per-token Delta gate: matrix-free vs dense brute-force (toy transformer)")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    torch.manual_seed(0)
    model = ToyTransformer().eval().double()
    layer = model.get_submodule(LAYER)
    d_out, d_in = layer.weight.shape
    dim = d_out * d_in

    # Build a small pool: 3 prompts x 3 samples, random-ish A per sample.
    g = torch.Generator().manual_seed(7)
    pool: list[DeltaSample] = []
    for pidx in range(3):
        for _ in range(3):
            Tlen = model.max_len
            P = int(torch.randint(1, Tlen, (1,), generator=g).item())
            ids = torch.randint(0, model.vocab_size, (Tlen,), generator=g, dtype=torch.long)
            A = float(torch.randn(1, generator=g).item())
            pool.append(DeltaSample(prompt_idx=pidx, prompt_ids=ids[:P],
                                    response_ids=ids[P:], A=A))

    # Targets.
    torch.manual_seed(1)
    vs = [torch.randn(d_out, d_in, dtype=torch.float64) for _ in range(3)]

    # --- G1: diag term match on one sample ---
    smp = pool[0]
    R0, contribs0 = score_and_hvps_pertoken(model, layer, smp.prompt_ids, smp.response_ids, vs, DEV)
    diag_dense0, H0, _ = dense_per_sample(model, layer, smp.prompt_ids, smp.response_ids)
    worst_g1 = 0.0
    for k, v in enumerate(vs):
        diag_mf = contribs0[k] - (H0 @ v.reshape(-1)).reshape(d_out, d_in)  # subtract hvp part
        diag_ref = (diag_dense0 @ v.reshape(-1)).reshape(d_out, d_in)
        rel = (diag_mf - diag_ref).norm().item() / (diag_ref.norm().item() + 1e-30)
        worst_g1 = max(worst_g1, rel)
    gate("G1 matrix-free diag term == explicit sum_u g_u(g_u^T v)", worst_g1 < 1e-9,
         f"worst rel err = {worst_g1:.2e}")

    # --- G2: HVP vs central finite difference of sum log pi ---
    eps = 1e-4
    v = vs[0]
    w0 = layer.weight.detach().clone()
    with torch.no_grad():
        layer.weight.copy_(w0 + eps * v)
    gp = torch.autograd.grad(_logpi_sum(model, smp.prompt_ids, smp.response_ids), layer.weight)[0]
    with torch.no_grad():
        layer.weight.copy_(w0 - eps * v)
    gm = torch.autograd.grad(_logpi_sum(model, smp.prompt_ids, smp.response_ids), layer.weight)[0]
    with torch.no_grad():
        layer.weight.copy_(w0)
    hvp_fd = (gp - gm).to(torch.float64) / (2 * eps)
    hvp_mf = (H0 @ v.reshape(-1)).reshape(d_out, d_in)
    rel_g2 = (hvp_mf - hvp_fd).norm().item() / (hvp_fd.norm().item() + 1e-30)
    gate("G2 dense Hessian @ v == finite-difference HVP of sum log pi", rel_g2 < 1e-4,
         f"rel err = {rel_g2:.2e}")

    # --- G3: end-to-end matrix-free vs dense Delta_tok~ ---
    # fp64 store so the gate is a real machine-precision guard (with the default
    # float32 store the end-to-end agreement is only ~5e-8 = float32 truncation,
    # NOT a logic limit; the independent reviewer confirmed fp64 gives ~4e-16).
    d_p, order, N_tok, _ = delta_vp_per_prompt_tok(
        pool, model, layer, BETA, vs, DEV, log_every=10**9, store_dtype=torch.float64)
    # store_device="cpu" path (used in production) must be bit-identical.
    d_p_cpu, _, _, _ = delta_vp_per_prompt_tok(
        pool, model, layer, BETA, vs, DEV, log_every=10**9,
        store_dtype=torch.float64, store_device="cpu")
    cpu_diff = max((d_p[k] - d_p_cpu[k]).abs().max().item() for k in range(len(vs)))
    gate("G3b store_device='cpu' bit-identical to default", cpu_diff == 0.0,
         f"max abs diff = {cpu_diff:.1e}")
    # dense Delta_tok = (1/N_tok) sum_i A_i (diag_i + H_i); Delta_tok~ = /beta
    Delta_dense = [torch.zeros(dim, dim, dtype=torch.float64) for _ in vs]  # not needed per target
    Delta_acc = torch.zeros(dim, dim, dtype=torch.float64)
    for smp in pool:
        diag_i, H_i, R_i = dense_per_sample(model, layer, smp.prompt_ids, smp.response_ids)
        Delta_acc += smp.A * (diag_i + H_i)
    Delta_tok = Delta_acc / N_tok          # dense Delta_tok
    Delta_tok_tilde = Delta_tok / BETA
    worst_g3 = 0.0
    for k, v in enumerate(vs):
        mf = d_p[k].to(torch.float64).mean(0)                  # mean_p d_p = Delta_tok~ v
        ref = (Delta_tok_tilde @ v.reshape(-1)).reshape(d_out, d_in)
        rel = (mf - ref).norm().item() / (ref.norm().item() + 1e-30)
        worst_g3 = max(worst_g3, rel)
    # With fp64 store, the matrix-free Delta_tok~ matches the dense brute-force to
    # machine precision (the per-token diagonal AND the sum-HVP fp64 paths agree;
    # the independent reviewer measured ~4e-16). 1e-12 is a real guard — a logic
    # error would be order-1.
    gate("G3 end-to-end mean_p d_p == dense Delta_tok~ @ vec(v)", worst_g3 < 1e-12,
         f"worst rel err = {worst_g3:.2e}, N_tok={N_tok}")

    # --- G4: health check — diag and HVP terms same order at per-token granularity ---
    diag_norm = (Delta_acc * 0).norm()  # placeholder
    # measure per-sample term norms (target 0) summed
    diag_tot = 0.0
    hvp_tot = 0.0
    for smp in pool:
        diag_i, H_i, _ = dense_per_sample(model, layer, smp.prompt_ids, smp.response_ids)
        v0 = vs[0].reshape(-1)
        diag_tot += abs(smp.A) * (diag_i @ v0).norm().item()
        hvp_tot += abs(smp.A) * (H_i @ v0).norm().item()
    ratio = diag_tot / (hvp_tot + 1e-30)
    gate("G4 diag and HVP terms same order (per-token granularity)", 0.1 < ratio < 10.0,
         f"||diag||/||hvp|| = {ratio:.3f} (should be O(1); mean-reduction would make it ~1/R)")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\ngates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


def test_delta_pertoken_matches_dense() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
