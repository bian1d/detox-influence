"""Phase 5 Stage 1 — real-OLMo-2 HVP smoke + finite-difference check.

The gated toy validated the SwiGLU double-backward with *generic* attention, but
NOT OLMo-2's specific ops (rotary position embeddings, QK-norm, RMSNorm). This
is the only independent test that those ops compute the *second* derivative
w.r.t. phi = layer-12 down_proj.weight correctly. A wrong second derivative here
would be a silent error — the thermometer would look fine but be meaningless.

Two finite-difference checks per sample/direction, both independent of the
analytic HVP, the first independent of autograd's *backward* entirely:

  (A) Scalar curvature (FORWARD-ONLY):
        v^T H v  ~=  [ logpi(W+eps v) - 2 logpi(W) + logpi(W-eps v) ] / eps^2
      compared to  <HVP_analytic(v), v>.  Uses only forward passes, so it cannot
      be fooled by a backward-pass bug.
  (B) Vector HVP (central difference of the analytic gradient s):
        HVP(v)  ~=  [ s(W+eps v) - s(W-eps v) ] / (2 eps)
      compared to HVP_analytic(v).

We sweep eps and report the achievable minimum relative error per check (FD has
a roundoff floor; the min over eps is the honest "how well does it agree"). fp32
typically bottoms out ~1e-2..1e-3; a best-effort fp64 pass (CPU, one short
sample) tightens this if the model runs in double.

Run with the olmo env python:
    /root/miniconda3/envs/olmo/bin/python tests/diag_olmo_hvp_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase4.grad_phi import get_phi_weight, score_logpi_phi  # noqa: E402
from phase4.hvp_logpi import _mean_logpi, score_and_hvps  # noqa: E402

SFT = "allenai/OLMo-2-0425-1B-SFT"
PHI = "model.layers.12.mlp.down_proj"
LEE = REPO / "data/lee_pairwise/toxicity_pairwise/split_0.jsonl"
EPS_SWEEP = [3e-2, 1e-2, 5e-3, 3e-3, 1e-3]


def _samples(tok, n_samples: int, p_max: int, r_max: int):
    """A few (prompt_ids, response_ids) from Lee pairs, truncated short so the
    double-backward graph (and a possible fp64 pass) stays cheap."""
    import json

    out = []
    with open(LEE) as fh:
        for line in fh:
            rec = json.loads(line)
            p = tok(rec["prompt_text"], return_tensors="pt")["input_ids"][0][:p_max]
            r = tok(rec["pert_gen_text"], return_tensors="pt", add_special_tokens=False)["input_ids"][0][:r_max]
            if len(p) >= 4 and len(r) >= 4:
                out.append((p, r))
            if len(out) >= n_samples:
                break
    return out


def _unit_dir(shape, seed: int, dtype, device):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(shape, generator=g, dtype=torch.float64).to(dtype=dtype, device=device)
    return v / v.norm()


def _check_sample(model, w, prompt_ids, response_ids, device, *, label, n_dirs=2):
    dtype = w.dtype
    for di in range(n_dirs):
        v = _unit_dir(w.shape, seed=di, dtype=dtype, device=device)
        # analytic s + HVP(v) (one shared graph)
        s, (hvp,) = score_and_hvps(model, w, prompt_ids, response_ids, [v], device)
        s64, v64, hvp64 = s.to(torch.float64), v.to(torch.float64), hvp.to(torch.float64)
        sdotv = float((s64 * v64).sum().item())                  # directional 1st deriv
        curv_analytic = float((hvp64 * v64).sum().item())        # v^T H v (often tiny)
        s_norm = float(s64.norm().item())
        hvp_norm = float(hvp64.norm().item())
        cos_Hv_v = curv_analytic / (hvp_norm + 1e-30)            # |.|<<1 => Hv _|_ v

        logpi0 = float(_mean_logpi(model, w, prompt_ids, response_ids, device, -100).item())

        best_deriv = best_scalar = best_vec = (None, float("inf"))
        for eps in EPS_SWEEP:
            with torch.no_grad():
                w.add_(eps * v)
            sp = score_logpi_phi(model, w, prompt_ids, response_ids, device)
            lp = float(_mean_logpi(model, w, prompt_ids, response_ids, device, -100).item())
            with torch.no_grad():
                w.sub_(2 * eps * v)
            sm = score_logpi_phi(model, w, prompt_ids, response_ids, device)
            lm = float(_mean_logpi(model, w, prompt_ids, response_ids, device, -100).item())
            with torch.no_grad():
                w.add_(eps * v)  # restore

            # (C) forward-only FIRST derivative <s,v> (resolves; grounds s)
            deriv_fd = (lp - lm) / (2 * eps)
            rel_d = abs(deriv_fd - sdotv) / (abs(sdotv) + 1e-30)
            # (A) forward-only scalar curvature v^T H v (tiny -> cancellation-limited)
            curv_fd = (lp - 2 * logpi0 + lm) / (eps * eps)
            rel_s = abs(curv_fd - curv_analytic) / (abs(curv_analytic) + 1e-30)
            # (B) vector HVP from central difference of the gradient (authoritative)
            hvp_fd = (sp - sm) / (2 * eps)
            rel_v = float((hvp_fd - hvp).to(torch.float64).norm().item()) / (hvp_norm + 1e-30)

            if rel_d < best_deriv[1]:
                best_deriv = (eps, rel_d)
            if rel_s < best_scalar[1]:
                best_scalar = (eps, rel_s)
            if rel_v < best_vec[1]:
                best_vec = (eps, rel_v)

        print(f"  [{label} dir{di}] |s|={s_norm:.3e} |Hv|={hvp_norm:.3e} "
              f"<s,v>={sdotv:+.4e} vHv={curv_analytic:+.4e} cos(Hv,v)={cos_Hv_v:+.2e}")
        print(f"      (C) 1st-deriv forward-only  rel err min={best_deriv[1]:.2e} @ eps={best_deriv[0]}  (resolves -> grounds s)")
        print(f"      (B) vector-HVP              rel err min={best_vec[1]:.2e} @ eps={best_vec[0]}  (AUTHORITATIVE 2nd-deriv check)")
        print(f"      (A) scalar-curv forward-only rel err min={best_scalar[1]:.2e} @ eps={best_scalar[0]}  (uninformative: vHv~{curv_analytic:.1e} below FD resolution)")
        yield best_vec[1]


def main() -> int:
    import transformers
    print("transformers", transformers.__version__, "| torch", torch.__version__)

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(SFT)
    print("\n=== fp32 GPU (eager attn) ===")
    model = AutoModelForCausalLM.from_pretrained(
        SFT, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    w = get_phi_weight(model, PHI)
    w.requires_grad_(True)
    print(f"phi {PHI} shape {tuple(w.shape)}")

    samples = _samples(tok, n_samples=2, p_max=16, r_max=20)
    print(f"{len(samples)} samples; seq lens {[ (len(p),len(r)) for p,r in samples ]}")

    worst_v = 0.0
    for si, (p, r) in enumerate(samples):
        for rel_v in _check_sample(model, w, p, r, device, label=f"fp32 s{si}"):
            worst_v = max(worst_v, rel_v)
    print(f"\nfp32 WORST vector-HVP rel err over samples/dirs: {worst_v:.2e}")

    # best-effort fp64 tightening (CPU, one very short sample)
    print("\n=== fp64 CPU (best-effort tightening) ===")
    try:
        del model
        torch.cuda.empty_cache()
        m64 = AutoModelForCausalLM.from_pretrained(
            SFT, torch_dtype=torch.float64, attn_implementation="eager"
        ).to("cpu").eval()
        for p in m64.parameters():
            p.requires_grad_(False)
        w64 = get_phi_weight(m64, PHI)
        w64.requires_grad_(True)
        p, r = _samples(tok, n_samples=1, p_max=8, r_max=8)[0]
        for rel_v in _check_sample(m64, w64, p, r, "cpu", label="fp64", n_dirs=1):
            print(f"fp64 vector-HVP rel err: {rel_v:.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"fp64 pass skipped ({type(exc).__name__}: {str(exc)[:160]})")

    print("\nHVP SMOKE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
