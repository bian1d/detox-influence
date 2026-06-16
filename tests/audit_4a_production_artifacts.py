"""INDEPENDENT AUDIT 4a — direct checks on the PRODUCTION EK-FAC artifacts.

Loads the real GPT-Neo step_0650 factors from
data/ekfac/influence/run_step0650/ and re-derives every consistency relation
from scratch in fp64, treating the saved tensors as untrusted inputs:

  C1. Q_A, Q_S orthonormality (||Q Qᵀ − I||_inf)
  C2. A ≈ Q_A diag(Λ_A) Q_Aᵀ and S ≈ Q_S diag(Λ_S) Q_Sᵀ (eigendecomp self-consistency)
  C3. saved p_seq / p_toxic == inverse_hvp(g_*, factors) recomputed in fp64
      (catches a stale or mis-saved IHVP, and re-checks the two-level damping)
  C4. saved I_seq[m] / I_toxic[m] == -<g_eval_proj, s_m_proj/denom> recomputed
      from a FRESH per-sample gradient (my own grad code path) for a sample of m
  C5. g_toxic recomputed from (x_eval, y_toxic) text matches the saved g_toxic
  C6. self-influence sign + magnitude sanity reproduced independently
  C7. Λ layout: production denom uses Lambda (d_in,d_out); confirm
      inverse_hvp(g) == U diag(1/(Λ.T.flat+a*mean clamped)) Uᵀ vec(g) shape-wise
      via the eigen-direction probe on the REAL Q's (sampled directions).

Run (base env):  python3 tests/audit_4a_production_artifacts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout, load_rollout_file  # noqa: E402
from ekfac.eigen import inverse_hvp  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402

RUN = Path("data/ekfac/influence/run_step0650")
CACHE = Path("data/ekfac/cache/g_scaled_step0650_layer9")
CKPT = Path("data/ppo_checkpoints/step_0650")
ROLL = Path("data/rollouts")

RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def info(name: str, detail: str) -> None:
    print(f"  [info] {name}: {detail}", flush=True)


def main() -> int:
    print("AUDIT 4a: direct checks on production EK-FAC artifacts", flush=True)
    print(f"torch {torch.__version__}, numpy {np.__version__}", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EKFACConfig(layer_name="transformer.h.9.mlp.c_proj")

    A = torch.load(RUN / "A.pt", map_location="cpu").double()
    S = torch.load(RUN / "S.pt", map_location="cpu").double()
    Q_A = torch.load(RUN / "Q_A.pt", map_location="cpu").double()
    Q_S = torch.load(RUN / "Q_S.pt", map_location="cpu").double()
    Lam_A = torch.load(RUN / "Lambda_A.pt", map_location="cpu").double()
    Lam_S = torch.load(RUN / "Lambda_S.pt", map_location="cpu").double()
    Lam = torch.load(RUN / "Lambda.pt", map_location="cpu").double()
    print(f"  shapes: A{tuple(A.shape)} S{tuple(S.shape)} Q_A{tuple(Q_A.shape)} "
          f"Q_S{tuple(Q_S.shape)} Lambda{tuple(Lam.shape)}", flush=True)

    # C1 orthonormality.
    eqa = (Q_A @ Q_A.T - torch.eye(Q_A.shape[0], dtype=torch.float64)).abs().max().item()
    eqs = (Q_S @ Q_S.T - torch.eye(Q_S.shape[0], dtype=torch.float64)).abs().max().item()
    gate("C1 Q_A, Q_S orthonormal", eqa < 1e-9 and eqs < 1e-9,
         f"||Q_A Q_Aᵀ−I||={eqa:.2e}, ||Q_S Q_Sᵀ−I||={eqs:.2e}")

    # C2 reconstruction.
    rA = (Q_A @ torch.diag(Lam_A) @ Q_A.T - A).norm().item() / A.norm().item()
    rS = (Q_S @ torch.diag(Lam_S) @ Q_S.T - S).norm().item() / S.norm().item()
    gate("C2 A,S eigen-reconstruction", rA < 1e-9 and rS < 1e-9,
         f"rel err A={rA:.2e}, S={rS:.2e}")

    # C7 Λ layout / damping direction probe on real Q's (a few random eigendirs).
    Q_A_d, Q_S_d, Lam_d = Q_A.to(dev), Q_S.to(dev), Lam.to(dev)
    d_in, d_out = Q_A.shape[0], Q_S.shape[0]
    floor, alpha = cfg.damping_floor, cfg.damping_alpha
    lam_mean = Lam_d.mean()
    rng = np.random.default_rng(0)
    ok7, worst7 = True, 0.0
    for _ in range(6):
        i = int(rng.integers(0, d_in)); j = int(rng.integers(0, d_out))
        g = torch.outer(Q_S_d[:, j], Q_A_d[:, i])              # (d_out, d_in)
        x = inverse_hvp(g, Q_A_d, Q_S_d, Lam_d, damping_floor=floor, damping_alpha=alpha)
        denom = max(float(Lam_d[i, j]) + alpha * float(lam_mean), floor)
        rel = (x - g / denom).norm().item() / (g / denom).norm().item()
        worst7 = max(worst7, rel); ok7 &= rel < 1e-9
    gate("C7 production two-level damping on real eigendirections", ok7,
         f"worst rel err = {worst7:.2e}")

    # C3 p == inverse_hvp(g).
    for tag in ["seq", "toxic_C1", "toxic_C2", "toxic_C3"]:
        gp = RUN / f"g_{tag}.pt"
        pp = RUN / f"p_{tag}.pt"
        if not gp.exists() or not pp.exists():
            info(f"C3 {tag}", "skipped (missing g/p)")
            continue
        g = torch.load(gp, map_location=dev).double()
        p_saved = torch.load(pp, map_location=dev).double()
        p_re = inverse_hvp(g, Q_A_d, Q_S_d, Lam_d, damping_floor=floor, damping_alpha=alpha)
        rel = (p_re - p_saved).norm().item() / p_saved.norm().item()
        gate(f"C3 p_{tag} == inverse_hvp(g_{tag}) fp64", rel < 1e-6, f"rel err = {rel:.2e}")

    # Load model for C4/C5/C6.
    print("  loading θ* GPT-Neo (this import can take ~10s) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(CKPT.as_posix()).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(CKPT.as_posix())
    tok.pad_token = tok.eos_token
    layer = model.get_submodule(cfg.layer_name)

    # C5 g_toxic recompute.
    F_TOX = {"x": "To say that black people have made progress would be",
             "y": " a flat-out lie. They're as backward as ever, and history shows they always will be."}
    pid = tok(F_TOX["x"], return_tensors="pt").input_ids[0]
    fid = tok(F_TOX["x"] + F_TOX["y"], return_tensors="pt").input_ids[0]
    r_tox = Rollout(prompt_ids=pid, response_ids=fid[pid.shape[0]:], reward=0.0, step=0)
    g_tox_re = compute_per_sample_grad(model, layer, r_tox, cfg, device=dev).double()
    g_tox_saved = torch.load(RUN / "g_toxic_C1.pt", map_location=dev).double()
    rel5 = (g_tox_re - g_tox_saved).norm().item() / g_tox_saved.norm().item()
    gate("C5 g_toxic_C1 recomputed from text == saved", rel5 < 1e-4,
         f"rel err = {rel5:.2e} (fp32 grad path)")

    # C4 influence recompute for a sample of rollouts from the FULL index.
    g_seq = torch.load(RUN / "g_seq.pt", map_location=dev).double()
    p_seq = inverse_hvp(g_seq, Q_A_d, Q_S_d, Lam_d, damping_floor=floor, damping_alpha=alpha)
    p_tox = inverse_hvp(g_tox_saved, Q_A_d, Q_S_d, Lam_d, damping_floor=floor, damping_alpha=alpha)
    I_seq_saved = torch.load(RUN / "I_seq.pt", map_location="cpu").double().numpy()
    I_tox_saved = torch.load(RUN / "I_toxic_C1.pt", map_location="cpu").double().numpy()
    index = torch.load(RUN / "rollout_index.pt", map_location="cpu")  # (N, 2): step, idx_in_file

    # Recompute for a stratified sample of global indices.
    N = I_seq_saved.shape[0]
    sample_m = sorted(set(np.linspace(0, N - 1, 25).astype(int).tolist()))
    # Build a per-step cache of loaded files to avoid reloading.
    file_cache: dict[int, list[Rollout]] = {}
    worst_seq, worst_tox = 0.0, 0.0
    cmp_rows = []
    for m in sample_m:
        step, idx_in_file = int(index[m, 0]), int(index[m, 1])
        if step not in file_cache:
            file_cache[step] = load_rollout_file(ROLL / f"step_{step:04d}.pt")
        roll = file_cache[step][idx_in_file]
        s_m = compute_per_sample_grad(model, layer, roll, cfg, device=dev).double()
        I_seq_re = influence_score_klrl(s_m, p_seq)
        I_tox_re = influence_score_klrl(s_m, p_tox)
        rs = abs(I_seq_re - I_seq_saved[m]) / (abs(I_seq_saved[m]) + 1e-12)
        rt = abs(I_tox_re - I_tox_saved[m]) / (abs(I_tox_saved[m]) + 1e-12)
        worst_seq = max(worst_seq, rs); worst_tox = max(worst_tox, rt)
        cmp_rows.append((m, I_seq_saved[m], I_seq_re, rs))
    gate("C4 I_seq[m] reproduced from fresh per-sample grad", worst_seq < 1e-3,
         f"worst rel err = {worst_seq:.2e} over {len(sample_m)} sampled rollouts")
    gate("C4 I_toxic[m] reproduced from fresh per-sample grad", worst_tox < 1e-3,
         f"worst rel err = {worst_tox:.2e}")
    info("C4 sample (m, saved, recomputed, relerr)",
         "; ".join(f"{m}:{a:.3e}/{b:.3e}({c:.1e})" for m, a, b, c in cmp_rows[:5]))

    # C6 self-influence sign+rank sanity on step_0650.pt, fully independent.
    rolls650 = load_rollout_file(ROLL / "step_0650.pt")
    grads = [compute_per_sample_grad(model, layer, r, cfg, device=dev).double() for r in rolls650]
    sign_ok, top5_ok = True, True
    for m in (0, 5, 10, 15, 20):
        if m >= len(grads):
            continue
        p = inverse_hvp(grads[m], Q_A_d, Q_S_d, Lam_d, damping_floor=floor, damping_alpha=alpha)
        I_all = np.array([influence_score_klrl(g, p) for g in grads])
        I_self = I_all[m]
        rank = int((np.abs(I_all) > abs(I_self)).sum()) + 1
        sign_ok &= I_self < 0
        top5_ok &= rank <= 5
    gate("C6 self-influence sign (KL-RL: I_self < 0)", sign_ok, "all sampled self-IF negative")
    gate("C6 self-influence |I_self| ranks top-5", top5_ok, "all sampled self-IF in top-5")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (audit 4a) ===", flush=True)
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    print(f"gates: {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
