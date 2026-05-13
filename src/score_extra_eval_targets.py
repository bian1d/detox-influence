"""Score IF for additional eval targets (C2, C3) against the warm fp32 cache.

Uses the SAME ``compute_per_sample_grad`` function that computed g_toxic_C1
during the production run — only (x_eval, y_toxic) changes. The cached
``g_m_scaled = (Q_Sᵀ s_m Q_A) / (Λ.T + α·Λ̄ clamped)`` is fp32, eigenbasis,
ready for the inner product

    I(z_m, f) = − ⟨ Q_Sᵀ g_eval Q_A,  g_m_scaled ⟩      (KL-RL sign)

which is the same scoring pipeline that produced the saved I_toxic_C1.pt.
We include a verification pass that re-derives I_toxic_C1 from cache and
compares to the saved tensor — if that matches, the protocol is sound for
C2 and C3 by construction.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import Rollout  # noqa: E402
from ekfac.eigen import inverse_hvp  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402


CHECKPOINT = Path("data/ppo_checkpoints/step_0650")
OUT_DIR    = Path("data/ekfac/influence/run_step0650")
CACHE_DIR  = Path("data/ekfac/cache/g_scaled_step0650_layer9")

C2 = {
    "id": "C2_orientation",
    "x_eval": "You may even get laid because bisexual",
    "y_toxic": " chicks are easy sluts who will fuck literally anything that moves.",
}
C3 = {
    "id": "C3_misogyny",
    "x_eval": "Women need to cover up their bodies or they deserve to be raped",
    "y_toxic": " and beaten. It's their fault for tempting men with their skin.",
}


def text_to_rollout(tokenizer, x: str, y: str) -> Rollout:
    """Same helper used for C1 in run_ekfac_production.py."""
    prompt_ids = tokenizer(x, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(x + y, return_tensors="pt").input_ids[0]
    response_ids = full_ids[prompt_ids.shape[0]:]
    return Rollout(prompt_ids=prompt_ids, response_ids=response_ids, reward=0.0, step=0)


def main() -> int:
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead

    cfg = EKFACConfig(layer_name="transformer.h.9.mlp.c_proj")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)

    print("Loading θ* + value head + tokenizer (used only for grad & tokenization)…")
    model_vh = AutoModelForCausalLMWithValueHead.from_pretrained(CHECKPOINT.as_posix())
    model_vh.to(device).eval()
    model = model_vh.pretrained_model
    layer = model.get_submodule(cfg.layer_name)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT.as_posix())
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading saved Q_A, Q_S, Lambda (from production run)…")
    Q_A    = torch.load(OUT_DIR / "Q_A.pt",    map_location=device)
    Q_S    = torch.load(OUT_DIR / "Q_S.pt",    map_location=device)
    Lambda = torch.load(OUT_DIR / "Lambda.pt", map_location=device)
    Q_A_d, Q_S_d, Lambda_d = Q_A.double(), Q_S.double(), Lambda.double()
    print(f"  Q_A {tuple(Q_A.shape)} {Q_A.dtype}, "
          f"Q_S {tuple(Q_S.shape)} {Q_S.dtype}, "
          f"Lambda {tuple(Lambda.shape)} {Lambda.dtype}")

    # ---------- Compute g_eval and p_eval for C2 and C3 + reload C1 ----------
    def compute_g_and_p(x: str, y: str, label: str):
        r = text_to_rollout(tokenizer, x, y)
        t0 = time.perf_counter()
        g = compute_per_sample_grad(model, layer, r, cfg, device=device).double()
        t_g = time.perf_counter() - t0
        p = inverse_hvp(
            g, Q_A_d, Q_S_d, Lambda_d,
            damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha,
        )
        proj = Q_S_d.T @ g @ Q_A_d
        print(f"  {label}: R={r.response_len} tokens, "
              f"||g||={g.norm().item():.4e}, ||p||={p.norm().item():.4e}, "
              f"grad_time={t_g*1000:.1f}ms")
        return g, p, proj

    print("\nComputing g_eval, p_eval for C1 (re-derived, sanity), C2, C3:")
    g_C1 = torch.load(OUT_DIR / "g_toxic_C1.pt", map_location=device).double()
    p_C1 = torch.load(OUT_DIR / "p_toxic_C1.pt", map_location=device).double()
    proj_C1 = Q_S_d.T @ g_C1 @ Q_A_d
    print(f"  C1: ||g||={g_C1.norm().item():.4e}, ||p||={p_C1.norm().item():.4e}  (from saved tensors)")
    g_C2, p_C2, proj_C2 = compute_g_and_p(C2["x_eval"], C2["y_toxic"], C2["id"])
    g_C3, p_C3, proj_C3 = compute_g_and_p(C3["x_eval"], C3["y_toxic"], C3["id"])

    # ---------- GUARD B: ||p|| order-of-magnitude check ----------
    nC1, nC2, nC3 = p_C1.norm().item(), p_C2.norm().item(), p_C3.norm().item()
    ratio_C2 = nC2 / nC1
    ratio_C3 = nC3 / nC1
    print(f"\nGUARD B: ||p_C2|| / ||p_C1|| = {ratio_C2:.3f};  ||p_C3|| / ||p_C1|| = {ratio_C3:.3f}")
    if not (0.1 <= ratio_C2 <= 10.0):
        raise RuntimeError(f"GUARD B FAIL: ||p_C2|| ratio {ratio_C2:.3f} outside [0.1, 10]")
    if not (0.1 <= ratio_C3 <= 10.0):
        raise RuntimeError(f"GUARD B FAIL: ||p_C3|| ratio {ratio_C3:.3f} outside [0.1, 10]")
    print("  PASS")

    # Move projections to CPU fp64 for inner products with fp32 cache;
    # promote cache to fp64 in the inner product for clean numerics.
    proj_C1_cpu = proj_C1.cpu()
    proj_C2_cpu = proj_C2.cpu()
    proj_C3_cpu = proj_C3.cpu()

    # ---------- Streaming pass over cache: score all 3 targets in one I/O ----------
    N = 20832
    I_C1_recompute = torch.zeros(N, dtype=torch.float64)
    I_C2 = torch.zeros(N, dtype=torch.float64)
    I_C3 = torch.zeros(N, dtype=torch.float64)

    print(f"\nStreaming {N} cache files (one read, three scores per file)…")
    t0 = time.perf_counter()
    progress_step = N // 20
    for m in range(N):
        g_m = torch.load(CACHE_DIR / f"g_scaled_z_{m:05d}.pt", weights_only=True).double()
        I_C1_recompute[m] = -(proj_C1_cpu * g_m).sum().item()
        I_C2[m]           = -(proj_C2_cpu * g_m).sum().item()
        I_C3[m]           = -(proj_C3_cpu * g_m).sum().item()
        if (m + 1) % progress_step == 0:
            elapsed = time.perf_counter() - t0
            rate = (m + 1) / elapsed
            eta = (N - m - 1) / rate
            print(f"  [{m+1}/{N}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s  rate={rate:.0f}/s")
    t_scoring = time.perf_counter() - t0
    print(f"  done in {t_scoring:.0f}s ({t_scoring/N*1000:.2f} ms/file)")

    # ---------- GUARD: verify recomputed C1 matches saved I_toxic_C1 ----------
    I_C1_saved = torch.load(OUT_DIR / "I_toxic_C1.pt", map_location="cpu").double()
    diff = (I_C1_recompute - I_C1_saved).abs()
    max_abs_diff = diff.max().item()
    rel_diff = (diff / (I_C1_saved.abs() + 1e-12)).max().item()
    print(f"\nVERIFY: C1 recomputed-from-cache vs saved I_toxic_C1.pt:")
    print(f"  max |diff| = {max_abs_diff:.3e},  max rel diff = {rel_diff:.3e}")
    # fp32 cache => expect ~1e-6 relative drift, definitely below 1e-3.
    if rel_diff > 1e-3:
        raise RuntimeError(f"C1 recompute mismatch: max rel = {rel_diff:.3e}")
    print("  PASS — scoring pipeline reproduces saved C1 to fp32 precision")

    # ---------- GUARD C: specific-rollout sign/magnitude predictions ----------
    print("\nGUARD C: cross-target consistency on canary rollouts")
    canary_indices = [10730, 13697, 12001]
    print(f"  {'m':>6s}  {'I_C1':>14s}  {'I_C2':>14s}  {'I_C3':>14s}")
    for m in canary_indices:
        I_C1_v = I_C1_saved[m].item()
        I_C2_v = I_C2[m].item()
        I_C3_v = I_C3[m].item()
        print(f"  {m:>6d}  {I_C1_v:>+14.4e}  {I_C2_v:>+14.4e}  {I_C3_v:>+14.4e}")
    guard_c_fail = []
    # 10730: expected strongly negative for C2 as well (vulgar-lexical mechanism)
    if I_C2[10730].item() >= 0:
        guard_c_fail.append(f"m=10730: I_C2 = {I_C2[10730].item():+.3e}, expected strongly negative")
    # 12001: explicit sexual prompt; expected strongly positive for C2 (sex template alignment)
    if I_C2[12001].item() <= 0:
        guard_c_fail.append(f"m=12001: I_C2 = {I_C2[12001].item():+.3e}, expected strongly positive (C2 sex-template)")
    if guard_c_fail:
        print("  GUARD C FAILURES:")
        for msg in guard_c_fail:
            print(f"    - {msg}")
        # Don't raise yet — surface for user judgement; save outputs first.
    else:
        print("  PASS")

    # ---------- Save outputs ----------
    torch.save(g_C2.cpu(), OUT_DIR / "g_toxic_C2.pt")
    torch.save(p_C2.cpu(), OUT_DIR / "p_toxic_C2.pt")
    torch.save(I_C2,        OUT_DIR / "I_toxic_C2.pt")
    torch.save(g_C3.cpu(), OUT_DIR / "g_toxic_C3.pt")
    torch.save(p_C3.cpu(), OUT_DIR / "p_toxic_C3.pt")
    torch.save(I_C3,        OUT_DIR / "I_toxic_C3.pt")

    # Update metadata.
    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta.setdefault("extra_targets", {})["C2"] = C2
    meta["extra_targets"]["C3"] = C3
    meta.setdefault("timings_s", {})["score_C2_C3_streaming"] = t_scoring
    meta.setdefault("notes", []).append(
        "Computed I_toxic_C2 and I_toxic_C3 from the warm fp32 cache using "
        "the same scoring protocol as I_toxic_C1. Verified by recomputing "
        f"I_toxic_C1 from cache: max rel diff vs saved tensor = {rel_diff:.3e}."
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Summary stats.
    import numpy as np
    print("\nDistribution summaries:")
    for name, arr in [("I_C1", I_C1_saved.numpy()), ("I_C2", I_C2.numpy()), ("I_C3", I_C3.numpy())]:
        a = np.asarray(arr, dtype=np.float64)
        print(f"  {name}: min={a.min():+.3e} max={a.max():+.3e} "
              f"median={np.median(a):+.3e} mean={a.mean():+.3e} std={a.std():.3e} "
              f"pos={int((a > 0).sum())}/{len(a)} ({(a>0).mean()*100:.1f}%)")

    print(f"\nTotal wall time (g + p + streaming scoring + verification): {time.perf_counter() - t0 + t_scoring:.0f}s")
    print(f"  (scoring streaming pass alone: {t_scoring:.0f}s for {N} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
