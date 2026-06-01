"""Q3 gate: what is data/ekfac/cache/g_scaled_step0650_layer9/g_scaled_z_*.pt?

The user asked, before any E3 mass run, to verify whether the cached
``g_scaled_z`` tensors are the raw per-sample score s_m.  If not, STOP and
report the exact numbers (the literal check the user specified plus the
correct reconstruction), do NOT auto-fallback.

This script only does linear algebra on already-cached tensors (no model).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "data" / "ekfac" / "influence" / "run_step0650"
CACHE = REPO / "data" / "ekfac" / "cache" / "g_scaled_step0650_layer9"


def main() -> None:
    g_scaled_0 = torch.load(CACHE / "g_scaled_z_00000.pt", map_location="cpu").double()
    p_seq = torch.load(RUN / "p_seq.pt", map_location="cpu").double()
    g_seq = torch.load(RUN / "g_seq.pt", map_location="cpu").double()
    I_seq = torch.load(RUN / "I_seq.pt", map_location="cpu")
    Q_A = torch.load(RUN / "Q_A.pt", map_location="cpu").double()
    Q_S = torch.load(RUN / "Q_S.pt", map_location="cpu").double()
    Lambda = torch.load(RUN / "Lambda.pt", map_location="cpu").double()
    rollout_index = torch.load(RUN / "rollout_index.pt", map_location="cpu")

    print(f"shapes: g_scaled_z[0] {tuple(g_scaled_0.shape)} {g_scaled_0.dtype}, "
          f"p_seq {tuple(p_seq.shape)}, g_seq {tuple(g_seq.shape)}")
    print(f"I_seq length {I_seq.numel()}, rollout_index[0] = {rollout_index[0].tolist()}")
    print(f"||g_scaled_z[0]||_F = {g_scaled_0.norm().item():.6e}")
    print(f"||p_seq||_F        = {p_seq.norm().item():.6e}")
    print()

    I_seq_0 = float(I_seq[0].item())
    print(f"PRODUCTION  I_seq[0]                         = {I_seq_0:.10e}")

    # (1) The literal check the user specified.
    literal = -(g_scaled_0 * p_seq).sum().item()
    print(f"(1) LITERAL  -<g_scaled_z[0], p_seq>         = {literal:.10e}")
    print(f"    match I_seq[0]?  {abs(literal - I_seq_0) < 1e-6 * (abs(I_seq_0)+1e-12)}  "
          f"(abs diff {abs(literal - I_seq_0):.4e}, ratio {literal / I_seq_0:.4f})")
    print()

    # (2) The CORRECT reconstruction implied by the production code:
    #     I_seq[m] = -<Q_S^T g_seq Q_A, g_scaled_z[m]>, and g_scaled_z =
    #     Q_S^T s_m Q_A / denom_T, so this equals -g_seq^T F^-1 s_m.
    g_eval_proj = Q_S.T @ g_seq @ Q_A                    # (d_out, d_in)
    correct = -(g_eval_proj * g_scaled_0).sum().item()
    print(f"(2) CORRECT  -<Q_S^T g_seq Q_A, g_scaled_z[0]> = {correct:.10e}")
    print(f"    match I_seq[0]?  {abs(correct - I_seq_0) < 1e-6 * (abs(I_seq_0)+1e-12)}  "
          f"(abs diff {abs(correct - I_seq_0):.4e})")
    print()

    # (3) Confirm the denom used: g_scaled_z should equal s_m_proj / denom_T.
    #     We cannot see s_m here, but we can confirm p_seq = F^-1 g_seq is
    #     consistent with g_scaled_z living in the eigenbasis: i.e.
    #     un-projecting g_scaled_z[0] gives F^-1 s_0 in original space, whose
    #     inner product with g_seq also reproduces I_seq[0].
    Fi_s0 = Q_S @ g_scaled_0 @ Q_A.T                     # = F^-1 s_0 (original space)
    via_unproj = -(g_seq * Fi_s0).sum().item()
    print(f"(3) UNPROJ   -<g_seq, Q_S g_scaled_z[0] Q_A^T> = {via_unproj:.10e}  "
          f"(== -g_seq^T F^-1 s_0)")
    print(f"    match I_seq[0]?  {abs(via_unproj - I_seq_0) < 1e-6 * (abs(I_seq_0)+1e-12)}")


if __name__ == "__main__":
    main()
