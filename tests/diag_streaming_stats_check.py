"""Gate the memory-streaming U-stat/jackknife/two-half implementation in
run_phase5_stage1_thermometer against the Phase-4 reference (jackknife_ci,
naive_quadratic, full-matrix two-half cosine) on a small tensor where both fit.

The streaming version exists because OLMo's (200, 16.78M) fp64 d_p matrix is
26.8 GB and OOMs the reference path; it must be numerically identical.

    /root/miniconda3/envs/olmo/bin/python tests/diag_streaming_stats_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from phase4.e2_stationarity import jackknife_ci, naive_quadratic  # noqa: E402
from run_phase5_stage1_thermometer import _streaming_stats  # noqa: E402


def main() -> int:
    torch.manual_seed(7)
    d_p = torch.randn(11, 5, 3, dtype=torch.float32) * 1e3   # OLMo-like large scale

    ref = jackknife_ci(d_p, None)
    ref_naive = naive_quadratic(d_p, None)
    gh = d_p.reshape(11, -1).to(torch.float64)
    a, b = gh[: 11 // 2].mean(0), gh[11 // 2:].mean(0)
    ref_cos = float((a @ b / (a.norm() * b.norm() + 1e-30)).item())

    st = _streaming_stats(d_p)

    checks = [
        ("point", st["point"], ref["point"]),
        ("jackknife_se", st["jackknife_se"], ref["jackknife_se"]),
        ("naive", st["naive"], ref_naive),
        ("two_half_cosine", st["two_half_cosine"], ref_cos),
    ]
    ok = True
    for name, got, want in checks:
        rel = abs(got - want) / (abs(want) + 1e-30)
        print(f"  {name:>16}: streaming={got:+.12e}  ref={want:+.12e}  rel={rel:.2e}")
        ok &= rel < 1e-12
    print("\nSTREAMING STATS MATCH REFERENCE" if ok else "\nMISMATCH — DO NOT RUN THERMOMETER")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
