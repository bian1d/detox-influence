"""Stage 3 real-data self-influence sanity.

Independent of the toy LOO test: validates that on real PPO rollouts
(``data/rollouts/step_0500.pt`` from the new min_length=10 run), the
EK-FAC IF pipeline assigns large-magnitude scores to self-influence
queries (a rollout used as both training data point AND eval target).

Conventions:
    * Real training is PPO/KL-RL → use ``influence_score_klrl``
      (CONTEXT.md §5 sign). Self-IF is theoretically ≤ 0:
      ``I_self = −s_mᵀ F⁻¹ s_m`` and F⁻¹ is PSD with damping.
    * Production damping: ``max(Λ + 0.1·Λ̄, 1e-5)`` via ``inverse_hvp``.

Acceptance:
    * All 5 self-IFs are NEGATIVE (consistent with KL-RL sign convention).
    * For each self-target z_m, ``|I(z_m, z_m)|`` ranks in the top 5 of
      ``{|I(z_n, z_m)| : n in 32 rollouts}``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ekfac.config import EKFACConfig  # noqa: E402
from ekfac.data import load_rollout_file  # noqa: E402
from ekfac.eigen import eigendecompose, fit_lambda, inverse_hvp  # noqa: E402
from ekfac.factors import accumulate_AS  # noqa: E402
from ekfac.influence import influence_score_klrl  # noqa: E402
from ekfac.training import compute_per_sample_grad  # noqa: E402


REAL_ROLLOUT_PATH = Path("data/rollouts/step_0500.pt")
MODEL_NAME = "EleutherAI/gpt-neo-125m"
SELF_INDICES = (0, 5, 10, 15, 20)  # 5 self-targets

REPORT_PATH = (
    Path(__file__).parent.parent / "reports" / "notes" / "stage3_real_self_if_report.md"
)


@pytest.fixture(scope="module")
def gpt_neo_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to("cuda").eval()
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def real_rollouts():
    if not REAL_ROLLOUT_PATH.exists():
        pytest.skip(f"{REAL_ROLLOUT_PATH} not present")
    rollouts = load_rollout_file(REAL_ROLLOUT_PATH)
    assert len(rollouts) == 32
    return rollouts


@pytest.fixture(scope="module")
def real_pipeline_state(gpt_neo_on_gpu, real_rollouts):
    """Run Stage 1A + 1B once; reused for all 5 self-IF queries."""
    cfg = EKFACConfig(layer_name="transformer.h.9.mlp.c_proj")
    layer = gpt_neo_on_gpu.get_submodule(cfg.layer_name)
    device = torch.device("cuda")
    t0 = time.perf_counter()
    A, S, _ = accumulate_AS(gpt_neo_on_gpu, layer, real_rollouts, cfg, device=device)
    Q_A, _, Q_S, _ = eigendecompose(A, S)
    Lambda, _ = fit_lambda(
        gpt_neo_on_gpu, layer, real_rollouts, cfg, Q_A=Q_A, Q_S=Q_S, device=device,
    )
    t_stage1 = time.perf_counter() - t0

    # Per-sample gradient cache: one s_n per rollout, reused across all evals.
    t0 = time.perf_counter()
    grads = [
        compute_per_sample_grad(gpt_neo_on_gpu, layer, r, cfg, device=device).double()
        for r in real_rollouts
    ]
    t_grads = time.perf_counter() - t0
    return {
        "cfg": cfg, "layer": layer, "device": device,
        "Q_A": Q_A, "Q_S": Q_S, "Lambda": Lambda,
        "grads": grads,
        "t_stage1": t_stage1, "t_grads": t_grads,
    }


def test_self_influence_sanity(gpt_neo_on_gpu, real_rollouts, real_pipeline_state) -> None:
    state = real_pipeline_state
    cfg = state["cfg"]
    layer = state["layer"]
    Q_A_d = state["Q_A"].double()
    Q_S_d = state["Q_S"].double()
    Lambda_d = state["Lambda"].double()
    grads = state["grads"]

    results = []  # one entry per self-target
    t0 = time.perf_counter()
    for m in SELF_INDICES:
        if m >= len(real_rollouts):
            pytest.skip(f"self-index {m} out of range for {len(real_rollouts)} rollouts")
        g_eval = grads[m]  # eval target = self
        p = inverse_hvp(
            g_eval, Q_A_d, Q_S_d, Lambda_d,
            damping_floor=cfg.damping_floor, damping_alpha=cfg.damping_alpha,
        )
        I_list = [influence_score_klrl(s_n, p) for s_n in grads]  # length 32
        self_I = I_list[m]
        abs_I = [abs(x) for x in I_list]
        # Rank of |self_I| (1 = largest)
        abs_self = abs(self_I)
        rank = sum(1 for x in abs_I if x > abs_self) + 1
        results.append({
            "m": m, "self_I": self_I, "abs_self": abs_self, "rank": rank,
            "I_all": I_list,
        })
    t_eval_loop = time.perf_counter() - t0

    # ---- Build the structured report --------------------------------------
    lines: list[str] = []
    lines.append("# Stage 3 real-data self-influence sanity report")
    lines.append("")
    lines.append(f"**Setup:** GPT-Neo-125M (post-PPO), layer "
                 f"`{cfg.layer_name}`, rollouts from "
                 f"`{REAL_ROLLOUT_PATH}` (n=32). Production damping "
                 f"`max(Λ + {cfg.damping_alpha}·Λ̄, {cfg.damping_floor:.0e})`. "
                 f"IF sign convention: KL-RL (`influence_score_klrl`).")
    lines.append("")
    lines.append("Self-influence is theoretically ``I_self = −s_mᵀ F⁻¹ s_m ≤ 0`` "
                 "under the KL-RL convention (F⁻¹ is PSD with damping).")
    lines.append("")
    lines.append("## Per-self-target results")
    lines.append("")
    lines.append("| self idx m | I_self (signed) | |I_self|  | rank of |I_self| / 32 |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(
            f"| z_{r['m']} | {r['self_I']:+.4e} | {r['abs_self']:.4e} | "
            f"**{r['rank']}/32** |"
        )

    all_negative = all(r["self_I"] < 0 for r in results)
    all_top5 = all(r["rank"] <= 5 for r in results)
    lines.append("")
    lines.append("## Distribution of |I| for each self-target (showing top-5)")
    for r in results:
        absI = sorted(enumerate(r["I_all"]), key=lambda kv: -abs(kv[1]))
        head = ", ".join(f"|I(z_{idx}|z_{r['m']})|={abs(v):.3e}" for idx, v in absI[:5])
        lines.append(f"- self z_{r['m']}: {head}")
    lines.append("")
    lines.append("## Gates")
    lines.append(f"- All 5 self-IFs are NEGATIVE: **{'PASS' if all_negative else 'FAIL'}** "
                 f"({sum(1 for r in results if r['self_I'] < 0)}/5)")
    lines.append(f"- All 5 self-IFs rank in top-5 of |I|: **{'PASS' if all_top5 else 'FAIL'}** "
                 f"(ranks: {[r['rank'] for r in results]})")
    lines.append("")
    lines.append("## Wall-clock")
    lines.append(f"- Stage 1A+1B (A, S, Λ): {state['t_stage1']:.2f} s")
    lines.append(f"- 32 per-sample gradient computations: {state['t_grads']:.2f} s")
    lines.append(f"- 5 IHVPs + 5×32 dot products: {t_eval_loop:.3f} s")

    report_text = "\n".join(lines)
    print()
    print(report_text)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")

    failures = []
    if not all_negative:
        failures.append(("Negative-sign", f"{sum(1 for r in results if r['self_I'] < 0)}/5"))
    if not all_top5:
        failures.append(("|I_self| top-5", f"ranks: {[r['rank'] for r in results]}"))
    if failures:
        msg = "Stage 3 real self-IF gates failed:\n" + "\n".join(f"  - {n}: {v}" for n, v in failures)
        raise AssertionError(msg + f"\n\nReport at {REPORT_PATH}")
