"""Phase 4 E4 (gradient-similarity baseline) — the must-do, near-zero-cost half.

I_grad(m) = -g_f^T s_m  (NO F^-1).  If the gradient-similarity ranking already
agrees closely with the EK-FAC IF ranking (-g_f^T F^-1 s_m), then the Fisher
preconditioning is not decisive for the ranking — and a perturbation Delta to F
is even less able to reorder it (independent corroboration of E3).  If they
disagree a lot, F^-1 matters and the Delta question is genuinely open.

Computed from the production cache without recomputing s_m:
    -<g_f, s_m> = -<Q_S^T g_f Q_A, Q_S^T s_m Q_A>
                = -<(Q_S^T g_f Q_A) * denom_T, g_scaled_z[m]>   (undo 1/denom)

The Bayesian-IF/SGLD variant is intentionally NOT run (expensive; deferred).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from phase4.config import PHASE4_DATA_DIR  # noqa: E402
from phase4.e3_delta import cache_dot_scores, project_eval_vector  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EKFAC_RUN = REPO / "data" / "ekfac" / "influence" / "run_step0650"
CACHE_DIR = REPO / "data" / "ekfac" / "cache" / "g_scaled_step0650_layer9"
TARGETS = {"seq": "seq", "toxic_C1": "toxic_C1", "toxic_C2": "toxic_C2", "toxic_C3": "toxic_C3"}
N_ROLLOUTS = 20832
DAMPING_FLOOR = 1e-5
DAMPING_ALPHA = 0.1


def _jaccard(a, b, k):
    ta, tb = set(np.argsort(-a)[:k].tolist()), set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PHASE4_DATA_DIR.mkdir(parents=True, exist_ok=True)

    Q_A = torch.load(EKFAC_RUN / "Q_A.pt", map_location=device).to(torch.float64)
    Q_S = torch.load(EKFAC_RUN / "Q_S.pt", map_location=device).to(torch.float64)
    Lambda = torch.load(EKFAC_RUN / "Lambda.pt", map_location=device).to(torch.float64)
    denom_T = (Lambda + DAMPING_ALPHA * Lambda.mean()).clamp(min=DAMPING_FLOOR).T.contiguous()

    left = {}
    for t, stub in TARGETS.items():
        g_f = torch.load(EKFAC_RUN / f"g_{stub}.pt", map_location=device).to(torch.float64)
        left[t] = project_eval_vector(g_f, Q_A, Q_S) * denom_T          # gradient-sim left vector

    print(f"E4: streaming {N_ROLLOUTS} cached g_scaled_z for gradient similarity ...")
    I_grad = cache_dot_scores(left, CACHE_DIR, N_ROLLOUTS, device)

    results = {"n_rollouts": N_ROLLOUTS, "targets": {}}
    for t, stub in TARGETS.items():
        I_if = torch.load(EKFAC_RUN / f"I_{stub}.pt", map_location="cpu").numpy()
        ig = I_grad[t]
        res = {
            "spearman_grad_vs_IF": float(spearmanr(ig, I_if).statistic),
            "pearson_grad_vs_IF": float(np.corrcoef(ig, I_if)[0, 1]),
            "jaccard_top10": _jaccard(ig, I_if, 10),
            "jaccard_top50": _jaccard(ig, I_if, 50),
            "jaccard_top100": _jaccard(ig, I_if, 100),
        }
        results["targets"][t] = res
        print(f"  [{t}] grad-sim vs EK-FAC IF: Spearman={res['spearman_grad_vs_IF']:.4f}  "
              f"Pearson={res['pearson_grad_vs_IF']:.4f}  "
              f"J@10={res['jaccard_top10']:.2f} J@50={res['jaccard_top50']:.2f} "
              f"J@100={res['jaccard_top100']:.2f}")

    out = PHASE4_DATA_DIR / "e4_gradsim.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
