"""Phase 4 — theoretical error-budget & robustness of the RL-KL influence
function.

Experiments (executed E1 → E2 → E3, each stop-and-report):
    E1  advantage A(x,y) distribution + within-prompt variance decomposition
        + ICC estimation CI.  Cheap; reuses Phase 1 data.
    E2  ||G(theta*)|| first-order non-stationarity (split-half unbiased).
    E3  Delta error-budget: does dropping Delta change per-rollout IF ranking.

All experiments analyse the FROZEN snapshot theta* = step_0650 with the
adaptive KL coefficient beta = 0.23652004338891985 that was in force at that
step (see phase1.config.BETA — bit-identical to the value used for Phase 1's
ICC=0.892).  KL estimator is k1 throughout, consistent with Phases 0/1/2.
"""
