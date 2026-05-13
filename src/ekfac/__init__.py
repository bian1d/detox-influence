"""EK-FAC influence function pipeline for PPO/RLHF rollouts.

Stages:
    0  data, toy model
    1A factors: A, S accumulation (pseudo-labeled, prompt-masked, per-token mean)
    1B eigen:  Λ_corrected fit in Kronecker eigenbasis
    2  influence (eval): IHVP g_eval → p
    3  influence (train): per-sample gradient, score = -⟨g_m, p⟩

See CONTEXT.md §13.2 and phases/phase2.md.
"""
