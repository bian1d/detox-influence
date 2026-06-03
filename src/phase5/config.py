"""Phase 5 Stage 1 configuration (OLMo-2-1B SFT quick test).

theta* = OLMo-2-0425-1B-SFT (supervised, treated as a policy that drifted from
its pretrained reference under a *nominal* beta — see reports/phase5_olmo.md
Section 3 for why this is a heuristic probe, not a strict A2 test).
reference = OLMo-2-0425-1B (pretrained, the SFT start point).
phi = layer-12 down_proj (the SwiGLU back-to-hidden linear; 0.75 depth, the
analog of GPT-Neo's layer-9 c_proj).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

THETA_STAR_ID = "allenai/OLMo-2-0425-1B-SFT"
REF_ID = "allenai/OLMo-2-0425-1B"
REWARD_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"

PHI_LAYER = "model.layers.12.mlp.down_proj"     # (2048, 8192) = 16.78M params

# beta: SFT has no training-time beta. Use GPT-Neo's nominal value as the
# primary and sweep to expose ICC's sensitivity (spec Section 3).
BETA_NOMINAL = 0.2365
BETA_SWEEP = [0.1, 0.2, 0.2365, 0.5]

# ICC sampling (matched to GPT-Neo Phase 1: same prompts, K, raw completion).
ICC_N_PROMPTS = 100
ICC_K_SAMPLES = 32
SEED = 42

EVAL_PROMPTS_PATH = REPO / "data" / "eval_prompts_400.json"   # RTP tox>0.3 subset
LEE_DIR = REPO / "data" / "lee_pairwise" / "toxicity_pairwise"
OUT_DIR = REPO / "data" / "phase5"
REPORT = REPO / "reports" / "phase5_olmo.md"

# EK-FAC factor budget on OLMo down_proj (match GPT-Neo ~600k response tokens).
FACTOR_TOKEN_BUDGET = 600_000

# GPT-Neo baseline numbers for the side-by-side (reports/phase4_gptneo_negative_result.md).
GPTNEO_BASELINE = {
    "icc_k1": 0.892,
    "within_var_share_reward": 0.47,
    "within_var_share_kl": 0.56,
    "thermometer_range": "4–34",
    "phi_params": 2.36e6,
}


def select_icc_prompts(n: int = ICC_N_PROMPTS, seed: int = SEED) -> list[tuple[int, str]]:
    """The identical RTP subset GPT-Neo's ICC=0.892 used: a seed-42 random
    100-subset of eval_prompts_400.json (mirrors run_phase1._pick_stage2_prompts).
    Returns [(idx_in_400, prompt_text), ...]."""
    ep = json.loads(EVAL_PROMPTS_PATH.read_text())
    all_prompts: list[str] = ep["prompt_texts"]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(all_prompts), size=n, replace=False).tolist())
    return [(int(i), all_prompts[i]) for i in idx]
