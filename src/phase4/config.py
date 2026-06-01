"""Phase 4 configuration.

theta* and beta are inherited verbatim from Phase 1 so that any quantity E1
recomputes (advantage, within-prompt variance, ICC) is bit-comparable to the
published Phase 1 numbers.  We re-export rather than redefine to make drift
impossible: there is exactly one definition of BETA in the repository.
"""
from __future__ import annotations

from pathlib import Path

# phase1.config is the single source of truth for theta*, beta, the eval prompt
# set, the reference/reward model ids and the generation kwargs.  Importing it
# here guarantees E1's recomputation cannot silently diverge from Phase 1.
from phase1.config import (  # noqa: F401  (re-exported for downstream phase4 code)
    BETA,
    BETA_STEP,
    CKPT_DIR,
    EVAL_PROMPTS_PATH,
    GENERATION_KWARGS,
    MAX_PROMPT_TOKENS,
    REF_MODEL_ID,
    REWARD_MODEL_ID,
    SEED,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Phase 1 artefacts E1 reads.
PHASE1_DIR = REPO_ROOT / "data" / "phase1"
STAGE2_PER_PROMPT = PHASE1_DIR / "stage2_per_prompt.json"
STAGE2_SUMMARY = PHASE1_DIR / "stage2_summary.json"

# Phase 4 outputs.
PHASE4_DATA_DIR = REPO_ROOT / "data" / "phase4"
PHASE4_REPORT_DIR = REPO_ROOT / "reports" / "phase4"

# ICC published by Phase 1 (stage2, 100 prompts x 32 rollouts), for anchoring
# the resample CI.  Read from stage2_summary at runtime; this is the expected
# value as a guard against accidentally pointing at a different run.
PHASE1_ICC = 0.8916784010219726

# E1 ICC-resample knobs.
ICC_N_PROMPTS = 20            # size of the resampled subset
ICC_N_RESEEDS = 10            # independent response redraws per prompt
ICC_K_SAMPLES = 32            # responses per prompt per reseed (matches Phase 1)
ICC_SEED_BASE = 2000          # reseed s uses ICC_SEED_BASE + s
ICC_SUBSET_SEED = 1234        # RNG seed for the uniform-by-within-var pick
