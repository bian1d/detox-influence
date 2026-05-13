"""Phase 1 configuration constants.

β is read from data/ppo_train_log.jsonl at the log entry where "step":650.
The kl_coef field there is the adaptive controller value POST-update at step
650; the step_0650 checkpoint weights are also POST-update, so the pair is
consistent. The initial controller value was 0.2.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CKPT_DIR = REPO_ROOT / "data" / "ppo_checkpoints" / "step_0650"
REF_MODEL_ID = "EleutherAI/gpt-neo-125m"
REWARD_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"

EVAL_PROMPTS_PATH = REPO_ROOT / "data" / "eval_prompts_400.json"
PPO_TRAIN_LOG = REPO_ROOT / "data" / "ppo_train_log.jsonl"

BETA = 0.23652004338891985
BETA_STEP = 650

SEED = 42
MAX_PROMPT_TOKENS = 64

GENERATION_KWARGS = {
    "do_sample": True,
    "top_k": 0,
    "top_p": 1.0,
    "max_new_tokens": 30,
    "min_new_tokens": 10,
}

N_PROMPTS_STAGE2 = 100
K_SAMPLES_STAGE2 = 32
