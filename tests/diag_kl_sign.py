"""
Understand TRL's objective/kl sign convention and its effect on effective reward.
kl_k1_mean is deeply negative (-500 by step 9) despite kl_coef being fixed at 0.2.
"""
import subprocess
from pathlib import Path

# Check TRL's record_step_stats for objective/kl computation
trl_trainer = Path("/root/miniconda3/lib/python3.10/site-packages/trl/trainer/ppo_trainer.py")
lines = trl_trainer.read_text().splitlines()

# Find record_step_stats
for i, line in enumerate(lines):
    if "record_step_stats" in line and "def " in line:
        print(f"=== record_step_stats at L{i+1} ===")
        # Print 80 lines of it
        for j in range(i, min(i+100, len(lines))):
            print(f"  L{j+1}: {lines[j]}")
        break

# Find how non_score_reward is computed
print("\n=== non_score_reward computation ===")
for i, line in enumerate(lines):
    if "non_score_reward" in line or "kl_coef" in line.lower() and "kl" in line:
        print(f"  L{i+1}: {line}")
