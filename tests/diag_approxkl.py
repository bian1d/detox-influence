"""
Inspect TRL's approxkl computation and the step 0 rollout to understand
whether approxkl=3.9 at step 0 is expected or indicates a real problem.
"""
import json
from pathlib import Path
import torch

repo = Path(__file__).parent.parent

# ── 1. Check what keys are in the JSONL ──────────────────────────────────────
log_path = repo / "data" / "smoke" / "ppo_train_log.jsonl"
if log_path.exists():
    entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    print(f"JSONL entries: {len(entries)}")
    if entries:
        print(f"  step 0 entry: {json.dumps(entries[0], indent=2)}")
else:
    print("No JSONL yet")

# ── 2. Look at TRL's stats computation in source ──────────────────────────────
import trl
import inspect

trl_path = Path(trl.__file__).parent
trainer_path = trl_path / "trainer" / "ppo_trainer.py"
print(f"\nTRL version: {trl.__version__}")
print(f"TRL ppo_trainer: {trainer_path}")

# Find approxkl in source
lines = trainer_path.read_text().splitlines()
for i, line in enumerate(lines):
    if "approxkl" in line.lower():
        print(f"  L{i+1}: {line}")

# ── 3. Inspect step 0 rollout logprobs ───────────────────────────────────────
rollout_path = repo / "data" / "smoke" / "rollouts" / "step_0000.pt"
if rollout_path.exists():
    rollouts = torch.load(rollout_path)
    print(f"\nStep 0 rollout: {len(rollouts)} samples")
    r = rollouts[0]
    pl = r["policy_logprobs"].float()
    rl = r["ref_logprobs"].float()
    print(f"  response length: {pl.shape[0]}")
    print(f"  policy_lp stats: min={pl.min():.3f} max={pl.max():.3f} mean={pl.mean():.3f}")
    print(f"  ref_lp    stats: min={rl.min():.3f} max={rl.max():.3f} mean={rl.mean():.3f}")
    # At step 0, policy == ref (before any updates), so these should be identical
    diff = (pl - rl).abs()
    print(f"  |policy - ref| max={diff.max():.6f} mean={diff.mean():.6f}")
    print(f"  (If policy==ref at step 0, these should be ~0 due to float16 rounding)")
else:
    print("No step 0 rollout found")

# ── 4. Understand the k2 formula TRL uses ────────────────────────────────────
# approxkl = mean(0.5 * (logprob_new - logprob_old)^2)
# This is the within-step drift, averaged over:
#   - all tokens in all responses
#   - all mini-batches in all PPO epochs (4 epochs × 16 mini-batches = 64 updates)
# So it accumulates across the full PPO step, not just one mini-batch update.
print("\nApproxKL interpretation:")
print("  TRL averages approxkl across all mini-batches in all ppo_epochs")
print("  With ppo_epochs=4, mini_batch=4, batch=64: 64 mini-batch updates per step")
print("  The cumulative drift from epoch 1 to epoch 4 is reflected in this number")
print("  Expected range in TRL detox tutorial: check the source for early-stop thresholds")
