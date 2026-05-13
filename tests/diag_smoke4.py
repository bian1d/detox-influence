"""Analyze smoke run 4 JSONL to understand the divergence pattern."""
import json
from pathlib import Path

log = Path("data/smoke/ppo_train_log.jsonl")
entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]

print(f"Steps logged: {len(entries)} (stopped at step {entries[-1]['step']})")
print(f"\n{'step':>5}  {'reward':>8}  {'approxkl':>9}  {'kl_k1':>9}  {'kl_coef':>7}  {'skips':>5}  {'len':>6}")
print("-" * 65)
for e in entries:
    print(
        f"{e['step']:>5}  {e['mean_reward']:>+8.3f}  "
        f"{e['approxkl']:>9.4f}  {e['kl_k1_mean']:>+9.3f}  "
        f"{e['kl_coef']:>7.4f}  {e['ratio_skip_count']:>5}  "
        f"{e['mean_response_length']:>6.1f}"
    )

# Highlight the divergence
print(f"\nMax approxkl: {max(e['approxkl'] for e in entries):.2f} at step {max(entries, key=lambda e: e['approxkl'])['step']}")
print(f"Max ratio_skip_count: {max(e['ratio_skip_count'] for e in entries)} at step {max(entries, key=lambda e: e['ratio_skip_count'])['step']}")
print(f"kl_coef range: {min(e['kl_coef'] for e in entries):.4f} – {max(e['kl_coef'] for e in entries):.4f}")
print(f"Reward range:  {min(e['mean_reward'] for e in entries):.3f} – {max(e['mean_reward'] for e in entries):.3f}")

# Check if reward is climbing
first5_avg = sum(e['mean_reward'] for e in entries[:5]) / min(5, len(entries))
last5_avg  = sum(e['mean_reward'] for e in entries[-5:]) / min(5, len(entries))
print(f"\nFirst-5 reward avg: {first5_avg:.3f}   Last-5 reward avg: {last5_avg:.3f}")
print(f"Reward trend: {'CLIMBING' if last5_avg > first5_avg + 0.1 else 'FLAT/FALLING'}")
