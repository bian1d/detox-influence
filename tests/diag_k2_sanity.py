"""
Sanity-check k2 = 0.5*(log_π_θ - log_π_ref)^2 on step_0001.pt rollout.

Expected:
- non_score_reward = -kl_coef * k2_sum ≤ 0 for ALL samples
- Sample 22 (the one with policy_lp=-18.34 for one token):
    k2 for that token ≈ 0.5 * 15.92^2 ≈ 126.7
    k2_sum for full response dominated by that token ≈ 127
    non_score_reward ≈ -0.2 * 127 ≈ -25.4
    effective reward ≈ task_reward - 25.4 ≈ 2.5 - 25.4 = -22.9
"""
import torch
from pathlib import Path

KL_COEF = 0.2
ROLLOUT_PATH = Path("data/smoke/rollouts/step_0001.pt")

if not ROLLOUT_PATH.exists():
    print(f"ERROR: {ROLLOUT_PATH} not found. Run smoke first.")
    raise SystemExit(1)

rollouts = torch.load(ROLLOUT_PATH)
print(f"Loaded {len(rollouts)} samples from {ROLLOUT_PATH}\n")

k2_sums = []
k1_sums = []
nsr_list = []  # non_score_reward = -kl_coef * k2_sum

for i, r in enumerate(rollouts):
    pl = r["policy_logprobs"].float()
    rl = r["ref_logprobs"].float()
    diff = pl - rl                         # log_π_θ - log_π_ref (k1 per token)
    k2_per_tok = 0.5 * diff.square()      # k2 per token, always >= 0
    k2_sum = k2_per_tok.sum().item()
    k1_sum = diff.sum().item()
    nsr = -KL_COEF * k2_sum               # non_score_reward, always <= 0
    k2_sums.append(k2_sum)
    k1_sums.append(k1_sum)
    nsr_list.append(nsr)

k2_tensor = torch.tensor(k2_sums)
nsr_tensor = torch.tensor(nsr_list)

print("=== k2_sum per sample (0.5*(Δlog)² summed over response tokens) ===")
print(f"  min    = {k2_tensor.min():.3f}")
print(f"  median = {k2_tensor.median():.3f}")
print(f"  p95    = {torch.quantile(k2_tensor, 0.95):.3f}")
print(f"  max    = {k2_tensor.max():.3f}  (at sample {k2_tensor.argmax().item()})")
print(f"  mean   = {k2_tensor.mean():.3f}")

print(f"\n=== non_score_reward = -kl_coef * k2_sum  (must be ≤ 0) ===")
n_positive = (nsr_tensor > 0).sum().item()
print(f"  min    = {nsr_tensor.min():.3f}")
print(f"  max    = {nsr_tensor.max():.3f}  {'OK (≤0)' if n_positive == 0 else f'FAIL: {n_positive} positive'}")
print(f"  n_positive = {n_positive}  (expected 0)")

# Sample 22 deep-dive
s22 = rollouts[22]
pl22 = s22["policy_logprobs"].float()
rl22 = s22["ref_logprobs"].float()
diff22 = pl22 - rl22
k2_22 = 0.5 * diff22.square()

# Find the extreme token
tok_max_idx = k2_22.argmax().item()
print(f"\n=== Sample 22 detail (the extreme token sample) ===")
print(f"  response length: {len(pl22)}")
print(f"  k2_sum: {k2_22.sum():.2f}  (expected ≈ 127)")
print(f"  non_score_reward: {-KL_COEF * k2_22.sum():.2f}  (expected ≈ -25.4)")
print(f"  effective_reward = task_reward + nsr ≈ {rollouts[22]['reward']:.3f} + {-KL_COEF * k2_22.sum():.2f} = {rollouts[22]['reward'] + (-KL_COEF * k2_22.sum()):.2f}  (expected ≈ -22.9)")
print(f"\n  Most extreme token (pos {tok_max_idx}):")
print(f"    policy_lp  = {pl22[tok_max_idx]:.3f}")
print(f"    ref_lp     = {rl22[tok_max_idx]:.3f}")
print(f"    k1 (diff)  = {diff22[tok_max_idx]:.3f}")
print(f"    k2         = {k2_22[tok_max_idx]:.3f}  (expected ≈ 126.7)")

print("\n=== k1_sum distribution (DIAGNOSTIC, not used for training) ===")
k1_tensor = torch.tensor(k1_sums)
print(f"  min    = {k1_tensor.min():.2f}")
print(f"  median = {k1_tensor.median():.2f}")
print(f"  max    = {k1_tensor.max():.2f}")
print(f"  mean   = {k1_tensor.mean():.2f}")

print("\nSanity check: PASS" if n_positive == 0 else "\nSanity check: FAIL — non_score_reward has positive values")
