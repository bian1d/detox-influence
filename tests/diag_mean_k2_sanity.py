"""
Sanity-check that MeanKLPPOTrainer's per-token-mean k2 keeps nsr in [-1, 0].

With kl_coef=0.2 and mean reduction:
  nsr = -0.2 * k2_mean = -0.2 * (k2_sum / n_tokens)

For step_0001.pt (30-token responses, k2_sum max ≈ 77):
  k2_mean max ≈ 77/30 ≈ 2.57
  nsr min ≈ -0.2 * 2.57 ≈ -0.51  → inside [-1, 0] ✓
  nsr mean ≈ -0.2 * (12.9/30) ≈ -0.086  → inside [-1, 0] ✓

Compare against old (sum-based) nsr:
  nsr_sum min ≈ -0.2 * 77 ≈ -15.4  → way outside [-1, 0] ✗
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

k2_means = []
nsr_mean_list = []
nsr_sum_list = []

for i, r in enumerate(rollouts):
    pl = r["policy_logprobs"].float()
    rl = r["ref_logprobs"].float()
    n_tokens = len(pl)
    k2_per_tok = 0.5 * (pl - rl).square()
    k2_mean = k2_per_tok.mean().item()         # per-token mean (what MeanKLPPOTrainer uses)
    k2_sum  = k2_per_tok.sum().item()          # sum (old behavior)
    nsr_mean = -KL_COEF * k2_mean
    nsr_sum  = -KL_COEF * k2_sum
    k2_means.append(k2_mean)
    nsr_mean_list.append(nsr_mean)
    nsr_sum_list.append(nsr_sum)

k2_mean_t = torch.tensor(k2_means)
nsr_mean_t = torch.tensor(nsr_mean_list)
nsr_sum_t  = torch.tensor(nsr_sum_list)

print("=== k2_mean (per-token) ===")
print(f"  min    = {k2_mean_t.min():.4f}")
print(f"  median = {k2_mean_t.median():.4f}")
print(f"  max    = {k2_mean_t.max():.4f}  (at sample {k2_mean_t.argmax().item()})")
print(f"  mean   = {k2_mean_t.mean():.4f}")

print(f"\n=== nsr_mean = -kl_coef * k2_mean  (target: -1 ≤ nsr ≤ 0) ===")
n_outside = ((nsr_mean_t < -1) | (nsr_mean_t > 0)).sum().item()
print(f"  min    = {nsr_mean_t.min():.4f}  {'OK' if nsr_mean_t.min() >= -1 else 'FAIL: < -1'}")
print(f"  max    = {nsr_mean_t.max():.4f}  {'OK (≤0)' if nsr_mean_t.max() <= 0 else 'FAIL: > 0'}")
print(f"  n_outside_[-1,0] = {n_outside}  (expected 0)")

print(f"\n=== nsr_sum (old behavior, for comparison) ===")
print(f"  min    = {nsr_sum_t.min():.4f}  (would have been the training reward penalty)")
print(f"  max    = {nsr_sum_t.max():.4f}")

# Per-response effective reward with mean reduction
# Load actual rewards from rollout for eff_r check
rewards = torch.tensor([r["reward"] for r in rollouts])
eff_r = rewards + nsr_mean_t
print(f"\n=== effective_reward = task_reward + nsr_mean ===")
print(f"  task_reward: min={rewards.min():.3f}  max={rewards.max():.3f}  mean={rewards.mean():.3f}")
print(f"  eff_r:       min={eff_r.min():.3f}  max={eff_r.max():.3f}  mean={eff_r.mean():.3f}")
n_negative_eff = (eff_r < 0).sum().item()
print(f"  n_negative_eff_r = {n_negative_eff} / {len(eff_r)}")

print("\n=== Sanity check ===")
if n_outside == 0:
    print("PASS — all nsr_mean in [-1, 0]")
else:
    print(f"FAIL — {n_outside} samples have nsr_mean outside [-1, 0]")
