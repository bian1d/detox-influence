"""Analyze smoke run 5 JSONL (k2 estimator)."""
import json
from pathlib import Path

log = Path("data/smoke/ppo_train_log.jsonl")
entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]

print(f"Steps logged: {len(entries)}")
print(f"\n{'step':>5}  {'reward':>8}  {'k2_kl':>9}  {'k1_kl':>9}  {'approxkl':>9}  {'kl_coef':>7}  {'nsr_est':>9}  {'eff_r':>8}")
print("-" * 80)
for e in entries:
    k2 = e.get("k2_kl_mean", float("nan"))
    k1 = e.get("k1_kl_mean", float("nan"))
    rew = e.get("mean_reward", float("nan"))
    akl = e.get("approxkl", float("nan"))
    kc  = e.get("kl_coef", float("nan"))
    nsr = -kc * k2 if k2 == k2 else float("nan")  # non_score_reward estimate
    eff = rew + nsr if rew == rew and nsr == nsr else float("nan")
    print(f"{e['step']:>5}  {rew:>+8.3f}  {k2:>9.3f}  {k1:>+9.3f}  {akl:>9.4f}  {kc:>7.4f}  {nsr:>+9.2f}  {eff:>+8.2f}")

print("\n(nsr_est = -kl_coef × k2_kl_mean; eff_r = reward + nsr_est)")
print("Note: nsr_est is PER RESPONSE (sums k2 over all tokens then × kl_coef).")
print("With kl_coef=0.2 and k2_sum=64 at step 1: nsr ≈ -12.9 → eff_r ≈ -10.7")
print("This negative effective reward from step 1 onward drives the instability.")
