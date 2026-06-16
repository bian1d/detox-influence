"""Dump rollout texts into labeling batches for the toxicity-type sub-agent pipeline.

Uses the C4 saved influence arrays (no re-scoring). Builds the sets the user
asked to characterise and writes deduplicated batch JSONL files (one rollout per
line) plus a manifest of per-rollout membership flags + RoBERTa reward, so the
aggregation can slice the sub-agent labels by set.

Sets:
  sub_top  = top-10% by |I| for the robust baseline-subtracted toxicity direction
  tox_top  = top-10% by |I| for the toxic-only direction
  random   = 1000 random rollouts (baseline toxicity rate control)
  bottom   = 1000 from the bottom-10% by |I| of sub (least influential control)

Run:  python3 src/run_tox_label_dump.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

C4 = Path("data/ekfac/influence/c4_top10")
OUT = Path("/tmp/tox_label")
BATCH = 550
PROMPT_CAP, RESP_CAP = 300, 260


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    I_base = torch.load(C4 / "I_baseline.pt")
    meta = json.loads((C4 / "meta.json").read_text())
    index = torch.load(C4 / "rollout_index.pt").tolist()
    N = len(meta)
    k10 = N // 10
    rng = np.random.RandomState(0)

    sub = np.abs(np.asarray(I_base["toxic_subtracted"]))
    tox = np.abs(np.asarray(I_base["toxic_only"]))
    sub_top = set(np.argsort(-sub)[:k10].tolist())
    tox_top = set(np.argsort(-tox)[:k10].tolist())
    bottom_pool = np.argsort(sub)[:k10]                       # least influential (sub)
    bottom = set(rng.choice(bottom_pool, size=1000, replace=False).tolist())
    random_set = set(rng.choice(N, size=1000, replace=False).tolist())

    all_ids = sorted(sub_top | tox_top | bottom | random_set)
    print(f"sets: sub_top={len(sub_top)} tox_top={len(tox_top)} "
          f"bottom={len(bottom)} random={len(random_set)} | union={len(all_ids)}")

    manifest = {}
    items = []
    for gid in all_ids:
        m = meta[gid]
        manifest[gid] = {
            "reward": round(m["reward"], 3), "step": m["step"], "resp_len": m["resp_len"],
            "in_sub_top": gid in sub_top, "in_tox_top": gid in tox_top,
            "in_bottom": gid in bottom, "in_random": gid in random_set,
        }
        items.append({"id": gid, "reward": round(m["reward"], 3),
                      "prompt": m["prompt"][:PROMPT_CAP], "response": m["response"][:RESP_CAP]})
    (OUT / "manifest.json").write_text(json.dumps(manifest))

    n_batches = (len(items) + BATCH - 1) // BATCH
    for b in range(n_batches):
        chunk = items[b * BATCH:(b + 1) * BATCH]
        with open(OUT / f"batch_{b:02d}.jsonl", "w") as f:
            for it in chunk:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {n_batches} batches of <= {BATCH} to {OUT}")
    # reward baselines for reference
    allrew = np.array([meta[i]["reward"] for i in range(N)])
    print(f"reward: all median={np.median(allrew):.3f} mean={allrew.mean():.3f}")
    return n_batches


if __name__ == "__main__":
    raise SystemExit(main())
