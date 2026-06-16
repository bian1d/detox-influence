"""Aggregate the sub-agent toxicity labels into the report the user asked for.

Reads /tmp/tox_label/manifest.json (per-rollout reward/step + set membership) and
all /tmp/tox_label/labels_*.jsonl (per-rollout is_toxic + category, reward-blind).
Produces, per set {sub_top, tox_top, random, bottom}:
  1. true-toxic rate + category breakdown
  2. RoBERTa cross-validation: among agent-toxic, fraction RoBERTa did NOT flag
     (reward > 0, i.e. RoBERTa says "not hate") — its MISSES — broken down by
     category (expect misses concentrated in NON-hate types: sexual/profanity/
     violence, since the RoBERTa model is hate-speech-only).
  3. comparison sub_top vs random vs bottom (does top-10% have higher toxic rate?)

Run:  python3 src/run_tox_label_aggregate.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LAB = Path("/tmp/tox_label")
REPORT = Path("reports/notes/c4_toxicity_labels.json")
CATS = ["race", "gender", "orientation", "profanity_abuse", "violence_threat", "sexual", "other"]
HATE_CATS = {"race", "gender", "orientation"}   # what the RoBERTa hate model targets
ROB_CLEAN = 0.0      # reward > 0 => RoBERTa nothate logit positive => "not hate"


def main() -> int:
    manifest = json.loads((LAB / "manifest.json").read_text())
    labels: dict[int, dict] = {}
    for f in sorted(LAB.glob("labels_*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            labels[int(d["id"])] = {"is_toxic": bool(d["is_toxic"]),
                                    "category": d.get("category", "none")}
    print(f"labels: {len(labels)} / manifest: {len(manifest)}")
    missing = [i for i in manifest if int(i) not in labels]
    if missing:
        print(f"  WARNING: {len(missing)} manifest ids have no label (agents incomplete?)")

    sets = {"sub_top": "in_sub_top", "tox_top": "in_tox_top",
            "random": "in_random", "bottom": "in_bottom"}
    report = {"n_labeled": len(labels), "n_missing": len(missing),
              "roberta_clean_threshold": ROB_CLEAN, "sets": {}}

    for sname, flag in sets.items():
        ids = [int(i) for i, mm in manifest.items() if mm[flag] and int(i) in labels]
        n = len(ids)
        toxic_ids = [i for i in ids if labels[i]["is_toxic"]]
        cat = Counter(labels[i]["category"] for i in toxic_ids)
        rewards = [manifest[str(i)]["reward"] for i in ids]
        # RoBERTa cross-val on agent-toxic
        rob_missed = [i for i in toxic_ids if manifest[str(i)]["reward"] > ROB_CLEAN]
        rob_missed_cat = Counter(labels[i]["category"] for i in rob_missed)
        # among RoBERTa "clean" (reward>0), what fraction does the agent call toxic?
        clean_ids = [i for i in ids if manifest[str(i)]["reward"] > ROB_CLEAN]
        clean_toxic = [i for i in clean_ids if labels[i]["is_toxic"]]
        report["sets"][sname] = {
            "n": n,
            "toxic_rate": round(len(toxic_ids) / max(n, 1), 4),
            "n_toxic": len(toxic_ids),
            "category_counts": {c: cat.get(c, 0) for c in CATS},
            "category_frac_of_toxic": {c: round(cat.get(c, 0) / max(len(toxic_ids), 1), 4) for c in CATS},
            "reward_mean": round(sum(rewards) / max(n, 1), 3),
            "roberta_missed_count": len(rob_missed),     # agent-toxic but reward>0
            "roberta_missed_frac_of_toxic": round(len(rob_missed) / max(len(toxic_ids), 1), 4),
            "roberta_missed_by_cat": {c: rob_missed_cat.get(c, 0) for c in CATS},
            "roberta_missed_nonhate_frac": round(
                sum(rob_missed_cat.get(c, 0) for c in CATS if c not in HATE_CATS) / max(len(rob_missed), 1), 4),
            "among_roberta_clean_agent_toxic_rate": round(len(clean_toxic) / max(len(clean_ids), 1), 4),
        }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    # console
    print("\n" + "=" * 74)
    print(f"{'set':10s} {'n':>5s} {'toxic%':>7s} {'reward':>7s} {'RoBERTa-miss%':>13s} {'miss-nonhate%':>13s}")
    for sname in sets:
        r = report["sets"][sname]
        print(f"{sname:10s} {r['n']:5d} {r['toxic_rate']*100:6.1f}% {r['reward_mean']:7.2f} "
              f"{r['roberta_missed_frac_of_toxic']*100:12.1f}% {r['roberta_missed_nonhate_frac']*100:12.1f}%")
    print("\nCategory frac of toxic (each set):")
    print(f"{'set':10s} " + " ".join(f"{c[:5]:>7s}" for c in CATS))
    for sname in sets:
        fr = report["sets"][sname]["category_frac_of_toxic"]
        print(f"{sname:10s} " + " ".join(f"{fr[c]*100:6.1f}%" for c in CATS))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
