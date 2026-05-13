"""Phase 1 orchestrator.

Modes:
  --smoke      1 prompt × 4 samples (Action Item 6).
  --stage2     N=100 prompts × K=32 samples from eval_prompts_400 (in-training-
               distribution variance).
  --framing-b  N=100 RTP prompts excluded from PPO training set AND eval set
               (held-out variance, generalization check).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from phase1.config import (  # noqa: E402
    BETA,
    BETA_STEP,
    CKPT_DIR,
    EVAL_PROMPTS_PATH,
    K_SAMPLES_STAGE2,
    N_PROMPTS_STAGE2,
    REF_MODEL_ID,
    REWARD_MODEL_ID,
    SEED,
)
from phase1.score import score_one_prompt  # noqa: E402
from phase1.variance import (  # noqa: E402
    aggregate_statistics,
    outliers_by_within_variance,
    per_prompt_variance,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "phase1"


def _load_models(device: str):
    print(f"Loading θ* from {CKPT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_star = AutoModelForCausalLM.from_pretrained(CKPT_DIR).to(device).eval()

    print(f"Loading π_ref = {REF_MODEL_ID}")
    model_ref = AutoModelForCausalLM.from_pretrained(REF_MODEL_ID).to(device).eval()

    print(f"Loading reward model = {REWARD_MODEL_ID}")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    reward_model = (
        AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID).to(device).eval()
    )
    return tokenizer, model_star, model_ref, reward_tokenizer, reward_model


def run_smoke(device: str) -> None:
    tokenizer, model_star, model_ref, reward_tokenizer, reward_model = _load_models(device)

    with open(EVAL_PROMPTS_PATH) as f:
        prompt_text = json.load(f)["prompt_texts"][0]

    print(f"\nβ at step {BETA_STEP} = {BETA:.12f}")
    print(f"Sampling K=4 from θ* for prompt:\n  {prompt_text!r}\n")

    _, rows = score_one_prompt(
        prompt_text=prompt_text,
        k=4,
        seed=SEED,
        device=device,
        tokenizer=tokenizer,
        model_star=model_star,
        model_ref=model_ref,
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        beta=BETA,
    )

    print(f"{'i':>2}  {'len':>3}  {'r':>8}  {'KL_k1':>8}  {'β·KL':>8}  {'R̃':>8}  response")
    print("-" * 100)
    for i, row in enumerate(rows):
        print(
            f"{i:>2}  {row['response_len']:>3}  {row['reward']:+8.4f}  "
            f"{row['k1_kl']:+8.4f}  {BETA * row['k1_kl']:+8.4f}  {row['R_tilde']:+8.4f}  "
            f"{row['response_text']!r}"
        )

    R = [row["R_tilde"] for row in rows]
    print(
        f"\nR̃ min={min(R):.4f}  max={max(R):.4f}  mean={sum(R)/len(R):.4f}  "
        f"span(max-min)={max(R) - min(R):.4f}"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "smoke_one_prompt.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "beta": BETA,
                "beta_step": BETA_STEP,
                "seed": SEED,
                "prompt_text": prompt_text,
                "samples": rows,
            },
            f,
            indent=2,
        )
    print(f"\nSaved smoke output to {out_path}")


def _run_prompts(
    prompt_records: list[tuple[int, str, int]],
    output_prefix: str,
    device: str,
    tokenizer,
    model_star,
    model_ref,
    reward_tokenizer,
    reward_model,
) -> dict:
    """Score K=32 samples for each (key, prompt_text, seed) record.

    Writes <prefix>_samples.jsonl, <prefix>_per_prompt.json, <prefix>_summary.json.
    Returns the summary dict.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = DATA_DIR / f"{output_prefix}_samples.jsonl"
    per_prompt_path = DATA_DIR / f"{output_prefix}_per_prompt.json"
    summary_path = DATA_DIR / f"{output_prefix}_summary.json"

    per_prompt_list = []
    n_total = len(prompt_records)
    t0 = time.time()
    with open(samples_path, "w") as f_jsonl:
        for i, (key, prompt_text, seed) in enumerate(prompt_records):
            _, rows = score_one_prompt(
                prompt_text=prompt_text,
                k=K_SAMPLES_STAGE2,
                seed=seed,
                device=device,
                tokenizer=tokenizer,
                model_star=model_star,
                model_ref=model_ref,
                reward_tokenizer=reward_tokenizer,
                reward_model=reward_model,
                beta=BETA,
            )
            for j, row in enumerate(rows):
                f_jsonl.write(
                    json.dumps(
                        {
                            "prompt_idx_in_subset": i,
                            "prompt_key": key,
                            "sample_idx": j,
                            "prompt_text": prompt_text,
                            **row,
                        }
                    )
                    + "\n"
                )

            pv = per_prompt_variance(key, prompt_text, rows)
            per_prompt_list.append(pv)

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_total - i - 1)
                print(
                    f"  [{output_prefix}] {i+1:>3}/{n_total}  "
                    f"mean_R̃={pv.mean_R_tilde:+.3f}  var_R̃={pv.var_R_tilde:.4f}  "
                    f"corr(len,R̃)={pv.corr_len_R:+.3f}  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                )

    print(f"\nWrote per-sample rows to {samples_path}")

    per_prompt_serializable = [
        {
            "prompt_key": p.prompt_idx,
            "prompt_text": p.prompt_text,
            "mean_R_tilde": p.mean_R_tilde,
            "var_R_tilde": p.var_R_tilde,
            "cv_R_tilde": p.cv_R_tilde,
            "corr_len_R": p.corr_len_R,
            "response_lens": p.response_lens,
            "reward_samples": p.reward_samples,
            "k1_kl_samples": p.k1_kl_samples,
            "R_tilde_samples": p.R_tilde_samples,
        }
        for p in per_prompt_list
    ]
    with open(per_prompt_path, "w") as f:
        json.dump(per_prompt_serializable, f, indent=2)
    print(f"Wrote per-prompt aggregates to {per_prompt_path}")

    summary = aggregate_statistics(per_prompt_list)
    summary["beta"] = BETA
    summary["beta_step"] = BETA_STEP
    summary["seed"] = SEED
    summary["mode"] = output_prefix
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote aggregate summary to {summary_path}")

    print(f"\n=== {output_prefix} aggregate ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n=== Top-10 {output_prefix} prompts by within-prompt variance ===")
    for p in outliers_by_within_variance(per_prompt_list, n_top=10):
        print(
            f"  var={p.var_R_tilde:8.4f}  mean_R̃={p.mean_R_tilde:+7.3f}  "
            f"corr(L,R̃)={p.corr_len_R:+.3f}  key={p.prompt_idx}  "
            f"text={p.prompt_text[:60]!r}"
        )

    print(f"\nTotal {output_prefix} wall-clock: {time.time() - t0:.1f}s")
    return summary


def _pick_stage2_prompts(n: int, seed: int) -> list[tuple[int, str, int]]:
    """In-training-distribution prompts: random subset of eval_prompts_400.json.

    Returns list of (key=idx_in_400, prompt_text, sample_seed).
    """
    with open(EVAL_PROMPTS_PATH) as f:
        ep = json.load(f)
    all_prompts: list[str] = ep["prompt_texts"]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(all_prompts), size=n, replace=False).tolist())
    return [(int(i), all_prompts[i], SEED + int(i)) for i in idx]


def _ppo_training_indices() -> set[int]:
    """Re-derive the exact 10K RTP indices Phase 0 PPO trained on.

    Mirrors src/train_ppo.py::load_training_prompts: toxicity > 0.3,
    excluding eval_prompts_400 indices, then random.Random(42).sample(n=10000).
    """
    import random as _random
    from datasets import load_dataset

    print("Reconstructing PPO training index set...")
    rtp = load_dataset("allenai/real-toxicity-prompts", split="train")
    with open(EVAL_PROMPTS_PATH) as f:
        eval_data = json.load(f)
    eval_set = set(eval_data["indices"])

    tox_col = rtp["prompt"]
    candidates = [
        i for i, row in enumerate(tox_col)
        if row.get("toxicity") is not None
        and float(row["toxicity"]) > 0.3
        and i not in eval_set
    ]
    rng = _random.Random(42)
    sampled = rng.sample(candidates, 10_000)
    print(f"  candidate pool: {len(candidates):,}, sampled: {len(sampled):,}")
    return set(sampled)


def _pick_framing_b_prompts(n: int, seed: int) -> list[tuple[int, str, int]]:
    """Held-out distribution: RTP prompts not in PPO training AND not in eval_prompts_400.

    Uniformly sampled (no toxicity filter), with a fresh seed to avoid
    collision with the SEED=42 used by training+stage2.
    """
    from datasets import load_dataset

    train_idx = _ppo_training_indices()
    with open(EVAL_PROMPTS_PATH) as f:
        eval_data = json.load(f)
    eval_idx = set(eval_data["indices"])

    rtp = load_dataset("allenai/real-toxicity-prompts", split="train")
    total = len(rtp)
    excluded = train_idx | eval_idx
    pool = [i for i in range(total) if i not in excluded]
    print(
        f"Framing B pool: total RTP={total:,}, excluded={len(excluded):,}, "
        f"available={len(pool):,}"
    )

    rng = np.random.RandomState(seed)
    chosen = sorted(rng.choice(pool, size=n, replace=False).tolist())
    prompts = [rtp[i]["prompt"]["text"] for i in chosen]
    return [
        (int(i), prompts[k], SEED + 100_000 + int(i))
        for k, i in enumerate(chosen)
    ]


def run_stage2(device: str) -> None:
    tokenizer, model_star, model_ref, reward_tokenizer, reward_model = _load_models(device)
    print(f"\nβ at step {BETA_STEP} = {BETA:.12f}")

    records = _pick_stage2_prompts(N_PROMPTS_STAGE2, seed=SEED)
    print(f"Stage 2: N={N_PROMPTS_STAGE2} prompts × K={K_SAMPLES_STAGE2} samples")
    print(f"First prompt: {records[0][1]!r}")
    print(f"Last prompt:  {records[-1][1]!r}")
    _run_prompts(
        records,
        output_prefix="stage2",
        device=device,
        tokenizer=tokenizer,
        model_star=model_star,
        model_ref=model_ref,
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
    )


def run_framing_b(device: str) -> None:
    tokenizer, model_star, model_ref, reward_tokenizer, reward_model = _load_models(device)
    print(f"\nβ at step {BETA_STEP} = {BETA:.12f}")

    records = _pick_framing_b_prompts(N_PROMPTS_STAGE2, seed=SEED + 1000)
    print(f"Framing B: N={N_PROMPTS_STAGE2} held-out prompts × K={K_SAMPLES_STAGE2} samples")
    print(f"First prompt: {records[0][1]!r}")
    print(f"Last prompt:  {records[-1][1]!r}")
    _run_prompts(
        records,
        output_prefix="framing_b",
        device=device,
        tokenizer=tokenizer,
        model_star=model_star,
        model_ref=model_ref,
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 orchestrator.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--framing-b", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        run_smoke(device)
    elif args.stage2:
        run_stage2(device)
    elif args.framing_b:
        run_framing_b(device)
    else:
        raise SystemExit("Pass --smoke | --stage2 | --framing-b.")


if __name__ == "__main__":
    main()
