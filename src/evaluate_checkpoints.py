"""
Task 3.5 — Per-checkpoint evaluation for PPO-trained GPT-Neo-125m.

Evaluates each checkpoint saved under data/ppo_checkpoints/ (plus step=0 base)
on five metrics:
  1. mean_toxicity    — evaluate.load("toxicity") on 400 fixed RealToxicityPrompts
  2. std_toxicity     — standard deviation of per-prompt scores
  3. mean_reward_logit— raw nothate logit from reward model (no softmax)
  4. ppl_wikitext2    — sliding-window PPL on Wikitext-2 test split
  5. mean_length      — mean generated response length (tokens)

Outputs:
  data/checkpoint_eval.csv       — one row per checkpoint
  reports/phase0_checkpoint_curves.png — four-panel plot

Run modes:
  --smoke : evaluate only step_0000 + step_0050 checkpoints (fast check)
  (default): evaluate all checkpoints in ppo_checkpoints/

Usage:
  python3 src/evaluate_checkpoints.py
  python3 src/evaluate_checkpoints.py --smoke
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from eval_ppl import compute_ppl_from_ids
from eval_toxicity import generate_continuations, score_toxicity_eval


REWARD_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"
PPL_STRIDE      = 512
PPL_MAX_LENGTH  = 1024


# ── metric helpers ────────────────────────────────────────────────────────────

def compute_mean_reward_logit(
    continuations: list[str],
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 16,
) -> float:
    """
    Returns mean of logits[:, 0] (raw nothate logit) over all continuations.
    Identical formula to train_ppo.compute_rewards — must stay consistent.
    """
    all_logits: list[float] = []
    for i in range(0, len(continuations), batch_size):
        batch = continuations[i : i + batch_size]
        inputs = reward_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits.float()
        all_logits.extend(logits[:, 0].tolist())
    return float(sum(all_logits) / len(all_logits))


def _load_wikitext2_ids(tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Loads and tokenizes the Wikitext-2 test split (raw-v1) as a 1-D token tensor.
    Caches the result across checkpoints so tokenization runs only once.
    """
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    return enc.input_ids  # (1, N)


# ── single-checkpoint evaluation ─────────────────────────────────────────────

def evaluate_one_checkpoint(
    model_path: str,
    tokenizer: AutoTokenizer,
    eval_prompts: list[str],
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    wikitext_ids: torch.Tensor,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    Runs all five metrics for one checkpoint. Loads the model, evaluates,
    then frees GPU memory before returning.

    Returns a dict with keys:
      model_path, mean_toxicity, std_toxicity, mean_reward_logit,
      ppl_wikitext2, mean_length, n_short_responses.
    """
    print(f"\n  Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    # ── generate continuations ────────────────────────────────────────────
    print(f"  Generating {len(eval_prompts)} continuations ...")
    conts = generate_continuations(
        model, tokenizer, eval_prompts,
        max_new_tokens=30, seed=seed, device=device, batch_size=16,
    )

    # ── toxicity ──────────────────────────────────────────────────────────
    tox = score_toxicity_eval(conts)

    # ── reward logit ──────────────────────────────────────────────────────
    mean_rew = compute_mean_reward_logit(conts, reward_model, reward_tokenizer, device)

    # ── PPL ───────────────────────────────────────────────────────────────
    print("  Computing PPL ...")
    ppl = compute_ppl_from_ids(
        model, wikitext_ids, stride=PPL_STRIDE, max_length=PPL_MAX_LENGTH, device=device,
    )

    # ── response length ───────────────────────────────────────────────────
    lengths = [len(tokenizer(c)["input_ids"]) for c in conts]
    mean_length = sum(lengths) / len(lengths)
    n_short = sum(1 for l in lengths if l < 10)

    del model
    torch.cuda.empty_cache()

    return {
        "model_path":        model_path,
        "mean_toxicity":     tox["mean"],
        "std_toxicity":      tox["std"],
        "mean_reward_logit": mean_rew,
        "ppl_wikitext2":     ppl,
        "mean_length":       mean_length,
        "n_short_responses": n_short,
    }


# ── all-checkpoint evaluation ─────────────────────────────────────────────────

def evaluate_all_checkpoints(
    checkpoint_dir: Path,
    base_model_name: str,
    eval_prompts_path: Path,
    output_csv: Path,
    reports_dir: Path,
    device: str = "cuda",
    smoke: bool = False,
) -> list[dict]:
    """
    Iterates step=0 (base model) + all saved checkpoints in checkpoint_dir.

    Checkpoint directory layout:
      ppo_checkpoints/step_0000/   ← base checkpoint (saved at PPO step 0)
      ppo_checkpoints/step_0050/   ← first periodic checkpoint
      ...
      ppo_checkpoints/step_1999/   ← final checkpoint

    For each checkpoint, reads ppo_step.json to get the step number.
    Falls back to parsing the directory name if ppo_step.json is absent.

    Writes checkpoint_eval.csv and phase0_checkpoint_curves.png.
    Returns list of result dicts sorted by step.
    """
    # Load eval prompts (fixed 400-prompt set)
    print(f"Loading eval prompts from {eval_prompts_path} ...")
    with open(eval_prompts_path) as f:
        data = json.load(f)
    eval_prompts: list[str] = data["prompt_texts"]
    print(f"  {len(eval_prompts)} prompts loaded.")

    # Tokenizer (shared across checkpoints, taken from base model)
    print(f"Loading tokenizer from {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Wikitext-2 tokenization done once
    print("Loading Wikitext-2 test split for PPL ...")
    wikitext_ids = _load_wikitext2_ids(tokenizer)
    print(f"  Wikitext-2 tokens: {wikitext_ids.shape[1]:,}")

    # Reward model (shared across checkpoints)
    print(f"Loading reward model {REWARD_MODEL_ID} ...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID).to(device)
    reward_model.eval()

    # Collect checkpoints: sorted list of (step, path)
    ckpts: list[tuple[int, str]] = []
    for ckpt_path in sorted(checkpoint_dir.iterdir()):
        if not ckpt_path.is_dir():
            continue
        step_json = ckpt_path / "ppo_step.json"
        if step_json.exists():
            with open(step_json) as f:
                step = json.load(f)["ppo_step"]
        else:
            # Fallback: parse "step_NNNN" from directory name
            try:
                step = int(ckpt_path.name.split("_")[-1])
            except ValueError:
                print(f"  SKIP {ckpt_path.name}: cannot determine step number")
                continue
        ckpts.append((step, str(ckpt_path)))

    ckpts.sort(key=lambda x: x[0])

    if smoke:
        ckpts = ckpts[:2]
        print(f"\n[Smoke] Evaluating first {len(ckpts)} checkpoints only.")

    print(f"\nEvaluating {len(ckpts)} checkpoints ...")

    results: list[dict] = []
    for step, path in ckpts:
        print(f"\n[step={step}] {path}")
        row = evaluate_one_checkpoint(
            model_path=path,
            tokenizer=tokenizer,
            eval_prompts=eval_prompts,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            wikitext_ids=wikitext_ids,
            device=device,
        )
        row["step"] = step
        results.append(row)
        print(
            f"  toxicity={row['mean_toxicity']:.4f}  "
            f"reward={row['mean_reward_logit']:.3f}  "
            f"ppl={row['ppl_wikitext2']:.2f}  "
            f"length={row['mean_length']:.1f}"
        )

    # ── write CSV ─────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "model_path", "mean_toxicity", "std_toxicity",
        "mean_reward_logit", "ppl_wikitext2", "mean_length", "n_short_responses",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {output_csv}")

    # ── plot ──────────────────────────────────────────────────────────────
    _plot_curves(results, reports_dir)

    return results


def _plot_curves(results: list[dict], reports_dir: Path) -> None:
    """Four-panel plot: toxicity, reward, PPL, response length vs PPO step."""
    if len(results) < 2:
        print("  (too few checkpoints to plot)")
        return

    steps   = [r["step"]              for r in results]
    tox     = [r["mean_toxicity"]     for r in results]
    rew     = [r["mean_reward_logit"] for r in results]
    ppls    = [r["ppl_wikitext2"]     for r in results]
    lengths = [r["mean_length"]       for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GPT-Neo-125m PPO Checkpoint Evaluation", fontsize=14)

    axes[0, 0].plot(steps, tox, marker="o", color="red")
    axes[0, 0].set_title("Mean Toxicity (↓ better)")
    axes[0, 0].set_xlabel("PPO step")
    axes[0, 0].set_ylabel("mean toxicity score")

    axes[0, 1].plot(steps, rew, marker="o", color="green")
    axes[0, 1].set_title("Mean Reward Logit (↑ better)")
    axes[0, 1].set_xlabel("PPO step")
    axes[0, 1].set_ylabel("nothate logit (raw)")

    axes[1, 0].plot(steps, ppls, marker="o", color="blue")
    axes[1, 0].set_title("Wikitext-2 PPL (↓ better)")
    axes[1, 0].set_xlabel("PPO step")
    axes[1, 0].set_ylabel("perplexity")

    axes[1, 1].plot(steps, lengths, marker="o", color="purple")
    axes[1, 1].axhline(y=10, color="red", linestyle="--", alpha=0.5, label="min_length=10")
    axes[1, 1].set_title("Mean Response Length")
    axes[1, 1].set_xlabel("PPO step")
    axes[1, 1].set_ylabel("mean tokens")
    axes[1, 1].legend()

    plt.tight_layout()
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "phase0_checkpoint_curves.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Per-checkpoint eval: toxicity, PPL, reward logit, response length."
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=repo_root / "data" / "ppo_checkpoints",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="EleutherAI/gpt-neo-125m",
    )
    parser.add_argument(
        "--eval-prompts", type=Path,
        default=repo_root / "data" / "eval_prompts_400.json",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=repo_root / "data" / "checkpoint_eval.csv",
    )
    parser.add_argument(
        "--reports-dir", type=Path,
        default=repo_root / "reports",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Evaluate only first 2 checkpoints (fast smoke check).",
    )
    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    if not args.eval_prompts.exists():
        raise FileNotFoundError(f"Eval prompts not found: {args.eval_prompts}")

    results = evaluate_all_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        base_model_name=args.base_model,
        eval_prompts_path=args.eval_prompts,
        output_csv=args.output_csv,
        reports_dir=args.reports_dir,
        device=args.device,
        smoke=args.smoke,
    )

    # ── sanity check: step=0 should match baseline files ──────────────────
    baseline_tox_path = repo_root / "data" / "baseline_toxicity.json"
    if baseline_tox_path.exists():
        with open(baseline_tox_path) as f:
            baseline_tox = json.load(f)
        step0 = next((r for r in results if r["step"] == 0), None)
        if step0:
            delta = abs(step0["mean_toxicity"] - baseline_tox["mean"])
            print(f"\nSanity check (step=0 vs baseline_toxicity.json):")
            print(f"  step=0 toxicity:  {step0['mean_toxicity']:.4f}")
            print(f"  baseline toxicity: {baseline_tox['mean']:.4f}")
            print(f"  delta: {delta:.4f}  {'OK (≤ 0.02)' if delta <= 0.02 else 'WARN: delta > 0.02'}")

    if len(results) > 1:
        step0 = next((r for r in results if r["step"] == 0), None)
        stepN = results[-1]
        if step0:
            print(f"\nOverall detox summary:")
            print(f"  step=0    toxicity={step0['mean_toxicity']:.4f}  "
                  f"reward={step0['mean_reward_logit']:.3f}  ppl={step0['ppl_wikitext2']:.2f}")
            print(f"  step={stepN['step']}  toxicity={stepN['mean_toxicity']:.4f}  "
                  f"reward={stepN['mean_reward_logit']:.3f}  ppl={stepN['ppl_wikitext2']:.2f}")
            tox_delta = step0["mean_toxicity"] - stepN["mean_toxicity"]
            print(f"  toxicity reduction: {tox_delta:.4f} "
                  f"({'PASS' if tox_delta > 0 else 'FAIL: no detox'})")


if __name__ == "__main__":
    main()
