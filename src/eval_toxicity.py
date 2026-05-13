"""
Task 3.3.2 — Baseline toxicity evaluation on a fixed 400-prompt subset.

Samples 400 prompts from RealToxicityPrompts (seed=42, no toxicity filter),
generates 30-token continuations, scores continuations only (NOT prompt+cont)
with evaluate.load("toxicity").

Run modes:
  --smoke   : 5-prompt test — verify continuation extraction and scoring
  (default) : Full 400-prompt baseline → data/baseline_toxicity.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── prompt sampling ───────────────────────────────────────────────────────────

def sample_and_save_eval_prompts(
    output_path: Path,
    n: int = 400,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Samples n prompts UNIFORMLY from RealToxicityPrompts (no toxicity filter).

    RealToxicityPrompts schema:
      row["prompt"]["text"]      — the prompt string
      row["prompt"]["toxicity"]  — float score (may be None)
    We do NOT filter by toxicity — sample from the full dataset uniformly.

    Saves {"indices": [...], "prompt_texts": [...]} to output_path.
    Returns (prompt_texts, indices).
    """
    import random
    print("Loading RealToxicityPrompts...")
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    total = len(dataset)
    print(f"  Dataset size: {total:,}")

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), n))
    prompts = [dataset[i]["prompt"]["text"] for i in indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"indices": indices, "prompt_texts": prompts}, f, indent=2)
    print(f"  Saved {n} prompt indices to {output_path}")
    return prompts, indices


def load_eval_prompts(path: Path) -> tuple[list[str], list[int]]:
    """Load the saved 400-prompt subset."""
    with open(path) as f:
        data = json.load(f)
    return data["prompt_texts"], data["indices"]


# ── generation ────────────────────────────────────────────────────────────────

def generate_continuations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 30,
    seed: int = 42,
    device: str = "cuda",
    batch_size: int = 16,
) -> list[str]:
    """
    Generates CONTINUATION-ONLY text (NOT prompt + continuation).

    Settings: do_sample=True, top_k=0, top_p=1.0 (pure sampling, no truncation).
    Seed is applied once before the first generation call so the full batch
    of 400 prompts is reproducible.

    Returns only the newly generated tokens decoded as strings.
    Each output is guaranteed to be ≤ max_new_tokens new tokens.
    """
    torch.manual_seed(seed)
    model.eval()

    continuations: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off prompt tokens — keep only the new tokens
        new_tokens = outputs[:, prompt_len:]   # (batch, new_len)
        for row in new_tokens:
            # Strip trailing eos if present before decoding
            ids = row.tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            continuations.append(tokenizer.decode(ids, skip_special_tokens=True))

    return continuations


# ── toxicity scoring ──────────────────────────────────────────────────────────

def score_toxicity_eval(continuations: list[str]) -> dict:
    """
    Scores continuations (NOT prompts) with evaluate.load("toxicity").

    evaluate toxicity returns scores ∈ [0, 1]:
      0 = neutral/safe,  1 = hate/toxic
    This is the softmax probability of the hate class, produced by
    facebook/roberta-hate-speech-dynabench-r4-target — the same model used as
    the PPO reward model (but wrapped here as a softmax probability, not a raw
    logit). The two are intentionally in opposite directions: eval score ↓ and
    training reward ↑ both mean less toxic output.

    Returns {"mean": float, "std": float, "n": int}.
    """
    import evaluate
    import numpy as np

    toxicity_metric = evaluate.load("toxicity", module_type="measurement")
    result = toxicity_metric.compute(predictions=continuations)
    scores = result["toxicity"]   # list of floats ∈ [0, 1]
    return {
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "n":    len(scores),
    }


# ── smoke ─────────────────────────────────────────────────────────────────────

def run_smoke(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    smoke_dir: Path,
) -> None:
    """
    5-prompt smoke test. Verifies:
      - Continuation-only output (not prompt+continuation)
      - Each continuation is ≤ 30 new tokens
      - Toxicity scorer runs and returns finite values
    Writes to data/smoke/.
    """
    smoke_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Smoke: 5-prompt generation + toxicity scoring ===")

    print("Loading RealToxicityPrompts (5 prompts)...")
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    import random
    rng = random.Random(42)
    indices = sorted(rng.sample(range(len(dataset)), 5))
    prompts = [dataset[i]["prompt"]["text"] for i in indices]

    print("Generating continuations (seed=42)...")
    conts = generate_continuations(
        model, tokenizer, prompts,
        max_new_tokens=30, seed=42, device=device, batch_size=5,
    )

    print("\nResults:")
    all_ok = True
    for i, (prompt, cont) in enumerate(zip(prompts, conts)):
        n_tokens = len(tokenizer(cont)["input_ids"])
        prompt_in_cont = prompt.strip() in cont
        status = "OK" if n_tokens <= 30 and not prompt_in_cont else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}] prompt={repr(prompt[:40])}...")
        print(f"         cont={repr(cont[:60])}")
        print(f"         n_tokens={n_tokens}, prompt_leaked={prompt_in_cont}")

    print("\nScoring toxicity on 5 continuations...")
    scores = score_toxicity_eval(conts)
    print(f"  Toxicity: mean={scores['mean']:.4f}  std={scores['std']:.4f}")
    assert 0.0 <= scores["mean"] <= 1.0, "Toxicity score out of [0,1]"

    out = smoke_dir / "toxicity_smoke.json"
    with open(out, "w") as f:
        json.dump({"prompts": prompts, "continuations": conts, "scores": scores}, f, indent=2)
    print(f"\nSmoke results saved to {out}")
    print(f"Smoke: {'PASS' if all_ok else 'FAIL'}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Baseline toxicity evaluation on 400 fixed RealToxicityPrompts."
    )
    parser.add_argument(
        "--eval-prompts", type=Path,
        default=repo_root / "data" / "eval_prompts_400.json",
        help="Path to save/load the 400 eval prompt indices.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "data" / "baseline_toxicity.json",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="5-prompt smoke test only. Writes to data/smoke/.",
    )
    args = parser.parse_args()

    MODEL_NAME = "EleutherAI/gpt-neo-125m"
    print(f"Loading {MODEL_NAME} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(args.device)

    if args.smoke:
        run_smoke(model, tokenizer, args.device, repo_root / "data" / "smoke")
        return

    # ── Full run ───────────────────────────────────────────────────────────
    if args.eval_prompts.exists():
        print(f"Loading existing eval prompts from {args.eval_prompts}...")
        prompts, indices = load_eval_prompts(args.eval_prompts)
    else:
        prompts, indices = sample_and_save_eval_prompts(
            args.eval_prompts, n=400, seed=42
        )

    print(f"\nGenerating continuations for {len(prompts)} prompts...")
    conts = generate_continuations(
        model, tokenizer, prompts,
        max_new_tokens=30, seed=42, device=args.device, batch_size=16,
    )
    assert len(conts) == 400, f"Expected 400 continuations, got {len(conts)}"

    print("Scoring toxicity...")
    result = score_toxicity_eval(conts)
    print(f"\nBaseline toxicity:  mean={result['mean']:.4f}  std={result['std']:.4f}  n={result['n']}")
    print(f"(Burtenshaw tutorial baseline for GPT-Neo-125m: ~0.1627)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
