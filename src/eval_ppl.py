"""
Task 3.3.1 — Wikitext-2 perplexity on base GPT-Neo-125m.

Uses the standard HuggingFace sliding-window recipe:
  stride=512, max_length=1024, wikitext-2-raw-v1 test split.

Expected PPL for GPT-Neo-125m: ~35-40 (vs GPT2-medium ~22; smaller model).

Run modes:
  --smoke   : PPL on a 200-token synthetic sequence (fast, no download)
  (default) : Full PPL on Wikitext-2 test split → data/baseline_ppl.json
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── core computation ──────────────────────────────────────────────────────────

def compute_ppl_from_ids(
    model: AutoModelForCausalLM,
    input_ids: Tensor,
    stride: int = 512,
    max_length: int = 1024,
    device: str = "cuda",
) -> float:
    """
    Standard HuggingFace sliding-window PPL given a pre-tokenized 1-D token tensor.

    For each window of max_length tokens starting every stride tokens:
      - The first (max_length - trg_len) tokens are context (labels = -100)
      - The last trg_len tokens are the targets (labels = real token ids)
      - trg_len = end_loc - prev_end_loc (= stride for all but the first window)
    model.forward(labels=...) returns mean NLL per non-masked token.
    Final PPL = exp(mean of per-window NLL losses).

    Using wikitext-2-RAW-v1 is required: the non-raw variant applies <unk>
    substitution and yields artificially low PPL numbers.
    """
    model.eval()
    seq_len = input_ids.size(1)

    nlls: list[Tensor] = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        chunk_ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = chunk_ids.clone()
        target_ids[:, :-trg_len] = -100   # mask context tokens

        with torch.no_grad():
            outputs = model(chunk_ids, labels=target_ids)
            nlls.append(outputs.loss.float())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()


def compute_ppl_wikitext2(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stride: int = 512,
    max_length: int = 1024,
    device: str = "cuda",
) -> float:
    """
    Compute PPL of GPT2-medium on the Wikitext-2 test split.

    MUST use 'wikitext-2-raw-v1' (not 'wikitext-2-v1'):
      - raw-v1 preserves original text → GPT2-medium scores ~22 PPL
      - v1 applies <unk> replacement → artificially lowers PPL, fails criterion

    Joins test documents with '\n\n' (standard practice) then tokenises the
    whole concatenated string as a single flat sequence.
    """
    print("Loading wikitext-2-raw-v1 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    print("Tokenising...")
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"  Total tokens: {seq_len:,}")

    n_windows = len(range(0, seq_len, stride))
    print(f"  Windows (stride={stride}, max_length={max_length}): {n_windows}")

    ppl = compute_ppl_from_ids(
        model, encodings.input_ids, stride=stride, max_length=max_length, device=device
    )
    return ppl


def save_baseline_ppl(ppl: float, output_path: Path) -> None:
    """Writes {"ppl_wikitext2_test": ppl} to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"ppl_wikitext2_test": ppl}, f, indent=2)
    print(f"Saved to {output_path}")


# ── smoke ─────────────────────────────────────────────────────────────────────

def run_smoke(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    smoke_dir: Path,
) -> None:
    """
    Smoke test on a 200-token sequence drawn from the GPT2 vocab.
    Verifies: sliding-window runs without error, returns finite float,
    output file is written with the right schema.
    """
    smoke_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Smoke: PPL on 200-token synthetic sequence ===")

    torch.manual_seed(42)
    # Random token ids within [0, vocab_size) to simulate a real sequence
    vocab_size = tokenizer.vocab_size
    fake_ids = torch.randint(0, vocab_size, (1, 200))

    ppl = compute_ppl_from_ids(
        model, fake_ids, stride=100, max_length=200, device=device
    )
    print(f"  Synthetic PPL: {ppl:.2f}  (expect large — random tokens)")
    assert 1.0 < ppl < 1e8, f"PPL out of sane range: {ppl}"

    # Also verify stride=512 / max_length=1024 settings on a slightly longer sequence
    fake_ids_long = torch.randint(0, vocab_size, (1, 600))
    ppl_long = compute_ppl_from_ids(
        model, fake_ids_long, stride=512, max_length=1024, device=device
    )
    print(f"  600-token sequence PPL (stride=512): {ppl_long:.2f}")
    assert 1.0 < ppl_long < 1e8

    out = smoke_dir / "ppl_smoke.json"
    with open(out, "w") as f:
        json.dump({"ppl_synthetic_200": ppl, "ppl_synthetic_600": ppl_long}, f, indent=2)
    print(f"  Smoke results saved to {out}")
    print("Smoke: PASS")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Wikitext-2 PPL evaluation for base GPT2-medium."
    )
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "data" / "baseline_ppl.json",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test only (200-token synthetic sequence). Writes to data/smoke/.",
    )
    args = parser.parse_args()

    MODEL_NAME = "EleutherAI/gpt-neo-125m"
    print(f"Loading {MODEL_NAME} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(args.device)

    if args.smoke:
        run_smoke(model, tokenizer, args.device, repo_root / "data" / "smoke")
    else:
        ppl = compute_ppl_wikitext2(
            model, tokenizer, stride=512, max_length=1024, device=args.device
        )
        print(f"\nWikitext-2 PPL: {ppl:.2f}  (expected range for GPT-Neo-125m: ~35–40)")
        if not (25.0 <= ppl <= 50.0):
            print(f"WARNING: PPL {ppl:.2f} outside expected range [25, 50]. "
                  "Check tokenisation or dataset config.")
        save_baseline_ppl(ppl, args.output)


if __name__ == "__main__":
    main()
