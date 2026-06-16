"""
Reproduction of Lee et al. 2024's toxicity probe on GPT2-medium.

This is an ablation diagnostic, not Phase 0 production. The question being
answered: when we run our probe-training pipeline against the SAME setup Lee
et al. used (GPT2-medium, Jigsaw, last-layer mean-pooled residual stream),
does logit-lens of our trained probe match their published probe.pt's
result (21/30 profanity in top-30)?

If yes → our pipeline is correct, GPT-Neo-125m's 0/30 result is structural
(distributed toxic mechanism, no concentrated direction at last layer).
If no → our pipeline has a bug to fix.

Methodology mirrors Lee et al. 2024, Section 3.1:
  - GPT2-medium (24 layers, d=1024)
  - residual stream at last layer (x_bar^{L-1}, here hidden_states[-1])
  - mean-pooled across all (non-pad) timesteps
  - Jigsaw toxic comment binary task: OR of 6 toxicity columns
  - 90:10 train/val split, seed=42

Probe architecture matches probe.pt structure (single d-dim tensor, no bias):
  P(Toxic | x_bar) = sigmoid(W . x_bar)
  W in R^d (no bias term, to match probe.pt's torch.Tensor shape (1024,))

Output: data/probe/probe_gpt2medium.pt (payload dict with weight + metadata).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "gpt2-medium"   # 24 layers, d=1024, d_mlp=4096
HIDDEN     = 1024


# ── data loading (identical to train_probe.py for parity) ─────────────────────

def load_jigsaw(
    jigsaw_csv: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """
    Loads Jigsaw train.csv (159,571 rows after correct pandas-handling of
    embedded newlines), OR-folds 6 toxicity columns into a binary label,
    90:10 split with seed=42.

    Note: Lee et al. report "561,808 comments" — this is wc -l of train.csv
    which over-counts due to embedded newlines in comment_text. Our row
    count of 159,571 matches the Kaggle Jigsaw spec.
    """
    if not jigsaw_csv.exists():
        raise FileNotFoundError(f"Jigsaw CSV not found at {jigsaw_csv}.")

    import pandas as pd
    df = pd.read_csv(jigsaw_csv, engine="python", quoting=csv.QUOTE_ALL)
    tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"] = df[tox_cols].any(axis=1).astype(int)
    texts  = df["comment_text"].tolist()
    labels = df["label"].tolist()
    print(f"  Jigsaw rows: {len(texts):,}  toxic ratio: {sum(labels)/len(labels):.3f}")

    rng = random.Random(seed)
    idxs = list(range(len(texts)))
    rng.shuffle(idxs)
    n_val = int(len(texts) * val_fraction)
    val_idx, train_idx = idxs[:n_val], idxs[n_val:]
    return (
        [texts[i]  for i in train_idx],
        [labels[i] for i in train_idx],
        [texts[i]  for i in val_idx],
        [labels[i] for i in val_idx],
    )


# ── hidden state extraction (last-layer residual, attn-mask weighted mean) ────

@torch.no_grad()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns the attention-mask-weighted mean of hidden_states[-1] over each text.

    hidden_states[-1] for GPT2LMHeadModel is the OUTPUT of the final transformer
    block (after final layernorm in GPT2), shape (B, T, 1024). This is the
    residual stream just before lm_head — i.e., the input to the unembedding.

    Mask-weighted pool excludes pad positions; pad positions in causal-LM
    hidden states still carry information from earlier tokens, but Lee et al.
    say "averaged across all timesteps" which is ambiguous. We use mask-
    weighted to match our own train_probe.py for parity with the GPT-Neo
    diagnostic. If the resulting probe fails the logit-lens test, an unweighted
    mean would be the next variant to try.
    """
    model.eval()
    all_reps = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1].float()       # (B, T, 1024)
        mask_f      = attention_mask.unsqueeze(-1).float()    # (B, T, 1)
        x_bar       = (last_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        all_reps.append(x_bar.cpu())

        if (start // batch_size) % 50 == 0:
            print(f"    extracted {start + len(batch_texts)}/{n}", end="\r")

    print(f"    extracted {n}/{n}    ")
    return torch.cat(all_reps, dim=0)   # (N, 1024)


# ── probe training (no-bias variant to match Lee's probe.pt structure) ────────

def train_probe(
    train_x: torch.Tensor,
    train_y: list[int],
    val_x: torch.Tensor,
    val_y: list[int],
    n_epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cuda",
    fit_bias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, float, float]:
    """
    Trains a linear probe sigmoid(W . x), W in R^d, optional bias.

    fit_bias=False matches Lee et al.'s probe.pt which is a bare torch.Tensor
    of shape (1024,) with no separate bias attribute. AdamW with pos_weight
    class balancing handles the ~10% toxic class imbalance.

    Returns (W, b, accuracy, threshold). b is None if fit_bias=False.
    """
    y_train = torch.tensor(train_y, dtype=torch.float32, device=device)
    y_val   = torch.tensor(val_y,   dtype=torch.float32, device=device)
    x_train = train_x.to(device)
    x_val   = val_x.to(device)

    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"  Class balance: pos={int(n_pos)}, neg={int(n_neg)}, pos_weight={pos_weight.item():.2f}")
    print(f"  fit_bias={fit_bias}")

    probe = nn.Linear(HIDDEN, 1, bias=fit_bias).to(device)
    nn.init.xavier_uniform_(probe.weight)
    if fit_bias:
        nn.init.zeros_(probe.bias)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    batch_size = 512
    for epoch in range(n_epochs):
        probe.train()
        perm = torch.randperm(len(x_train), device=device)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(x_train), batch_size):
            idx = perm[i : i + batch_size]
            logits = probe(x_train[idx]).squeeze(-1)
            loss   = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        probe.eval()
        with torch.no_grad():
            val_logits = probe(x_val).squeeze(-1)
            val_loss   = criterion(val_logits, y_val).item()
        val_probs = torch.sigmoid(val_logits).cpu()
        acc_at_05 = ((val_probs > 0.5).int() == y_val.cpu().int()).float().mean().item()
        print(f"  Epoch {epoch+1}/{n_epochs}  "
              f"train_loss={total_loss/n_batches:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"acc@0.5={acc_at_05:.4f}")

    probe.eval()
    with torch.no_grad():
        val_logits = probe(x_val).squeeze(-1).cpu()
    val_probs = torch.sigmoid(val_logits)
    y_val_np  = y_val.cpu()

    best_acc, best_thresh = 0.0, 0.5
    for t in torch.linspace(0.1, 0.9, 81):
        preds = (val_probs > t.item()).int()
        acc   = (preds == y_val_np.int()).float().mean().item()
        if acc > best_acc:
            best_acc, best_thresh = acc, t.item()
    print(f"  Optimal threshold: {best_thresh:.3f}  val_accuracy: {best_acc:.4f}")

    W = probe.weight.squeeze(0).detach().cpu()
    b = probe.bias.detach().cpu() if fit_bias else None
    return W, b, best_acc, best_thresh


# ── logit lens (probe → top-30 vocab) ─────────────────────────────────────────

def logit_lens_top_k(
    probe_weight: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    top_k: int = 30,
) -> list[tuple[int, str, float]]:
    """
    Projects probe weight through the unembedding matrix (lm_head.weight) and
    returns the top-K (token_id, token_str, score) tuples.

    For GPT2-medium, lm_head.weight is shape (50257, 1024). The probe weight
    is a (1024,) direction in residual-stream space; tokens with the largest
    E . W are those the residual stream's "toxicity component" most promotes.
    """
    E = model.lm_head.weight.detach().float().cpu()   # (vocab, 1024)
    scores = E @ probe_weight.cpu().float()           # (vocab,)
    top_vals, top_ids = scores.topk(top_k)
    return [
        (int(tid.item()), tokenizer.decode([int(tid.item())]), float(sc.item()))
        for sc, tid in zip(top_vals, top_ids)
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train toxicity probe on GPT2-medium (Lee et al. 2024 reproduction)."
    )
    parser.add_argument("--jigsaw-csv", type=Path, default=Path("data/jigsaw/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/probe"))
    parser.add_argument("--device",     type=str,  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int,  default=16)
    parser.add_argument("--max-length", type=int,  default=256)
    parser.add_argument("--epochs",     type=int,  default=3)
    parser.add_argument("--fit-bias",   action="store_true",
                        help="Fit a bias term (default: False, to match Lee's probe.pt structure)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "probe_gpt2medium.pt"

    print(f"=== Reproduce Lee et al. 2024 probe on {MODEL_NAME} ===")
    print(f"Device: {args.device}")

    # Load GPT2-medium
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()
    print(f"  {MODEL_NAME} loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  config: n_layer={model.config.n_layer}, hidden_size={model.config.hidden_size}")
    assert model.config.hidden_size == HIDDEN, f"unexpected hidden_size {model.config.hidden_size}"

    # Load Jigsaw
    print("\nLoading Jigsaw...")
    tr_texts, tr_labels, vl_texts, vl_labels = load_jigsaw(args.jigsaw_csv)
    print(f"  Train: {len(tr_texts):,}  Val: {len(vl_texts):,}")

    # Extract hidden states
    print("\nExtracting training hidden states from last layer...")
    train_x = extract_hidden_states(
        model, tokenizer, tr_texts,
        batch_size=args.batch_size, max_length=args.max_length, device=args.device,
    )
    print("Extracting validation hidden states...")
    val_x = extract_hidden_states(
        model, tokenizer, vl_texts,
        batch_size=args.batch_size, max_length=args.max_length, device=args.device,
    )
    print(f"  train_x: {train_x.shape}  val_x: {val_x.shape}")

    # Train probe
    print("\nTraining probe...")
    W, b, accuracy, threshold = train_probe(
        train_x, tr_labels, val_x, vl_labels,
        n_epochs=args.epochs, device=args.device,
        fit_bias=args.fit_bias,
    )

    # Save probe
    payload = {
        "weight":     W,                  # shape (1024,)
        "bias":       b,                  # None unless --fit-bias
        "model_name": MODEL_NAME,
        "accuracy":   accuracy,
        "threshold":  threshold,
        "val_n":      len(vl_labels),
        "fit_bias":   args.fit_bias,
        "layer":      "last (hidden_states[-1])",
    }
    torch.save(payload, out_path)
    print(f"\nProbe saved: {out_path}")
    print(f"  weight shape: {W.shape}  norm: {W.norm():.4f}")
    print(f"  accuracy: {accuracy:.4f}  (Lee et al. reported: 0.94)")

    # Logit lens — the key diagnostic
    print(f"\n=== Logit lens (E @ W) — top-30 promoted tokens ===")
    top30 = logit_lens_top_k(W, model, tokenizer, top_k=30)
    for rank, (tid, tok, sc) in enumerate(top30, 1):
        print(f"  [{rank:2d}]  id={tid:>5}  token={tok!r:<30}  score={sc:+.4f}")

    # Save logit-lens results
    lens_path = args.output_dir / "probe_gpt2medium_logit_lens.json"
    with open(lens_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "probe_path": str(out_path),
            "accuracy": accuracy,
            "threshold": threshold,
            "top30": [
                {"rank": r, "token_id": tid, "token": tok, "score": sc}
                for r, (tid, tok, sc) in enumerate(top30, 1)
            ],
        }, f, indent=2)
    print(f"\nLogit-lens results saved: {lens_path}")


if __name__ == "__main__":
    main()
