"""
Train a linear toxicity probe on GPT-Neo-125m and replicate Lee et al.'s
toxic-value-vector analysis on the new base model.

Probe architecture:
  logit = (x_bar @ W) + b
  x_bar = attention-mask-weighted mean of last hidden state (shape: N x 768)
  W: probe weight (shape: 768), b: scalar bias

Loss: BCEWithLogitsLoss with pos_weight for 10:1 class imbalance.
Data: data/jigsaw/train.csv, 90:10 split, OR-fold of 6 toxicity columns.
Target: >= 90% accuracy on validation at optimal threshold.

Output: data/probe/probe_gpt_neo_125m.pt
  {weight, bias, model_name, accuracy, threshold, val_n}

Cosine ranking: for each of 12 layers x 3072 neurons, compute
  cosine_sim(W_probe, c_proj.weight[:, i])
Value vector i for GPT-Neo (nn.Linear): c_proj.weight[:, i], shape (768,).
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/gpt-neo-125m"
N_LAYERS   = 12
HIDDEN     = 768
INTER      = 3072
SMOKE_N_TRAIN = 2000
SMOKE_N_VAL   = 200


# ── data loading ──────────────────────────────────────────────────────────────

def load_jigsaw(
    jigsaw_csv: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """
    Returns (train_texts, train_labels, val_texts, val_labels).
    Binary label: OR-fold of all 6 toxicity columns.
    Handles embedded newlines via csv.QUOTE_ALL / Python engine via pandas fallback.
    """
    if not jigsaw_csv.exists():
        raise FileNotFoundError(
            f"Jigsaw CSV not found at {jigsaw_csv}. "
            "Place data/jigsaw/train.csv before running."
        )

    import pandas as pd
    df = pd.read_csv(jigsaw_csv, engine="python", quoting=csv.QUOTE_ALL)
    tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"] = df[tox_cols].any(axis=1).astype(int)
    texts  = df["comment_text"].tolist()
    labels = df["label"].tolist()
    print(f"  Jigsaw rows: {len(texts):,}  toxic ratio: {sum(labels)/len(labels):.3f}")

    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    n_val = int(len(texts) * val_fraction)
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    train_texts  = [texts[i]  for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts    = [texts[i]  for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    print(f"  Train: {len(train_texts):,}  Val: {len(val_texts):,}")
    return train_texts, train_labels, val_texts, val_labels


# ── hidden state extraction ───────────────────────────────────────────────────

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
    Returns mean-pooled last-layer hidden states, shape (N, 768).
    Mean-pooling is attention_mask-weighted to exclude pad positions.
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
        # hidden_states[-1]: shape (batch, seq_len, hidden_size)
        last_hidden = outputs.hidden_states[-1].float()

        # Attention-mask-weighted mean pool (excludes pad tokens)
        mask_f = attention_mask.unsqueeze(-1).float()           # (B, T, 1)
        x_bar  = (last_hidden * mask_f).sum(dim=1)              # (B, H)
        x_bar  = x_bar / mask_f.sum(dim=1).clamp(min=1e-9)     # (B, H)
        all_reps.append(x_bar.cpu())

        if (start // batch_size) % 50 == 0:
            print(f"    extracted {start + len(batch_texts)}/{n}", end="\r")

    print(f"    extracted {n}/{n}    ")
    return torch.cat(all_reps, dim=0)  # (N, H)


# ── probe training ────────────────────────────────────────────────────────────

def train_probe(
    train_x: torch.Tensor,
    train_y: list[int],
    val_x: torch.Tensor,
    val_y: list[int],
    n_epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns (probe_weight [768], probe_bias [1], val_accuracy_at_optimal_threshold).
    """
    y_train = torch.tensor(train_y, dtype=torch.float32, device=device)
    y_val   = torch.tensor(val_y,   dtype=torch.float32, device=device)
    x_train = train_x.to(device)
    x_val   = val_x.to(device)

    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"  Class balance: pos={int(n_pos)}, neg={int(n_neg)}, pos_weight={pos_weight.item():.2f}")

    probe = nn.Linear(HIDDEN, 1).to(device)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    batch_size = 512
    for epoch in range(n_epochs):
        probe.train()
        perm = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, len(x_train), batch_size):
            idx = perm[i : i + batch_size]
            logits = probe(x_train[idx]).squeeze(-1)
            loss   = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(x_val).squeeze(-1)
            val_loss   = criterion(val_logits, y_val).item()

        val_probs = torch.sigmoid(val_logits).cpu()
        acc_at_0  = ((val_probs > 0.5).int() == y_val.cpu().int()).float().mean().item()
        print(f"  Epoch {epoch+1}/{n_epochs}  "
              f"train_loss={total_loss/n_batches:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"acc@threshold=0: {acc_at_0:.4f}")

    # Find optimal threshold by grid search on validation
    probe.eval()
    with torch.no_grad():
        val_logits = probe(x_val).squeeze(-1).cpu()
    val_probs  = torch.sigmoid(val_logits)
    y_val_np   = y_val.cpu()

    best_acc   = 0.0
    best_thresh = 0.5
    for thresh in torch.linspace(0.1, 0.9, 81):
        preds = (val_probs > thresh.item()).int()
        acc   = (preds == y_val_np.int()).float().mean().item()
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh.item()

    print(f"  Optimal threshold: {best_thresh:.3f}  val_accuracy: {best_acc:.4f}")

    W = probe.weight.squeeze(0).detach().cpu()   # (768,)
    b = probe.bias.detach().cpu()                 # (1,)
    return W, b, best_acc, best_thresh


# ── cosine ranking ────────────────────────────────────────────────────────────

def rank_value_vecs(
    model: AutoModelForCausalLM,
    probe_weight: torch.Tensor,
    top_k: int = 10,
) -> list[tuple[float, int, int]]:
    """
    Returns sorted list of (cosine_sim, layer_idx, neuron_idx), descending.

    For GPT-Neo (nn.Linear): value_vector_i = c_proj.weight[:, i], shape (768,).
    c_proj.weight shape: (768, 3072), so value_vectors = c_proj.weight.T, shape (3072, 768).
    """
    device = next(model.parameters()).device
    probe_norm = F.normalize(probe_weight.to(device).unsqueeze(0), dim=-1)  # (1, 768)
    results = []

    for layer_idx, block in enumerate(model.transformer.h):
        w = block.mlp.c_proj.weight.detach().float()  # (768, 3072)
        # value_vectors[i] = w[:, i] → stack as w.T, shape (3072, 768)
        value_vecs = w.T                               # (3072, 768)
        vv_norm    = F.normalize(value_vecs, dim=-1)   # (3072, 768)
        cos_sims   = (vv_norm @ probe_norm.T).squeeze(-1)  # (3072,)

        top_vals, top_idxs = cos_sims.topk(top_k)
        for cos, nidx in zip(top_vals.tolist(), top_idxs.tolist()):
            results.append((cos, layer_idx, nidx))

    results.sort(key=lambda x: -x[0])
    return results


# ── vocabulary projection ─────────────────────────────────────────────────────

def project_to_vocab(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    neuron_idx: int,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Projects value vector for (layer_idx, neuron_idx) through LM head.
    GPT-Neo uses weight tying: lm_head.weight = transformer.wte.weight, shape (vocab, 768).
    """
    w_value = (
        model.transformer.h[layer_idx].mlp.c_proj.weight[:, neuron_idx]
        .detach().float()
    )  # (768,)
    E = model.lm_head.weight.detach().float()   # (vocab_size, 768)
    scores = E @ w_value                         # (vocab_size,)
    top_scores, top_ids = scores.topk(top_k)
    return [
        (tokenizer.decode([tid.item()]), sc.item())
        for tid, sc in zip(top_ids, top_scores)
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train toxicity probe on GPT-Neo-125m and run cosine ranking."
    )
    parser.add_argument("--jigsaw-csv", type=Path,
                        default=Path("data/jigsaw/train.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/probe"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true",
                        help=f"Quick test on {SMOKE_N_TRAIN} train / {SMOKE_N_VAL} val samples.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = args.output_dir / f"probe_gpt_neo_125m{suffix}.pt"

    print(f"=== Train probe on {MODEL_NAME} ===")
    print(f"Device: {args.device}  Smoke: {args.smoke}")

    # Load model + tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-Neo has no pad token by default; set to EOS
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()
    print(f"  {MODEL_NAME} loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load Jigsaw data
    print("\nLoading Jigsaw...")
    tr_texts, tr_labels, vl_texts, vl_labels = load_jigsaw(args.jigsaw_csv)

    if args.smoke:
        rng = random.Random(42)
        tr_idx = rng.sample(range(len(tr_texts)), min(SMOKE_N_TRAIN, len(tr_texts)))
        vl_idx = rng.sample(range(len(vl_texts)), min(SMOKE_N_VAL, len(vl_texts)))
        tr_texts  = [tr_texts[i]  for i in tr_idx]
        tr_labels = [tr_labels[i] for i in tr_idx]
        vl_texts  = [vl_texts[i]  for i in vl_idx]
        vl_labels = [vl_labels[i] for i in vl_idx]
        print(f"  Smoke: using {len(tr_texts)} train, {len(vl_texts)} val samples")

    # Extract hidden states
    print("\nExtracting training hidden states...")
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
    )

    # Save probe
    payload = {
        "weight":     W,
        "bias":       b,
        "model_name": MODEL_NAME,
        "accuracy":   accuracy,
        "threshold":  threshold,
        "val_n":      len(vl_labels),
    }
    torch.save(payload, out_path)
    print(f"\nProbe saved: {out_path}  (accuracy={accuracy:.4f})")

    if accuracy < 0.90:
        print(f"  WARN: accuracy={accuracy:.4f} < 0.90 threshold")
    else:
        print(f"  OK: accuracy >= 0.90")

    # Cosine ranking
    print(f"\n=== Cosine ranking: probe vs MLP value vectors ===")
    print(f"  12 layers x 3072 neurons = 36,864 vectors")
    top_all = rank_value_vecs(model, W, top_k=10)
    top10_global = top_all[:10]

    print(f"\nTop-10 (layer, neuron) by cosine similarity to probe weight:")
    print(f"  {'rank':>4}  {'cos_sim':>8}  {'layer':>5}  {'neuron':>6}")
    print("  " + "-" * 35)
    for rank, (cos, layer, neuron) in enumerate(top10_global, 1):
        print(f"  {rank:>4}  {cos:>8.4f}  {layer:>5}  {neuron:>6}")

    # Vocabulary projection for top-7 vectors
    print(f"\n=== Vocabulary projections for top-7 value vectors ===")
    for rank, (cos, layer, neuron) in enumerate(top10_global[:7], 1):
        top_tokens = project_to_vocab(model, tokenizer, layer, neuron, top_k=10)
        token_strs = [f"'{t}'" for t, _ in top_tokens]
        print(f"  [{rank}] L{layer}N{neuron} (cos={cos:.4f}): {', '.join(token_strs)}")

    # Acceptance check: at least 3 of top-7 should have profanity in top-10 tokens
    profanity_keywords = {
        "shit", "ass", "crap", "fuck", "bitch", "dick", "cock", "damn",
        "bastard", "asshole", "cunt", "whore", "faggot", "nigger",
        "fucking", "bullshit", "moron", "idiot", "stupid", "retard",
    }
    n_with_profanity = 0
    for cos, layer, neuron in top10_global[:7]:
        top_tokens = project_to_vocab(model, tokenizer, layer, neuron, top_k=10)
        token_words = {t.strip().lower() for t, _ in top_tokens}
        if token_words & profanity_keywords:
            n_with_profanity += 1

    print(f"\n  Vectors with profanity in top-10 tokens (of top-7): {n_with_profanity}/7")
    if n_with_profanity >= 3:
        print("  OK: >= 3 vectors have profanity tokens (acceptance criterion met)")
    else:
        print("  WARN: < 3 vectors with profanity tokens")

    # Save ranking results
    results = {
        "model_name":   MODEL_NAME,
        "probe_path":   str(out_path),
        "accuracy":     accuracy,
        "threshold":    threshold,
        "top10_vectors": [
            {"rank": r+1, "cos_sim": cos, "layer": layer, "neuron": neuron}
            for r, (cos, layer, neuron) in enumerate(top10_global)
        ],
    }
    results_path = args.output_dir / f"probe_analysis_gpt_neo_125m{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
