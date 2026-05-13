"""
Full diagnostic for trained GPT-Neo-125m toxicity probe.

Computes:
  - Accuracy at threshold=0.5 (sign of logit)
  - Accuracy at trained optimal threshold
  - AUC-ROC
  - F1, precision, recall at optimal threshold
  - Confusion matrix counts

  - Full cosine ranking: 12 layers x 3072 neurons = 36,864 vectors
  - Top-30 vocabulary projections (top-10 promoted tokens each)
  - Classification: clearly-toxic / partially-toxic / not-toxic per Lee et al. tier

Usage:
  python3 tests/diag_probe_analysis.py [--probe data/probe/probe_gpt_neo_125m.pt]
"""

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/gpt-neo-125m"

# Root substrings that count as profanity — tokenizer produces fragments with
# leading spaces, so we strip and lower before checking.
PROFANITY_ROOTS = {
    "shit", "ass", "crap", "fuck", "bitch", "dick", "cock", "damn",
    "bastard", "cunt", "whore", "fag", "nigger", "nigga", "slut",
    "retard", "moron", "idiot", "asshole", "bullshit", "motherfuck",
    "piss", "twat", "wanker", "skank",
}


def is_profanity(token: str) -> bool:
    t = token.strip().lower()
    return any(root in t for root in PROFANITY_ROOTS)


def load_jigsaw_val(jigsaw_csv: Path, val_fraction: float = 0.1, seed: int = 42):
    import pandas as pd
    df = pd.read_csv(jigsaw_csv, engine="python", quoting=csv.QUOTE_ALL)
    tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"] = df[tox_cols].any(axis=1).astype(int)
    texts  = df["comment_text"].tolist()
    labels = df["label"].tolist()

    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    n_val = int(len(texts) * val_fraction)
    val_idx = indices[:n_val]
    return [texts[i] for i in val_idx], [labels[i] for i in val_idx]


@torch.no_grad()
def extract_hidden_states(model, tokenizer, texts, batch_size=16, max_length=256, device="cuda"):
    model.eval()
    all_reps = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out  = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        last = out.hidden_states[-1].float()                        # (B, T, H)
        mask_f = mask.unsqueeze(-1).float()
        x_bar  = (last * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        all_reps.append(x_bar.cpu())
        if (start // batch_size) % 50 == 0:
            print(f"  extracted {start + len(batch)}/{n}", end="\r")
    print(f"  extracted {n}/{n}    ")
    return torch.cat(all_reps, dim=0)


def classification_report_block(y_true, y_pred_binary, y_score, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    auc    = roc_auc_score(y_true, y_score)
    f1     = f1_score(y_true, y_pred_binary)
    prec   = precision_score(y_true, y_pred_binary)
    rec    = recall_score(y_true, y_pred_binary)
    n_pos  = sum(y_true)
    n_neg  = len(y_true) - n_pos
    acc    = (tp + tn) / len(y_true)
    naive  = n_neg / len(y_true)

    print(f"\n=== Probe classification metrics ===")
    print(f"  Threshold used       : {threshold:.4f} (trained optimal)")
    print(f"  Accuracy at threshold: {acc:.4f}  (trivial baseline: {naive:.4f})")
    acc_at_half = ((torch.tensor(y_score) > 0.5).int().numpy() == y_true).mean()
    print(f"  Accuracy at 0.5      : {acc_at_half:.4f}")
    print(f"  AUC-ROC              : {auc:.4f}")
    print(f"  F1 score             : {f1:.4f}")
    print(f"  Precision            : {prec:.4f}")
    print(f"  Recall               : {rec:.4f}")
    print(f"  Confusion matrix     : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Val set size         : {len(y_true)} (pos={n_pos}, neg={n_neg})")

    gate_pass = not (acc <= 0.91 and auc < 0.95)
    if gate_pass:
        print(f"  GATE: PASS (accuracy > 0.91 OR AUC-ROC >= 0.95)")
    else:
        print(f"  GATE: FAIL — accuracy={acc:.4f} ≤ 0.91 AND AUC-ROC={auc:.4f} < 0.95")
        print(f"  → probe has NOT learned toxicity direction; debug before proceeding")
    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec,
            "accuracy": acc, "gate_pass": gate_pass,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def rank_all_value_vecs(model, probe_weight, device):
    probe_norm = F.normalize(probe_weight.to(device).unsqueeze(0), dim=-1)  # (1, H)
    results = []
    for layer_idx, block in enumerate(model.transformer.h):
        w = block.mlp.c_proj.weight.detach().float()   # (768, 3072) for GPT-Neo
        vv_norm = F.normalize(w.T, dim=-1)              # (3072, 768) — value_vec_i = w[:, i]
        cos = (vv_norm @ probe_norm.T).squeeze(-1)      # (3072,)
        for nidx in range(cos.shape[0]):
            results.append((cos[nidx].item(), layer_idx, nidx))
    results.sort(key=lambda x: -x[0])
    return results


def project_to_vocab(model, tokenizer, layer_idx, neuron_idx, top_k=10):
    v = model.transformer.h[layer_idx].mlp.c_proj.weight[:, neuron_idx].detach().float()
    E = model.lm_head.weight.detach().float()   # (vocab, H)
    scores = E @ v
    top_s, top_ids = scores.topk(top_k)
    return [(tokenizer.decode([tid.item()]), sc.item())
            for tid, sc in zip(top_ids, top_s)]


def classify_vector(top_tokens: list[tuple[str, float]]) -> str:
    n_profanity = sum(1 for tok, _ in top_tokens if is_profanity(tok))
    if n_profanity >= 5:
        return "clearly-toxic"
    elif n_profanity >= 2:
        return "partially-toxic"
    else:
        return "not-toxic"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path,
                        default=Path("data/probe/probe_gpt_neo_125m.pt"))
    parser.add_argument("--jigsaw-csv", type=Path,
                        default=Path("data/jigsaw/train.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    if not args.probe.exists():
        print(f"ERROR: probe not found at {args.probe}")
        raise SystemExit(1)

    # Load probe
    payload = torch.load(args.probe, map_location="cpu")
    W         = payload["weight"]        # (768,)
    b         = payload["bias"]          # (1,)
    threshold = payload["threshold"]
    print(f"Loaded probe: accuracy={payload['accuracy']:.4f}  threshold={threshold:.4f}")
    print(f"  val_n in probe file: {payload.get('val_n', '?')}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()

    # Load validation data
    print("\nLoading Jigsaw validation split...")
    val_texts, val_labels = load_jigsaw_val(args.jigsaw_csv)
    print(f"  val size: {len(val_texts)}")

    # Extract hidden states
    print("\nExtracting validation hidden states...")
    val_x = extract_hidden_states(
        model, tokenizer, val_texts,
        batch_size=args.batch_size, device=args.device,
    )
    print(f"  val_x: {val_x.shape}")

    # Score with probe
    W_dev = W.to(args.device)
    b_dev = b.to(args.device)
    with torch.no_grad():
        logits = (val_x.to(args.device) @ W_dev) + b_dev.squeeze()  # (N,)
        probs  = torch.sigmoid(logits).cpu().numpy()

    y_true   = [int(l) for l in val_labels]
    y_pred   = (probs > threshold).astype(int)
    metrics  = classification_report_block(y_true, y_pred, probs, threshold)

    if not metrics["gate_pass"]:
        print("\nHalting — probe gate failed. Debug before continuing.")
        raise SystemExit(1)

    # Full cosine ranking
    print(f"\n=== Cosine ranking: probe vs all 36,864 value vectors ===")
    all_ranked = rank_all_value_vecs(model, W, args.device)
    top_n = all_ranked[: args.top_n]

    print(f"\nTop-{args.top_n} (layer, neuron) by cosine similarity:")
    print(f"  {'rank':>4}  {'cos_sim':>8}  {'layer':>5}  {'neuron':>6}  {'category':<16}  top-10 tokens")
    print("  " + "-" * 90)

    category_counts = {"clearly-toxic": 0, "partially-toxic": 0, "not-toxic": 0}
    target_vector   = None

    for rank, (cos, layer, neuron) in enumerate(top_n, 1):
        top_tokens = project_to_vocab(model, tokenizer, layer, neuron, top_k=10)
        category   = classify_vector(top_tokens)
        category_counts[category] += 1
        token_strs = [f"'{t.strip()}'" for t, _ in top_tokens]
        print(f"  {rank:>4}  {cos:>8.4f}  {layer:>5}  {neuron:>6}  {category:<16}  "
              f"{', '.join(token_strs)}")

        if target_vector is None and category == "clearly-toxic":
            target_vector = (rank, cos, layer, neuron)

    # Summary
    print(f"\n=== Top-{args.top_n} category distribution ===")
    for cat, cnt in category_counts.items():
        print(f"  {cat:<20} : {cnt:>3} / {args.top_n}")

    print(f"\n  Cosine similarity range (top-{args.top_n}): "
          f"{top_n[-1][0]:.4f} – {top_n[0][0]:.4f}")

    # Target vector verdict
    print(f"\n=== Target vector verdict (Phase 2/3 parameter subspace φ) ===")
    if target_vector is not None:
        rank, cos, layer, neuron = target_vector
        print(f"  FOUND clearly-toxic vector at rank {rank}: "
              f"(layer={layer}, neuron={neuron})  cos={cos:.4f}")
        print(f"  → Phase 2/3 will use c_proj.weight[:, {neuron}] of layer {layer} as W_2 target.")
        print(f"  → Proceed: structure analog to Lee et al.'s (layer=19, neuron=770) found.")
    else:
        top3_cats = [classify_vector(
            project_to_vocab(model, tokenizer, l, n, top_k=10)
        ) for _, l, n in [top_n[i][:3] for i in range(min(3, len(top_n)))]]
        print(f"  NO clearly-toxic vector in top-{args.top_n}.")
        print(f"  Top-3 categories: {top3_cats}")
        print(f"  → SURFACE to user: may need multi-vector φ or project reframing.")

    # Full vocab projections for all top-30 (already printed inline above)
    print(f"\n(Full top-10 token projections shown in the ranking table above.)")


if __name__ == "__main__":
    main()
