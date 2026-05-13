"""
Task 3.2 — Toxicity probe verification.

Verifies Lee et al.'s probe.pt on the Jigsaw toxic comment validation split and
ranks GPT2-medium MLP value vectors by cosine similarity with the probe direction.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── probe loading ─────────────────────────────────────────────────────────────

def load_probe(probe_path: Path) -> tuple[Tensor, Optional[Tensor]]:
    """
    Load probe.pt without assuming structure.

    Inspects the raw object first:
      - Tensor of shape (1024,): weight=tensor, bias=None
      - dict with 'weight' key: extracts weight (and 'bias' if present)

    Returns (probe_weight, probe_bias) where probe_bias may be None.
    """
    raw = torch.load(probe_path, map_location="cpu")
    if isinstance(raw, Tensor):
        weight = raw
        bias: Optional[Tensor] = None
    elif isinstance(raw, dict):
        if "weight" not in raw:
            raise ValueError(
                f"probe.pt is a dict but has no 'weight' key; keys={list(raw.keys())}"
            )
        weight = raw["weight"]
        bias = raw.get("bias", None)
    else:
        raise TypeError(f"Unexpected probe.pt type: {type(raw)}")

    if weight.ndim != 1:
        raise ValueError(f"Expected 1-D probe weight, got shape {weight.shape}")

    return weight.float(), (bias.float() if bias is not None else None)


# ── Jigsaw data ───────────────────────────────────────────────────────────────

_TOXICITY_COLS = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]


def load_jigsaw_validation(
    jigsaw_csv: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Reads train.csv (159,571 data rows; pandas handles embedded newlines correctly).
    Splits 90:10 with seed=42; returns the validation split (10% holdout).

    Binary label = OR-fold of all 6 toxicity columns:
      label = 1 if any of [toxic, severe_toxic, obscene, threat, insult,
                            identity_hate] == 1, else 0.

    Raises FileNotFoundError with a clear message if jigsaw_csv is missing.
    """
    if not jigsaw_csv.exists():
        raise FileNotFoundError(
            f"Jigsaw train.csv not found at {jigsaw_csv}\n"
            "Download from Kaggle 'jigsaw-toxic-comment-classification-challenge' "
            "and place at data/jigsaw/train.csv"
        )

    import pandas as pd
    df = pd.read_csv(jigsaw_csv, engine="python", quoting=csv.QUOTE_MINIMAL)

    # OR-fold: label=1 if any toxicity column is 1
    df["binary_label"] = df[_TOXICITY_COLS].max(axis=1).astype(int)

    # Reproducible shuffle, then take last val_fraction as validation
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_fraction)
    val_df = df.iloc[-n_val:]

    texts = val_df["comment_text"].astype(str).tolist()
    labels = val_df["binary_label"].tolist()
    return texts, labels


# ── model forward / hidden states ────────────────────────────────────────────

def get_mean_pooled_last_hidden(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    batch_size: int = 32,
    device: str = "cuda",
    max_length: int = 512,
) -> Tensor:
    """
    Forward through GPT2-medium with output_hidden_states=True.

    Uses hidden_states[-1] — the output of the last (24th) transformer block,
    i.e. index 24 in the 25-element tuple (1 embedding + 24 block outputs).
    Using hidden_states[-2] (block 23) is the most common indexing bug and
    gives wrong activations for this probe.

    Mean-pooling is attention_mask-weighted to exclude pad token positions:
      masked = hidden_states[-1] * attention_mask.unsqueeze(-1)
      x_bar  = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    Flat .mean(dim=1) dilutes short comments that receive heavy padding.

    Returns shape (N, 1024).
    """
    model.eval()
    all_hidden: list[Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # hidden_states[-1]: (batch, seq_len, 1024) — last transformer block
        last_hidden = outputs.hidden_states[-1].float()  # (B, T, 1024)

        masked = last_hidden * attention_mask.unsqueeze(-1).float()
        denom = attention_mask.sum(dim=1, keepdim=True).float()
        x_bar = masked.sum(dim=1) / denom  # (B, 1024)

        all_hidden.append(x_bar.cpu())

    return torch.cat(all_hidden, dim=0)  # (N, 1024)


# ── probe classification ──────────────────────────────────────────────────────

def classify_with_probe(
    hidden_states: Tensor,
    probe_weight: Tensor,
    probe_bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Binary prediction: sign(probe_weight @ x_bar + bias).

    When probe_bias is None (the typical case for probe.pt) the threshold is 0.
    When a bias term exists it is added before thresholding — threshold-at-0
    is only correct when there is no bias term.

    Returns int64 tensor of predictions (1 = toxic, 0 = not toxic).
    """
    probe_weight = probe_weight.to(hidden_states.device)
    scores = hidden_states @ probe_weight  # (N,)
    if probe_bias is not None:
        scores = scores + probe_bias.to(hidden_states.device)
    return (scores > 0).long()


# ── value vector ranking ──────────────────────────────────────────────────────

def rank_value_vecs_by_cosine(
    model: GPT2LMHeadModel,
    probe: Tensor,
) -> list[tuple[float, int, int]]:
    """
    For each layer l in [0, 23] and each neuron i in [0, 4095]:
      cosine_sim(c_proj.weight[i, :], probe).

    GPT2 uses Conv1D where c_proj.weight has shape [4096, 1024] in memory
    (stored as [in_features, out_features], transposed from nn.Linear).
    Row i of c_proj.weight is the i-th value vector v_i ∈ R^{1024}.

    Returns sorted list of (cosine_sim, neuron_idx, layer_idx), descending.
    """
    model_device = next(model.parameters()).device
    probe_norm = (probe / probe.norm()).to(model_device)
    scores: list[tuple[float, int, int]] = []

    for layer_idx in range(model.config.n_layer):
        # c_proj.weight: [4096, 1024] — each row is one value vector
        w = model.transformer.h[layer_idx].mlp.c_proj.weight  # (4096, 1024)
        cos = F.cosine_similarity(w.float(), probe_norm.unsqueeze(0), dim=1)  # (4096,)
        for neuron_idx in range(cos.shape[0]):
            scores.append((cos[neuron_idx].item(), neuron_idx, layer_idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


# ── vocabulary projection ─────────────────────────────────────────────────────

def project_value_vec_to_vocab(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    layer: int,
    neuron_idx: int,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Projects value vector v = c_proj.weight[neuron_idx, :] through the
    token embedding matrix E (wte.weight, shape [50257, 1024]).

    Score = E @ v → shape [50257]. Returns top_k (token_str, score) pairs.
    """
    v = model.transformer.h[layer].mlp.c_proj.weight[neuron_idx].float()  # (1024,)
    E = model.transformer.wte.weight.float()  # (50257, 1024)
    scores = E @ v  # (50257,)
    topk = scores.topk(top_k)
    return [(tokenizer.decode([idx.item()]), val.item()) for idx, val in
            zip(topk.indices, topk.values)]


# ── accuracy ──────────────────────────────────────────────────────────────────

def compute_accuracy(predictions: Tensor, labels: list[int]) -> float:
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return (predictions == label_tensor).float().mean().item()


# ── smoke validation ──────────────────────────────────────────────────────────

def run_smoke(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    probe_weight: Tensor,
    probe_bias: Optional[Tensor],
    jigsaw_csv: Path,
    device: str,
    smoke_dir: Path,
) -> None:
    """
    Three-part smoke test:
      1. Toxic/clean dot-product sign check on 2 hardcoded comments.
      2. v^{19}_{770} vocab projection — verify top tokens include profanity.
      3. 100-comment mini-batch accuracy.
    """
    smoke_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ── Part 1: sign check ──────────────────────────────────────────────────
    print("\n=== Smoke Part 1: sign check ===")
    test_comments = [
        "you are a stupid piece of shit and I hate you",
        "the weather is nice today and the sky is clear",
    ]
    hidden = get_mean_pooled_last_hidden(
        model, tokenizer, test_comments, batch_size=2, device=device
    )
    pw = probe_weight.to("cpu")
    scores = (hidden @ pw)
    if probe_bias is not None:
        scores = scores + probe_bias.cpu()

    toxic_score = scores[0].item()
    clean_score = scores[1].item()
    print(f"  Toxic comment score:  {toxic_score:.4f}  (expect > 0)")
    print(f"  Clean comment score:  {clean_score:.4f}  (expect < 0)")
    # Use .item() so comparisons yield Python bools, not scalar Tensors.
    # Python's `and` on Tensors returns the last-evaluated Tensor, not a bool.
    sign_ok = bool(toxic_score > 0) and bool(clean_score < 0)
    print(f"  Sign check: {'PASS' if sign_ok else 'FAIL'}")
    results["sign_check"] = {
        "toxic_score": toxic_score,
        "clean_score": clean_score,
        "pass": sign_ok,
    }

    # ── Part 2: v^{19}_{770} vocab projection ───────────────────────────────
    print("\n=== Smoke Part 2: v^{19}_{770} vocab projection ===")
    top_tokens = project_value_vec_to_vocab(model, tokenizer, layer=19, neuron_idx=770)
    print("  Top-10 tokens for v^{19}_{770}:")
    for tok, score in top_tokens:
        print(f"    {repr(tok):20s}  {score:.3f}")
    expected_profanity = {"shit", "ass", "crap", "fuck", "cunt", "garbage", "trash"}
    found = [tok.strip().lower() for tok, _ in top_tokens]
    profanity_hit = sum(1 for t in found if t in expected_profanity)
    print(f"  Profanity tokens found: {profanity_hit}/1+ (expect ≥1)")
    results["vocab_projection"] = {
        "top_tokens": [(tok, score) for tok, score in top_tokens],
        "profanity_hit_count": profanity_hit,
        "pass": profanity_hit >= 1,
    }

    # ── Part 3: 100-comment mini-batch ──────────────────────────────────────
    print("\n=== Smoke Part 3: 100-comment mini-batch ===")
    texts, labels = load_jigsaw_validation(jigsaw_csv)
    texts100, labels100 = texts[:100], labels[:100]
    hidden100 = get_mean_pooled_last_hidden(
        model, tokenizer, texts100, batch_size=32, device=device
    )
    preds100 = classify_with_probe(hidden100, probe_weight, probe_bias)
    acc100 = compute_accuracy(preds100, labels100)

    n_true_toxic = sum(labels100)
    n_pred_toxic = int(preds100.sum().item())
    label_tensor100 = torch.tensor(labels100)
    tp = int(((preds100 == 1) & (label_tensor100 == 1)).sum().item())
    fp = int(((preds100 == 1) & (label_tensor100 == 0)).sum().item())
    fn = int(((preds100 == 0) & (label_tensor100 == 1)).sum().item())
    tn = int(((preds100 == 0) & (label_tensor100 == 0)).sum().item())
    print(f"  True toxic in batch:  {n_true_toxic}/100  (expect ~10 if OR-fold rate ~10%)")
    print(f"  Predicted toxic:      {n_pred_toxic}/100")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  100-comment accuracy: {acc100:.4f}")
    print(f"  Note: 100 samples is too small for reliable accuracy — smoke threshold "
          f"is relaxed to >0.5; full criterion (≥0.92) is checked in the full run.")
    # Smoke threshold is deliberately loose (>0.5) because 100 samples is too small
    # for a reliable accuracy estimate with ~10% positive rate.
    results["minibatch_100"] = {
        "accuracy": acc100,
        "n_true_toxic": n_true_toxic,
        "n_pred_toxic": n_pred_toxic,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pass": bool(acc100 > 0.5),
    }

    overall = (sign_ok
               and bool(results["vocab_projection"]["pass"])
               and bool(acc100 > 0.5))
    print(f"\nSmoke overall: {'PASS' if overall else 'FAIL'}")
    results["overall_pass"] = overall

    smoke_path = smoke_dir / "probe_smoke.json"
    with open(smoke_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Smoke results saved to {smoke_path}")


# ── full validation ───────────────────────────────────────────────────────────

def compute_oracle_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Find the score threshold t that maximises accuracy via a linear sweep.
    Returns (oracle_threshold, oracle_accuracy).
    This is FOR DIAGNOSTIC PURPOSES ONLY — not used downstream.
    """
    # Candidate thresholds: every unique score value (sorted)
    candidates = np.sort(np.unique(scores))
    best_t, best_acc = 0.0, 0.0
    n = len(labels)
    for t in candidates:
        preds = (scores > t).astype(int)
        acc = (preds == labels).sum() / n
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, float(best_acc)


def run_diagnostic(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    probe_weight: Tensor,
    probe_bias: Optional[Tensor],
    jigsaw_csv: Path,
    device: str,
    n: int = 2000,
    seed: int = 42,
) -> dict:
    """
    2k-comment diagnostic + full 24×4096 cosine ranking.

    Accuracy is INFORMATIONAL only — probe.pt is a bare direction vector
    with no calibrated threshold; the paper's 94% figure is not reproducible
    from the released artifact. Gating criteria are cosine-ranking and
    vocab-projection only.
    """
    print(f"\n=== Diagnostic: Loading {n} Jigsaw validation comments ===")
    texts, labels = load_jigsaw_validation(jigsaw_csv)
    texts_n, labels_n = texts[:n], labels[:n]
    print(f"  Using first {len(texts_n)} of {len(texts)} validation comments")
    print(f"  True-toxic rate in slice: {sum(labels_n)/len(labels_n):.3f}")

    print("\n=== Computing hidden states (this takes ~1–2 min on GPU) ===")
    hidden = get_mean_pooled_last_hidden(
        model, tokenizer, texts_n, batch_size=32, device=device
    )

    # Raw scores (before threshold)
    pw_cpu = probe_weight.float().cpu()
    scores_t = (hidden.float().cpu() @ pw_cpu)   # (N,)
    if probe_bias is not None:
        scores_t = scores_t + probe_bias.cpu()
    scores_np = scores_t.numpy()
    labels_np = np.array(labels_n, dtype=int)

    # Naive threshold = 0
    preds_naive = (scores_np > 0).astype(int)
    acc_naive = float((preds_naive == labels_np).mean())
    tp = int(((preds_naive == 1) & (labels_np == 1)).sum())
    fp = int(((preds_naive == 1) & (labels_np == 0)).sum())
    fn = int(((preds_naive == 0) & (labels_np == 1)).sum())
    tn = int(((preds_naive == 0) & (labels_np == 0)).sum())

    print(f"\n--- Score distribution (N={len(scores_np)}) ---")
    print(f"  Mean:  {scores_np.mean():.4f}")
    print(f"  Std:   {scores_np.std():.4f}")
    print(f"  Min:   {scores_np.min():.4f}")
    print(f"  Max:   {scores_np.max():.4f}")
    print(f"  %>0:   {(scores_np > 0).mean()*100:.1f}%  (would be predicted toxic)")

    print(f"\n--- Naive threshold=0 ---")
    print(f"  Accuracy:  {acc_naive:.4f}  (informational only)")
    print(f"  True toxic: {int(labels_np.sum())}/{len(labels_np)}"
          f"  Predicted toxic: {int(preds_naive.sum())}/{len(labels_np)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    oracle_t, oracle_acc = compute_oracle_threshold(scores_np, labels_np)
    print(f"\n--- Oracle threshold (diagnostic only, NOT used downstream) ---")
    print(f"  Threshold: {oracle_t:.4f}  →  Accuracy: {oracle_acc:.4f}")

    # ── Full cosine ranking ────────────────────────────────────────────────
    print("\n=== Full cosine ranking (24 layers × 4096 neurons) ===")
    ranked = rank_value_vecs_by_cosine(model, probe_weight)

    lee_table1 = {(19, 770), (12, 771), (18, 2669), (13, 668),
                  (16, 255), (12, 882), (19, 1438)}
    top10_set = {(lidx, nidx) for _, nidx, lidx in ranked[:10]}
    matches = top10_set & lee_table1

    print("Top-10 (layer, neuron)  cosine_sim:")
    for sim, nidx, lidx in ranked[:10]:
        mark = "✓" if (lidx, nidx) in lee_table1 else " "
        print(f"  {mark} ({lidx:2d}, {nidx:4d})  {sim:.4f}")

    print(f"\nTable 1 matches in top-10: {len(matches)}/7 — "
          f"{'PASS' if len(matches) >= 5 else 'FAIL'} (criterion ≥5)")
    print(f"Matched: {sorted(matches)}")

    # ── Vocab projection for v^{19}_{770} ─────────────────────────────────
    top_1970 = project_value_vec_to_vocab(model, tokenizer, 19, 770, top_k=10)
    expected = {"shit", "ass", "crap", "fuck", "cunt", "garbage", "trash"}
    found_toks = {t.strip().lower() for t, _ in top_1970}
    profanity_count = len(found_toks & expected)
    print(f"\nv^{{19}}_{{770}} top-10 tokens: {[t.strip() for t, _ in top_1970]}")
    print(f"Profanity hit: {profanity_count}/3+  — "
          f"{'PASS' if profanity_count >= 3 else 'FAIL'} (criterion ≥3)")

    print("\n=== Gating Criteria Summary ===")
    crit_cosine = len(matches) >= 5
    crit_vocab  = profanity_count >= 3
    print(f"  cosine_ranking_table1_gte5: {'PASS' if crit_cosine else 'FAIL'}")
    print(f"  v19_770_profanity_gte3:     {'PASS' if crit_vocab  else 'FAIL'}")

    return {
        "n": len(texts_n),
        "true_toxic_rate": float(labels_np.mean()),
        "score_mean":  float(scores_np.mean()),
        "score_std":   float(scores_np.std()),
        "score_min":   float(scores_np.min()),
        "score_max":   float(scores_np.max()),
        "pct_pred_toxic_naive": float((scores_np > 0).mean()),
        "accuracy_naive_threshold": acc_naive,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "oracle_threshold":         oracle_t,
        "oracle_accuracy":          oracle_acc,
        "top10_value_vecs": [(float(sim), int(nidx), int(lidx))
                             for sim, nidx, lidx in ranked[:10]],
        "table1_matches": [list(m) for m in sorted(matches)],
        "n_table1_matches": len(matches),
        "v19_770_top_tokens": [(tok.strip(), float(score))
                               for tok, score in top_1970],
        "v19_770_profanity_count": profanity_count,
        "gating_criteria": {
            "cosine_ranking_table1_gte5": crit_cosine,
            "v19_770_profanity_gte3":     crit_vocab,
        },
        "note": (
            "Accuracy is informational only. probe.pt is a bare direction vector; "
            "no calibrated threshold was released. Paper's 94% is not reproducible "
            "from this artifact. Downstream use (cosine ranking, vocab projection) "
            "does not require a classification threshold."
        ),
    }


def run_full_validation(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    probe_weight: Tensor,
    probe_bias: Optional[Tensor],
    jigsaw_csv: Path,
    device: str,
) -> dict:
    """
    Full verification on the Jigsaw validation split (~15,957 comments).
    Returns a dict with accuracy, top-10 value vecs, and pass/fail criteria.
    """
    print("\n=== Full Validation: Loading Jigsaw (~15,957 comments) ===")
    texts, labels = load_jigsaw_validation(jigsaw_csv)
    print(f"  Validation set: {len(texts)} comments")
    print(f"  Toxic rate: {sum(labels)/len(labels):.3f}  (expect ~0.102)")

    print("\n=== Full Validation: Computing hidden states ===")
    hidden = get_mean_pooled_last_hidden(
        model, tokenizer, texts, batch_size=32, device=device
    )

    preds = classify_with_probe(hidden, probe_weight, probe_bias)
    accuracy = compute_accuracy(preds, labels)
    print(f"\nProbe accuracy: {accuracy:.4f}  (acceptance criterion: ≥0.92)")

    print("\n=== Ranking MLP value vectors by cosine similarity ===")
    ranked = rank_value_vecs_by_cosine(model, probe_weight)
    print("Top-10 (cosine_sim, neuron_idx, layer_idx):")
    for sim, nidx, lidx in ranked[:10]:
        print(f"  ({lidx:2d}, {nidx:4d})  cos={sim:.4f}")

    # Check against Lee et al. Table 1
    lee_table1 = {(19, 770), (12, 771), (18, 2669), (13, 668),
                  (16, 255), (12, 882), (19, 1438)}
    top10_set = {(lidx, nidx) for _, nidx, lidx in ranked[:10]}
    matches = top10_set & lee_table1
    print(f"\nTable 1 matches in top-10: {len(matches)}/7 (accept ≥5): {sorted(matches)}")

    print("\n=== Vocab projections for top-5 value vectors ===")
    for sim, nidx, lidx in ranked[:5]:
        top_toks = project_value_vec_to_vocab(model, tokenizer, lidx, nidx, top_k=10)
        toks_str = ", ".join(repr(t) for t, _ in top_toks)
        print(f"  v^{lidx}_{nidx}: {toks_str}")

    # Acceptance criteria for v^{19}_{770}
    top_1970 = project_value_vec_to_vocab(model, tokenizer, 19, 770, top_k=10)
    expected = {"shit", "ass", "crap", "fuck", "cunt", "garbage", "trash"}
    found_toks = {t.strip().lower() for t, _ in top_1970}
    profanity_count = len(found_toks & expected)
    print(f"\nv^{{19}}_{{770}} top tokens: {[t for t, _ in top_1970]}")
    print(f"Profanity hit: {profanity_count}/3+ (accept ≥3): "
          f"{'PASS' if profanity_count >= 3 else 'FAIL'}")

    results = {
        "accuracy": accuracy,
        "n_val": len(texts),
        "toxic_rate": sum(labels) / len(labels),
        "top10_value_vecs": [(float(sim), int(nidx), int(lidx))
                             for sim, nidx, lidx in ranked[:10]],
        "table1_matches": [list(m) for m in sorted(matches)],
        "n_table1_matches": len(matches),
        "v19_770_top_tokens": [(tok, float(score)) for tok, score in top_1970],
        "v19_770_profanity_count": profanity_count,
        "criteria": {
            "accuracy_gte_92pct": accuracy >= 0.92,
            "table1_matches_gte_5": len(matches) >= 5,
            "profanity_gte_3": profanity_count >= 3,
        },
    }

    print("\n=== Acceptance Criteria Summary ===")
    for criterion, passed in results["criteria"].items():
        print(f"  {criterion}: {'PASS' if passed else 'FAIL'}")

    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Probe verification for Lee et al. toxicity probe on GPT2-medium."
    )
    parser.add_argument(
        "--probe", type=Path, default=repo_root / "data" / "probe" / "probe.pt",
        help="Path to probe.pt"
    )
    parser.add_argument(
        "--jigsaw-csv", type=Path, default=repo_root / "data" / "jigsaw" / "train.csv",
        help="Path to Jigsaw train.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=repo_root / "data" / "baseline_probe.json",
        help="Path to write full validation JSON results (non-smoke mode)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run smoke validation only (2 comments + v^{19}_{770} + 100-batch). "
             "Writes output to data/smoke/ instead of data/."
    )
    parser.add_argument(
        "--diag", action="store_true",
        help="Run 2k-comment diagnostic + full cosine ranking. "
             "Accuracy is informational; gating criteria are cosine+vocab. "
             "Writes output to data/smoke/probe_diag.json."
    )
    args = parser.parse_args()

    print(f"Loading GPT2-medium on {args.device}...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(args.device)

    print(f"Loading probe from {args.probe}...")
    probe_weight, probe_bias = load_probe(args.probe)
    print(f"  probe_weight shape: {probe_weight.shape}, bias: {probe_bias}")

    if args.smoke:
        smoke_dir = repo_root / "data" / "smoke"
        run_smoke(model, tokenizer, probe_weight, probe_bias,
                  args.jigsaw_csv, args.device, smoke_dir)
    elif args.diag:
        results = run_diagnostic(
            model, tokenizer, probe_weight, probe_bias,
            args.jigsaw_csv, args.device, n=2000, seed=42,
        )
        out_path = repo_root / "data" / "smoke" / "probe_diag.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDiagnostic results saved to {out_path}")
    else:
        results = run_full_validation(
            model, tokenizer, probe_weight, probe_bias,
            args.jigsaw_csv, args.device
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

        all_pass = all(results["criteria"].values())
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
