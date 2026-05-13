"""
Logit lens diagnostic: project probe direction through unembedding matrix.

Runs on two models as positive-control vs. test-subject:
  1. GPT2-medium  + Lee et al.'s probe.pt (references/DPO-detoxify/checkpoints/probe.pt)
     → positive control: top tokens MUST include profanity for the test to be valid
  2. GPT-Neo-125m + our trained probe (data/probe/probe_gpt_neo_125m.pt)
     → test subject: tells us whether the probe learned the toxic direction

For each, computes:
  a. Probe → unembedding: scores = E @ W_probe, top-30 tokens
  b. Value vector cosine ranking (to compare cosine magnitudes)
  c. Top-10 value vector vocab projections

Usage:
  python3 tests/diag_logit_lens_probe.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

PROFANITY_ROOTS = {
    "shit", "ass", "crap", "fuck", "bitch", "dick", "cock", "damn",
    "bastard", "cunt", "whore", "fag", "nigger", "nigga", "slut",
    "retard", "moron", "idiot", "asshole", "bullshit", "motherfuck",
    "piss", "twat", "wanker", "skank",
}


def is_profanity(token: str) -> bool:
    t = token.strip().lower()
    return any(root in t for root in PROFANITY_ROOTS)


def logit_lens(probe_weight: torch.Tensor, E: torch.Tensor, tokenizer, top_k: int = 30):
    """Project probe direction through unembedding E, return top_k tokens."""
    scores = E.float() @ probe_weight.float()  # (vocab_size,)
    top_s, top_ids = scores.topk(top_k)
    tokens = [(tokenizer.decode([tid.item()]), sc.item()) for tid, sc in zip(top_ids, top_s)]
    n_prof = sum(1 for t, _ in tokens if is_profanity(t))
    return tokens, n_prof


def rank_value_vecs_gpt2(model, probe_weight: torch.Tensor, top_k: int = 10):
    """GPT2 Conv1D: value_vector_i = c_proj.weight[i, :] (row), shape (1024,)."""
    device = next(model.parameters()).device
    p = F.normalize(probe_weight.to(device).unsqueeze(0), dim=-1)  # (1, 1024)
    results = []
    for layer_idx, block in enumerate(model.transformer.h):
        w = block.mlp.c_proj.weight.detach().float()  # (4096, 1024) for GPT2-medium
        vv_norm = F.normalize(w, dim=-1)               # (4096, 1024), each row = value vec
        cos = (vv_norm @ p.T).squeeze(-1)               # (4096,)
        top_v, top_i = cos.topk(top_k)
        for c, ni in zip(top_v.tolist(), top_i.tolist()):
            results.append((c, layer_idx, ni))
    results.sort(key=lambda x: -x[0])
    return results


def rank_value_vecs_gpt_neo(model, probe_weight: torch.Tensor, top_k: int = 10):
    """GPT-Neo nn.Linear: value_vector_i = c_proj.weight[:, i] (col), shape (768,)."""
    device = next(model.parameters()).device
    p = F.normalize(probe_weight.to(device).unsqueeze(0), dim=-1)  # (1, 768)
    results = []
    for layer_idx, block in enumerate(model.transformer.h):
        w = block.mlp.c_proj.weight.detach().float()  # (768, 3072) for GPT-Neo-125m
        vv_norm = F.normalize(w.T, dim=-1)             # (3072, 768)
        cos = (vv_norm @ p.T).squeeze(-1)               # (3072,)
        top_v, top_i = cos.topk(top_k)
        for c, ni in zip(top_v.tolist(), top_i.tolist()):
            results.append((c, layer_idx, ni))
    results.sort(key=lambda x: -x[0])
    return results


def project_to_vocab_gpt2(model, tokenizer, layer_idx, neuron_idx, top_k=10):
    """Value vector = row of c_proj.weight for GPT2."""
    v = model.transformer.h[layer_idx].mlp.c_proj.weight[neuron_idx].detach().float()
    E = model.transformer.wte.weight.detach().float()
    scores = E @ v
    top_s, top_ids = scores.topk(top_k)
    return [(tokenizer.decode([tid.item()]), sc.item()) for tid, sc in zip(top_ids, top_s)]


def project_to_vocab_gpt_neo(model, tokenizer, layer_idx, neuron_idx, top_k=10):
    """Value vector = col of c_proj.weight for GPT-Neo."""
    v = model.transformer.h[layer_idx].mlp.c_proj.weight[:, neuron_idx].detach().float()
    E = model.lm_head.weight.detach().float()
    scores = E @ v
    top_s, top_ids = scores.topk(top_k)
    return [(tokenizer.decode([tid.item()]), sc.item()) for tid, sc in zip(top_ids, top_s)]


def run_analysis(model_name, probe_weight, model, tokenizer, rank_fn, proj_fn):
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Probe weight: shape={probe_weight.shape}  norm={probe_weight.norm():.4f}")

    # Get unembedding matrix
    if hasattr(model, 'lm_head'):
        E = model.lm_head.weight.detach().float()
    else:
        E = model.transformer.wte.weight.detach().float()
    print(f"Unembedding E: shape={E.shape}")

    # Logit lens: probe → vocab
    tokens, n_prof = logit_lens(probe_weight, E.cpu(), tokenizer, top_k=30)
    print(f"\n--- Logit lens: E @ probe_weight → top-30 promoted tokens ---")
    print(f"  Profanity count in top-30: {n_prof}/30")
    for rank, (tok, sc) in enumerate(tokens, 1):
        flag = " ← PROFANITY" if is_profanity(tok) else ""
        print(f"  [{rank:2d}]  '{tok.strip()}'  (score={sc:.3f}){flag}")

    # Value vector cosine ranking
    print(f"\n--- Value vector cosine ranking (top-10) ---")
    ranked = rank_fn(model, probe_weight.cpu(), top_k=5)[:10]
    print(f"  Max cosine: {ranked[0][0]:.4f}  at (layer={ranked[0][1]}, neuron={ranked[0][2]})")
    for rank, (cos, layer, neuron) in enumerate(ranked, 1):
        top_toks = proj_fn(model, tokenizer, layer, neuron, top_k=5)
        tok_str = ", ".join(f"'{t.strip()}'" for t, _ in top_toks)
        n_p = sum(1 for t, _ in top_toks if is_profanity(t))
        print(f"  [{rank:2d}] L{layer}N{neuron} cos={cos:.4f}  {tok_str}  (prof={n_p}/5)")

    verdict = "POSITIVE" if n_prof >= 5 else ("WEAK" if n_prof >= 2 else "NEGATIVE")
    print(f"\n  Logit lens verdict: {verdict} (≥5 profanity in top-30 = positive signal)")
    return {"n_profanity_top30": n_prof, "max_cosine": ranked[0][0], "verdict": verdict}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Positive control: GPT2-medium + Lee et al.'s probe ──────────────────
    print("\nLoading GPT2-medium (positive control)...")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    gpt2_tok   = GPT2Tokenizer.from_pretrained("gpt2-medium")

    lee_probe = torch.load("references/DPO-detoxify/checkpoints/probe.pt", map_location="cpu")
    print(f"Lee et al. probe: shape={lee_probe.shape}  norm={lee_probe.norm():.4f}")

    r1 = run_analysis(
        "GPT2-medium + Lee et al. probe.pt",
        lee_probe, gpt2_model, gpt2_tok,
        rank_value_vecs_gpt2, project_to_vocab_gpt2,
    )

    # Free GPT2-medium memory
    del gpt2_model
    torch.cuda.empty_cache()

    # ── Test subject: GPT-Neo-125m + our probe ───────────────────────────────
    print("\nLoading GPT-Neo-125m (test subject)...")
    neo_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m").to(device)
    neo_tok   = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    neo_tok.pad_token = neo_tok.eos_token

    payload = torch.load("data/probe/probe_gpt_neo_125m.pt", map_location="cpu")
    neo_probe = payload["weight"]  # (768,)
    print(f"Our probe: shape={neo_probe.shape}  norm={neo_probe.norm():.4f}  "
          f"accuracy={payload['accuracy']:.4f}")

    r2 = run_analysis(
        "GPT-Neo-125m + our trained probe",
        neo_probe, neo_model, neo_tok,
        rank_value_vecs_gpt_neo, project_to_vocab_gpt_neo,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"  GPT2-medium  logit lens: n_profanity={r1['n_profanity_top30']}/30  "
          f"max_cosine={r1['max_cosine']:.4f}  verdict={r1['verdict']}")
    print(f"  GPT-Neo-125m logit lens: n_profanity={r2['n_profanity_top30']}/30  "
          f"max_cosine={r2['max_cosine']:.4f}  verdict={r2['verdict']}")

    if r1['verdict'] != 'POSITIVE':
        print("\n  WARN: GPT2-medium positive control FAILED — Lee et al.'s probe "
              "does not show profanity via logit lens. The diagnostic itself may be wrong.")
    elif r2['verdict'] == 'POSITIVE':
        print("\n  CONCLUSION: GPT-Neo-125m probe shows profanity in logit lens.")
        print("  Probe is valid. GPT-Neo-125m likely has distributed toxicity (no single vector).")
    elif r2['verdict'] == 'WEAK':
        print("\n  CONCLUSION: GPT-Neo-125m probe is weakly aligned with toxic direction.")
        print("  Partial probe learning — consider retraining or broadening φ.")
    else:
        print("\n  CONCLUSION: GPT-Neo-125m probe does NOT show profanity in logit lens.")
        print("  Possible causes:")
        print("    1. Probe learned topic/sentiment (political content) not token-level toxicity")
        print("    2. GPT-Neo-125m has no vocabulary-level toxic direction (architecture difference)")
        print("    3. Training data or tokenization mismatch")
        print("  → Surface to user. Do NOT proceed with value-vector analysis until this is resolved.")


if __name__ == "__main__":
    main()
