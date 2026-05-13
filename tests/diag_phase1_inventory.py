"""
Phase 1 kickoff diagnostic — verify Phase 0 artifacts load.

Confirms:
  - step_0650 checkpoint loads via AutoModelForCausalLM (no value head)
  - EleutherAI/gpt-neo-125m loads as ref model
  - data/rollouts/step_0650.pt schema matches phase1.md
  - data/eval_prompts_400.json has expected structure

Not run by pytest. Run directly: python tests/diag_phase1_inventory.py
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO / "data" / "ppo_checkpoints" / "step_0650"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Load θ* — step_0650 checkpoint as plain CausalLM (no value head)
    print("\n[1] Loading θ* from", CKPT_DIR)
    tok_star = AutoTokenizer.from_pretrained(CKPT_DIR)
    model_star = AutoModelForCausalLM.from_pretrained(CKPT_DIR).to(device).eval()
    print(f"    θ* params: {sum(p.numel() for p in model_star.parameters()):,}")
    print(f"    θ* class:  {type(model_star).__name__}")

    # 2. Load ref model
    print("\n[2] Loading π_ref = EleutherAI/gpt-neo-125m (untrained)")
    tok_ref = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    model_ref = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m").to(device).eval()
    print(f"    π_ref params: {sum(p.numel() for p in model_ref.parameters()):,}")

    # Sanity: vocab + config alignment
    assert tok_star.vocab_size == tok_ref.vocab_size, "tokenizer vocab mismatch"
    assert model_star.config.hidden_size == model_ref.config.hidden_size == 768
    assert model_star.config.num_layers == model_ref.config.num_layers == 12

    # Sanity: their weights should differ (PPO trained vs untrained)
    delta = (
        model_star.transformer.h[9].mlp.c_proj.weight.float()
        - model_ref.transformer.h[9].mlp.c_proj.weight.float()
    ).norm().item()
    print(f"    ||θ*.layer9.c_proj − π_ref.layer9.c_proj||_F = {delta:.4f}")
    assert delta > 1e-3, "θ* identical to ref model — checkpoint may be wrong"

    # 3. Quick forward pass on a 1-token input
    print("\n[3] Forward pass smoke")
    test_ids = tok_star("Hello world", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out_star = model_star(test_ids).logits
        out_ref = model_ref(test_ids).logits
    print(f"    logits shape (θ*):  {tuple(out_star.shape)}")
    print(f"    logits shape (ref): {tuple(out_ref.shape)}")
    assert out_star.shape == out_ref.shape

    # 4. Inspect a rollout file
    print("\n[4] Inspect data/rollouts/step_0650.pt")
    rollout = torch.load(REPO / "data" / "rollouts" / "step_0650.pt", weights_only=False)
    assert isinstance(rollout, list), f"expected list, got {type(rollout).__name__}"
    print(f"    rollout count in step_0650.pt: {len(rollout)}")
    first = rollout[0]
    print(f"    keys: {sorted(first.keys())}")
    for k in ("prompt_text", "response_text", "reward", "step"):
        v = first[k]
        if isinstance(v, str):
            v = v[:60]
        print(f"      {k!r}: {v!r}")
    for k in ("prompt_token_ids", "response_token_ids", "policy_logprobs", "ref_logprobs"):
        v = first[k]
        if hasattr(v, "shape"):
            print(f"      {k!r}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"      {k!r}: type={type(v).__name__} len={len(v)}")

    # 5. eval_prompts_400.json
    print("\n[5] Inspect data/eval_prompts_400.json")
    import json
    with open(REPO / "data" / "eval_prompts_400.json") as f:
        ep = json.load(f)
    print(f"    keys: {sorted(ep.keys())}")
    print(f"    n prompts: {len(ep['prompt_texts'])}")
    print(f"    first prompt: {ep['prompt_texts'][0]!r}")

    print("\nAll inventory checks passed.")


if __name__ == "__main__":
    main()
