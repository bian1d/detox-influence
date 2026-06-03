"""Phase 5 Section 0 Dependency A: confirm OLMo-2-1B loads + forwards in the
olmo conda env (transformers >= 4.48). Run with the olmo env python:

    /root/miniconda3/envs/olmo/bin/python tests/diag_olmo_load.py

Checks: Olmo2 import, from_pretrained (SFT = theta-star, base = reference),
a forward pass, tokenizer round-trip, VRAM, and the MLP module structure
(confirms gate/up/down_proj + the layer-12 down_proj shape = Section 1 phi).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT = "allenai/OLMo-2-0425-1B-SFT"
BASE = "allenai/OLMo-2-0425-1B"
PHI_LAYER = 12


def inspect(model_id: str, *, full: bool) -> None:
    print(f"\n{'='*70}\n{model_id}\n{'='*70}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda").eval()
    print(f"  loaded: {model.__class__.__name__}, dtype {next(model.parameters()).dtype}")
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    # Tokenizer round-trip.
    text = "The argument that some groups are inherently"
    ids = tok(text, return_tensors="pt").to("cuda")
    decoded = tok.decode(ids["input_ids"][0], skip_special_tokens=True)
    print(f"  tokenizer round-trip ok: {decoded == text}  (n_tokens={ids['input_ids'].shape[1]})")

    # Forward pass + VRAM.
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(**ids)
    print(f"  forward ok: logits {tuple(out.logits.shape)} (vocab={out.logits.shape[-1]})")
    print(f"  peak VRAM (fp32 load + fwd): {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if full:
        # MLP structure at the proposed phi layer.
        layer = model.model.layers[PHI_LAYER]
        mlp = layer.mlp
        names = [n for n, _ in mlp.named_children()]
        print(f"  layer-{PHI_LAYER} mlp submodules: {names}")
        dp = mlp.down_proj.weight
        print(f"  layer-{PHI_LAYER} down_proj.weight shape (out,in): {tuple(dp.shape)} "
              f"= {dp.numel()/1e6:.2f}M params")
        print(f"  total layers: {len(model.model.layers)}  "
              f"(phi depth {PHI_LAYER}/{len(model.model.layers)} = {PHI_LAYER/len(model.model.layers):.3f})")
        # Confirm get_submodule path works (what EK-FAC will use).
        path = f"model.layers.{PHI_LAYER}.mlp.down_proj"
        sub = model.get_submodule(path)
        print(f"  get_submodule('{path}') ok: weight {tuple(sub.weight.shape)}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import transformers
    print("transformers", transformers.__version__, "| torch", torch.__version__)
    from transformers import Olmo2ForCausalLM  # noqa: F401
    print("Olmo2ForCausalLM import: OK")
    inspect(SFT, full=True)
    inspect(BASE, full=False)
    print("\nALL OLMO-2 LOAD CHECKS PASSED")
