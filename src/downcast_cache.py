"""Downcast the EK-FAC g_scaled cache from fp64 → fp32 in place.

The production run saved at native fp64 (the dtype dropped out of the
upstream ``Q_*_d.double()`` factors after the user removed the
``.to(float16)`` cast). This script reads each file, casts to fp32,
writes a sibling ``.tmp`` then ``os.replace``s it onto the original — so
peak disk during the run is +9 MB above the steady state (the single
temporary file), and any individual rollout's cache is *atomic*: the
original is replaced only after the fp32 version is fully on disk.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

CACHE_DIR = Path("data/ekfac/cache/g_scaled_step0650_layer9")
META_PATH = Path("data/ekfac/influence/run_step0650/metadata.json")


def main() -> int:
    if not CACHE_DIR.is_dir():
        print(f"FATAL: {CACHE_DIR} not found", file=sys.stderr)
        return 1

    files = sorted(CACHE_DIR.glob("g_scaled_z_*.pt"))
    if not files:
        print(f"FATAL: no cache files found in {CACHE_DIR}", file=sys.stderr)
        return 1
    n = len(files)
    print(f"Found {n} cache files in {CACHE_DIR}")

    # Sanity-check the first file before mass downcast.
    head = torch.load(files[0], map_location="cpu", weights_only=True)
    print(f"First file dtype = {head.dtype}, shape = {tuple(head.shape)}")
    if head.dtype == torch.float32:
        print("First file already fp32 — nothing to do.")
        return 0
    if head.dtype != torch.float64:
        print(f"FATAL: unexpected dtype {head.dtype} — refusing to proceed",
              file=sys.stderr)
        return 1
    del head

    n_done = 0
    n_skipped = 0
    t0 = time.perf_counter()
    expected_shape = None
    for i, fp in enumerate(files):
        try:
            t = torch.load(fp, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"FATAL: cannot load {fp.name}: {e}", file=sys.stderr)
            return 1
        if expected_shape is None:
            expected_shape = tuple(t.shape)
        else:
            if tuple(t.shape) != expected_shape:
                print(f"FATAL: shape drift at {fp.name}: {tuple(t.shape)} vs "
                      f"{expected_shape}", file=sys.stderr)
                return 1
        if t.dtype == torch.float32:
            n_skipped += 1
        else:
            t32 = t.to(torch.float32)
            del t
            tmp_path = fp.with_suffix(".pt.tmp")
            torch.save(t32, tmp_path)
            # Verify the temp file before replacing.
            verify = torch.load(tmp_path, map_location="cpu", weights_only=True)
            if verify.dtype != torch.float32 or tuple(verify.shape) != expected_shape:
                tmp_path.unlink(missing_ok=True)
                print(f"FATAL: verification failed for {fp.name}", file=sys.stderr)
                return 1
            del verify
            os.replace(tmp_path, fp)
            n_done += 1
        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  [{i+1}/{n}]  done={n_done}  skipped={n_skipped}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  rate={rate:.0f}/s")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_done} downcast, {n_skipped} already fp32, "
          f"elapsed {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Update metadata.json (the production run accidentally wrote "float32"
    # while the actual cache was fp64; now the label is finally truthful).
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        old = meta.get("cache_dtype", "<unset>")
        meta["cache_dtype"] = "float32"
        meta.setdefault("notes", []).append(
            f"Cache downcast fp64→fp32 in place; previous label '{old}' "
            f"was incorrect (cache was actually fp64). {n_done} files "
            f"converted, {n_skipped} already fp32."
        )
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"Updated metadata cache_dtype: '{old}' -> 'float32'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
