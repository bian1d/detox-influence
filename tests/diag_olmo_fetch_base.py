"""Fetch the two OLMo-2-0425-1B safetensors shards through the flaky proxy by
parallel byte-range chunks, because (a) the HF python downloader hangs at 0
bytes on the large streamed responses through this proxy, and (b) a single curl
stream caps at ~0.1-0.3 MB/s while N parallel connections scale near-linearly.

Strategy:
  * Each shard is split into fixed-size chunks; each chunk is a curl range
    request with resume (appends to its .part file) and a stall-abort
    (--speed-limit/--speed-time) wrapped in a retry loop -- so a hung socket
    aborts and resumes instead of blocking forever.
  * Chunks run in a thread pool (curl subprocesses). .part files persist, so
    re-running the script resumes wherever it stopped.
  * When a shard's chunks are all complete, they are concatenated, the sha256
    is verified against the HF-published value, and the file is placed into the
    HF cache as blobs/<sha256> with the snapshot symlink -- so
    from_pretrained("allenai/OLMo-2-0425-1B") then loads by id from cache.

Run with the olmo env python (proxy env vars already exported):
    /root/miniconda3/envs/olmo/bin/python tests/diag_olmo_fetch_base.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "allenai/OLMo-2-0425-1B"
REV = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
CACHE = Path("/root/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B")
BLOBS = CACHE / "blobs"
SNAP = CACHE / "snapshots" / REV
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REV}"

SHARDS = [
    ("model-00001-of-00002.safetensors", 4983360992,
     "c52e4ac8bf4867e698d908f9e8c2b84aa891b91dff066d46bf2a7cf453ad77fb"),
    ("model-00002-of-00002.safetensors", 956326560,
     "5f80762572fe28aca49b01b98895d96ca529b9792fc10925a977a161ad768d30"),
]

import time  # noqa: E402

CHUNK = 128 * 1024 * 1024  # 128 MB
WORKERS = int(os.environ.get("OLMO_FETCH_WORKERS", "8"))
WORKDIR = Path("/root/rlhf-influence/data/olmo_base_chunks")


def _download_chunk(url: str, lo: int, hi: int, part: Path) -> tuple[int, bool]:
    """Resume `part` to cover bytes [lo, hi] inclusive. Returns (idx unused, ok).

    Corruption guard: this proxy intermittently answers a Range request with a
    200 (whole file from byte 0) instead of a 206 (the requested partial). The
    earlier version appended that 200 body blindly and only checked size, which
    poisoned the head of a chunk while still reaching the exact expected size
    (right size, wrong sha256). We now fetch each attempt into a temp file and
    only append it if curl reports http 206 -- a 200 / error body is discarded.
    """
    expected = hi - lo + 1
    att = part.with_suffix(part.suffix + ".att")
    for _attempt in range(500):
        have = part.stat().st_size if part.exists() else 0
        if have >= expected:
            if have > expected:  # trim any accidental overshoot to exact size
                with open(part, "r+b") as fh:
                    fh.truncate(expected)
            return (lo, True)
        start = lo + have
        att.unlink(missing_ok=True)
        proc = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", "900",
             "--speed-limit", "4096", "--speed-time", "30",
             "-r", f"{start}-{hi}", "-o", str(att), "-w", "%{http_code}", url],
            capture_output=True, text=True,
        )
        code = (proc.stdout or "").strip()[-3:]
        got = att.stat().st_size if att.exists() else 0
        if code == "206" and 0 < got <= expected - have:
            with open(part, "ab") as dst, open(att, "rb") as src:
                shutil.copyfileobj(src, dst)
        else:
            # 200 (range ignored), error page, or refused -> discard, back off so
            # a transient outage can't burn all 500 attempts in seconds.
            time.sleep(3)
        att.unlink(missing_ok=True)
    return (lo, (part.stat().st_size if part.exists() else 0) >= expected)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_shard(name: str, size: int, sha: str) -> bool:
    blob = BLOBS / sha
    link = SNAP / name
    if blob.exists() and blob.stat().st_size == size:
        print(f"[{name}] blob already present ({size/1e9:.2f} GB)")
    else:
        url = f"{BASE_URL}/{name}"
        n_chunks = (size + CHUNK - 1) // CHUNK
        ranges = [(i * CHUNK, min((i + 1) * CHUNK, size) - 1) for i in range(n_chunks)]
        parts = [WORKDIR / f"{name}.{i:03d}.part" for i in range(n_chunks)]
        print(f"[{name}] {size/1e9:.2f} GB in {n_chunks} chunks x{WORKERS} workers", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(_download_chunk, url, lo, hi, parts[i]): i
                    for i, (lo, hi) in enumerate(ranges)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                _, ok = fut.result()
                done += 1
                if not ok:
                    print(f"[{name}] chunk {i} FAILED", flush=True)
                    return False
                print(f"[{name}] chunk {i:03d} done ({done}/{n_chunks})", flush=True)
        # assemble
        tmp = WORKDIR / f"{name}.assembled"
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as fh:
                    for block in iter(lambda: fh.read(8 << 20), b""):
                        out.write(block)
        got = tmp.stat().st_size
        if got != size:
            print(f"[{name}] SIZE MISMATCH assembled={got} expected={size}")
            return False
        print(f"[{name}] verifying sha256 ...", flush=True)
        digest = _sha256(tmp)
        if digest != sha:
            print(f"[{name}] SHA256 MISMATCH got={digest} want={sha}")
            return False
        BLOBS.mkdir(parents=True, exist_ok=True)
        tmp.rename(blob)
        for p in parts:
            p.unlink(missing_ok=True)
        print(f"[{name}] sha256 OK -> placed blob")

    # (re)create snapshot symlink and clear any stale .incomplete
    SNAP.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(blob, SNAP))
    stale = BLOBS / f"{sha}.incomplete"
    stale.unlink(missing_ok=True)
    print(f"[{name}] symlink -> {link}")
    return True


def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    for name, size, sha in SHARDS:
        if not fetch_shard(name, size, sha):
            print("FETCH FAILED")
            return 1
    print("\nALL SHARDS PRESENT + VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
