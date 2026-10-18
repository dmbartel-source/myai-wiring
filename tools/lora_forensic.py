#!/usr/bin/env python3
"""Read-only LoRA forensic probe (completion plan step 6, Oct 17).

Settles the alpha question for the trained person LoRAs without a GPU:
prints each file's size, sha256, training metadata (network dim/alpha
and friends), tensor counts, and the stored per-module alpha scalars.

Pure stdlib -- the safetensors header is parsed directly, so this runs
under any python on the pod. It reads header bytes + a few scalar
tensors only; it never loads the full weights and never writes.

Usage: python3 lora_forensic.py [path ...]
Defaults: the two installed person LoRAs on the pod volume.
"""
import hashlib
import json
import struct
import sys

DEFAULT_PATHS = [
    "/workspace/ComfyUI/models/loras/skswoman.safetensors",
    "/workspace/ComfyUI/models/loras/skswoman_v2.safetensors",
]

_DTYPES = {  # name -> (struct fmt or None, byte size)
    "F64": ("<d", 8), "F32": ("<f", 4), "F16": ("<e", 2),
    "BF16": (None, 2), "I64": ("<q", 8), "I32": ("<i", 4),
    "I16": ("<h", 2), "I8": ("<b", 1), "U8": ("<B", 1),
    "BOOL": ("<B", 1),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("file too short for a safetensors header")
        (n,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(n))
    return header, 8 + n


def _scalar(path, data_start, info):
    dtype = info.get("dtype")
    if dtype not in _DTYPES:
        return None
    fmt, size = _DTYPES[dtype]
    begin, end = info["data_offsets"]
    if end - begin != size:
        return None
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        buf = f.read(size)
    if dtype == "BF16":
        (bits,) = struct.unpack("<H", buf)
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    return float(struct.unpack(fmt, buf)[0])


def summarize(path):
    out = {"path": path}
    try:
        header, data_start = read_header(path)
    except FileNotFoundError:
        out["status"] = "missing"
        return out
    except Exception as e:  # unreadable header: report, don't crash
        out["status"] = f"error: {type(e).__name__}: {e}"
        return out
    import os
    out["status"] = "ok"
    out["bytes"] = os.path.getsize(path)
    out["sha256"] = _sha256(path)
    meta = header.get("__metadata__", {}) or {}
    keep = ("alpha", "dim", "network", "base_model", "learning_rate",
            "output_name", "epoch", "step")
    out["metadata"] = {k: meta[k] for k in sorted(meta)
                       if any(s in k for s in keep)}
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    out["tensor_count"] = len(tensors)
    out["lora_down_count"] = sum(1 for k in tensors if "lora_down" in k)
    out["lora_up_count"] = sum(1 for k in tensors if "lora_up" in k)
    alphas = []
    for name, info in tensors.items():
        if name.endswith(".alpha") or name == "alpha":
            val = _scalar(path, data_start, info)
            if val is not None:
                alphas.append(val)
    if alphas:
        uniq = sorted(set(alphas))
        out["alpha"] = {"count": len(alphas), "min": min(alphas),
                        "max": max(alphas),
                        "unique_first10": uniq[:10]}
    else:
        out["alpha"] = {"count": 0}
    return out


def main(argv):
    paths = argv[1:] or DEFAULT_PATHS
    for p in paths:
        print(json.dumps(summarize(p), sort_keys=True))
    print("LORA-FORENSIC-DONE")


if __name__ == "__main__":
    main(sys.argv)
