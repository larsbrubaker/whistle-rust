# The Engram n-gram hash fixture (test-data/fixtures/engram_indices.json) for the Rust port's hashing test.
"""Writes upstream's `engram_indices` (needle-reference, pinned in REFERENCE) for a few token
sequences at Whistle's geometry (orders (2, 3), two hash heads, 18,432 slots, seed heads 0 as
`Whistle.decode` passes them), plus the n-gram and tap masks `Whistle.decode` derives from the
causal mask. The sequences cover a real decoder input, the shortest sequences, and token ids at
the edges of u32 (int32 ids that wrap when cast, the largest int32), so the wrapping-multiply
hash is pinned exactly. It depends on no clip and no weights. Regenerate only when REFERENCE
changes:

    NEEDLE_TELEMETRY=0 oracle/.venv/bin/python -I oracle/dump_engram.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "needle-reference"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from needle.model.architecture import (  # noqa: E402
    ENGRAM_CONV_TAPS, _mask_diag, engram_geometry, engram_indices, make_causal_mask)
from needle.model.whistle import WhistleConfig  # noqa: E402

# Each case is a decoder input. Negative values are int32 ids that the reference's
# `astype(uint32)` wraps to the top of u32 (-1 -> 0xFFFFFFFF).
CASES = [
    [2, 8192, 7425, 309, 315, 4445, 1727, 325, 8107, 7125, 353, 7647, 309, 1593, 1510, 3008, 8116, 1],
    [2],
    [2, 8193],
    [0, 0, 0, 0, 0, 0, 0],
    [2147483647, -1, -2147483648, 4294967 * 400, 18432, 18431, 36864, 1],
    [-1, -1, -1, 7, 2147483647, 2147483647, 8198, 0, 65535, 65536, 16777216, 3],
]


def main():
    cfg = WhistleConfig()
    orders, heads, _ = engram_geometry(cfg.needle())
    cases = []
    for tokens in CASES:
        t = jnp.asarray(np.array([tokens], np.int64).astype(np.int32))
        idx = np.asarray(engram_indices(t, orders, heads, cfg.slots, 0))[0]
        mask = make_causal_mask(len(tokens))
        ngram_ok = np.asarray(jnp.stack(
            [_mask_diag(mask, o - 1) for o in orders for _ in range(heads)], axis=-1))[0]
        tap_ok = np.asarray(jnp.stack(
            [_mask_diag(mask, j * max(orders)) for j in range(ENGRAM_CONV_TAPS)]))[:, 0]
        cases.append({
            # The ids as u32, which is what the hash reads.
            "tokens": [int(v) & 0xFFFFFFFF for v in tokens],
            # [T, orders * heads], table t = order_index * heads + head.
            "indices": idx.astype(np.int64).tolist(),
            "ngram_ok": ngram_ok.astype(np.int64).tolist(),
            # [taps, T].
            "tap_ok": tap_ok.astype(np.int64).tolist(),
        })
    out = {
        "orders": list(orders),
        "heads": heads,
        "slots": cfg.slots,
        "seed_heads": 0,
        "cases": cases,
    }
    path = os.path.join(ROOT, "test-data", "fixtures", "engram_indices.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {path}: {len(cases)} cases")


if __name__ == "__main__":
    main()
