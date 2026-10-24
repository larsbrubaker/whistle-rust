# The mel filter bank fixture (test-data/fixtures/mel.npy) for the Rust front end's mel_matrix test.
"""Writes upstream's `mel_matrix(WhistleConfig())` (needle-reference, pinned in REFERENCE), float32
[257, 80], to test-data/fixtures/mel.npy. It depends only on the config, not on any clip, so it
lives beside the per-clip fixtures rather than in them. Regenerate only when REFERENCE changes:

    NEEDLE_TELEMETRY=0 oracle/.venv/bin/python -I oracle/dump_mel.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "needle-reference"))

import numpy as np  # noqa: E402

from needle.model.whistle import WhistleConfig, mel_matrix  # noqa: E402


def main():
    mel = mel_matrix(WhistleConfig())
    assert mel.dtype == np.float32 and mel.shape == (257, 80), (mel.dtype, mel.shape)
    path = os.path.join(ROOT, "test-data", "fixtures", "mel.npy")
    np.save(path, mel)
    print(f"wrote {path}: {mel.shape} {mel.dtype}")


if __name__ == "__main__":
    main()
