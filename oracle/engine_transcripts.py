# Black-box engine transcripts for every test clip (test-data/engine/*.json).
"""Runs the closed-source Needle 3 engine CLI (fetched by scripts/fetch-weights.sh --engine) over
every test clip and writes test-data/engine/<clip>.json: text, language and word timestamps,
timing fields dropped. A black-box quality reference for the decode features we design ourselves
(beam search, keywords, timestamps); the per-tensor oracle is dump.py.

The reference's greedy transcripts match the engine's word for word on seven of the nine clips.
Polish differs at the end ("... do 10 ml." against the engine's "... do 10 milimetrów lewo."),
which three beams reproduce; on a second of silence the greedy decode says "Thank you." where
the engine returns an empty transcript, as the port's silence gate does."""
import glob
import json
import os
import platform
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORMS = {("Darwin", "arm64"): "macos-arm64", ("Linux", "x86_64"): "linux-x86_64",
             ("Linux", "aarch64"): "linux-arm64"}


def main():
    folder = PLATFORMS[(platform.system(), platform.machine())]
    engine = os.path.join(ROOT, "oracle", ".engine", folder, "needle")
    model = os.path.join(ROOT, "models", "whistle.cact")
    out = os.path.join(ROOT, "test-data", "engine")
    os.makedirs(out, exist_ok=True)
    for clip in sorted(glob.glob(os.path.join(ROOT, "test-data", "clips", "*.wav"))):
        run = subprocess.run([engine, "--model", model, "--audio", clip, "--audio-word-timestamps",
                              "--threads", "1"], check=True, capture_output=True, text=True)
        result = json.loads(run.stdout)
        for timing in ("ttft_ms", "decode_tps"):
            result.pop(timing, None)
        name = os.path.splitext(os.path.basename(clip))[0]
        with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"{name:12} {result['language'] or '--':3} {result['text']}")


if __name__ == "__main__":
    main()
