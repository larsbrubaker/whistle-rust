# Black-box engine streaming transcripts for the contract clips (test-data/engine-stream/*.json).
"""Runs the closed-source Needle 3 engine CLI (fetched by scripts/fetch-weights.sh --engine) in
its documented `--audio-stream` mode (default 1 s chunks, one thread) over every contract clip
and writes test-data/engine-stream/<clip>.json: the list of updates it printed (committed `text`,
`words` with times from stream start, `pending`, `language`, `received` seconds), the
nondeterministic `pass_ms` dropped. A black-box reference for the streaming we design ourselves
(docs/design/streaming.md): the final committed text and the commit latency."""
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
    out = os.path.join(ROOT, "test-data", "engine-stream")
    os.makedirs(out, exist_ok=True)
    for clip in sorted(glob.glob(os.path.join(ROOT, "test-data", "clips", "*.wav"))):
        name = os.path.splitext(os.path.basename(clip))[0]
        if name == "silence":
            continue
        run = subprocess.run([engine, "--model", model, "--audio", clip, "--audio-stream",
                              "--threads", "1"], check=True, capture_output=True, text=True)
        updates = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
        for update in updates:
            update.pop("pass_ms", None)
        with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"updates": updates}, f, ensure_ascii=False, indent=1)
            f.write("\n")
        text = " ".join(u["text"] for u in updates if u["text"])
        print(f"{name:12} {len(updates):2} updates  {text}")


if __name__ == "__main__":
    main()
