# Black-box engine transcripts for the extra clips, with and without keywords (test-data/engine-extra/).
"""Runs the closed-source Needle 3 engine CLI (fetched by scripts/fetch-weights.sh --engine) over
every clip in test-data/clips-extra/clips.json twice: once plain, once with the clip's keywords
passed through --audio-keywords (one per line). Writes test-data/engine-extra/<clip>.json and
<clip>.keywords.json: text, language and word timestamps, timing fields dropped. A black-box run
only: the quality reference for our own beam search and keyword biasing."""
import json
import os
import platform
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORMS = {("Darwin", "arm64"): "macos-arm64", ("Linux", "x86_64"): "linux-x86_64",
             ("Linux", "aarch64"): "linux-arm64"}


def run(engine, model, wav, keywords_file):
    cmd = [engine, "--model", model, "--audio", wav, "--audio-word-timestamps", "--threads", "1"]
    if keywords_file:
        cmd += ["--audio-keywords", keywords_file]
    result = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    for timing in ("ttft_ms", "decode_tps"):
        result.pop(timing, None)
    return result


def write(path, result):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")


def main():
    folder = PLATFORMS[(platform.system(), platform.machine())]
    engine = os.path.join(ROOT, "oracle", ".engine", folder, "needle")
    model = os.path.join(ROOT, "models", "whistle.cact")
    clips_dir = os.path.join(ROOT, "test-data", "clips-extra")
    out = os.path.join(ROOT, "test-data", "engine-extra")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(clips_dir, "clips.json"), encoding="utf-8") as f:
        clips = json.load(f)
    for clip in clips:
        wav = os.path.join(clips_dir, clip["name"] + ".wav")
        plain = run(engine, model, wav, None)
        write(os.path.join(out, clip["name"] + ".json"), plain)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as k:
            k.write("\n".join(clip["keywords"]) + "\n")
        try:
            biased = run(engine, model, wav, k.name)
        finally:
            os.unlink(k.name)
        write(os.path.join(out, clip["name"] + ".keywords.json"), biased)
        print(f"{clip['name']:10} plain:    {plain['text']}")
        print(f"{'':10} keywords: {biased['text']}")


if __name__ == "__main__":
    main()
