# Extra test-clip synthesis for the keyword and beam-search tests (macOS `say` -> test-data/clips-extra/).
"""Synthesizes the clips listed in test-data/clips-extra/clips.json (names, places, product words
and jargon, each with the keywords a caller would pass) with macOS `say` into
test-data/clips-extra/<name>.wav as 16 kHz mono PCM16 WAV, as make_clips.py does for the contract
clips. The WAVs are committed, so this only reruns when clips.json changes; `say` voices differ
between macOS releases."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test-data", "clips-extra")


def main():
    if sys.platform != "darwin":
        raise SystemExit("make_extra_clips.py needs macOS `say`")
    with open(os.path.join(OUT, "clips.json"), encoding="utf-8") as f:
        clips = json.load(f)
    for clip in clips:
        path = os.path.join(OUT, clip["name"] + ".wav")
        subprocess.run(["say", "-v", clip["voice"], "--file-format=WAVE", "--data-format=LEI16@16000",
                        "-o", path, clip["text"]], check=True)
        print(path)


if __name__ == "__main__":
    main()
