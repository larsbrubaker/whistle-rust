# Test-clip synthesis for the oracle (macOS `say` -> test-data/clips/*.wav).
"""Synthesizes the test clips in clips.json with macOS `say` into test-data/clips/ as 16 kHz mono
PCM16 WAV (what MatterCAD's recorders produce), plus a second of silence. The WAVs are committed,
so this only reruns when clips.json changes; `say` voices differ between macOS releases."""
import json
import os
import subprocess
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test-data", "clips")


def main():
    if sys.platform != "darwin":
        raise SystemExit("make_clips.py needs macOS `say`")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(ROOT, "oracle", "clips.json"), encoding="utf-8") as f:
        clips = json.load(f)
    for clip in clips:
        path = os.path.join(OUT, clip["name"] + ".wav")
        subprocess.run(["say", "-v", clip["voice"], "--file-format=WAVE", "--data-format=LEI16@16000",
                        "-o", path, clip["text"]], check=True)
        print(path)
    with wave.open(os.path.join(OUT, "silence.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000)
    print(os.path.join(OUT, "silence.wav"))


if __name__ == "__main__":
    main()
