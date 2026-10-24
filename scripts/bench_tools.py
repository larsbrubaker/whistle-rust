# Helpers for scripts/bench.sh (run with `oracle/.venv/bin/python -I`): builds the benchmark's
# extra clips, times the closed engine CLI as a black box, and prints the comparison table.
#
#   bench_tools.py clips OUT_DIR      write long.wav (speech clips joined by 0.3 s gaps, under
#                                     30 s) and tiny.wav (0.2 s of silence) into OUT_DIR
#   bench_tools.py engine ENGINE MODEL THREADS RUNS CLIP...
#                                     run the engine RUNS times per clip, print a JSON line per
#                                     clip with the wall time per invocation and its own timings
#   bench_tools.py table RESULTS_DIR  read the *.jsonl files bench.sh collected and print the
#                                     table, medians of the per-round medians
#   bench_tools.py wasmtable RESULTS_DIR
#                                     read the <runtime>.jsonl files scripts/wasm-bench.sh
#                                     collected (native, node-*, wasmtime-*) and print one
#                                     single-thread table per beam count
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import wave

GAP_SECONDS = 0.3
LIMIT_SECONDS = 29.5  # the library rejects audio over 30 s


def read(path):
    with wave.open(path) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        return w.readframes(w.getnframes())


def write(path, frames):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(frames)


def make_clips(out):
    os.makedirs(out, exist_ok=True)
    speech = [p for p in sorted(glob.glob("test-data/clips/*.wav")) if "silence" not in p]
    gap = bytes(2 * int(16000 * GAP_SECONDS))
    parts = [read(speech[0])]
    i = 1
    # Cycle through the speech clips until the next one would not fit under the limit.
    while True:
        nxt = read(speech[i % len(speech)])
        if (sum(map(len, parts)) + len(gap) * len(parts) + len(nxt)) / 32000 >= LIMIT_SECONDS:
            break
        parts.append(nxt)
        i += 1
    write(os.path.join(out, "long.wav"), gap.join(parts))
    write(os.path.join(out, "tiny.wav"), bytes(2 * 3200))
    print(f"long.wav: {len(parts)} clips, {(sum(map(len, parts)) + len(gap) * (len(parts) - 1)) / 32000:.2f} s")


def engine(exe, model, threads, runs, *clips):
    for clip in clips:
        walls, last = [], {}
        for _ in range(int(runs)):
            start = time.perf_counter()
            run = subprocess.run([exe, "--model", model, "--audio", clip, "--threads", threads],
                                 check=True, capture_output=True, text=True)
            walls.append((time.perf_counter() - start) * 1e3)
            last = json.loads(run.stdout)
        walls.sort()
        print(json.dumps({"kind": "engine", "clip": os.path.splitext(os.path.basename(clip))[0],
                          "threads": int(threads), "runs": len(walls),
                          "median_ms": round(statistics.median(walls), 1),
                          "min_ms": round(walls[0], 1), "ttft_ms": last.get("ttft_ms"),
                          "decode_tps": last.get("decode_tps"), "text": last.get("text", "")}),
              flush=True)


def load(results):
    rows = []
    for path in sorted(glob.glob(os.path.join(results, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.startswith("{")]
    return rows


def table(results):
    rows = load(results)
    med = statistics.median

    def pick(kind, clip, **kw):
        return [r for r in rows if r["kind"] == kind and r.get("clip") == clip
                and all(r.get(k) == v for k, v in kw.items())]

    clips = []
    for r in rows:
        if r["kind"] == "clip" and r["clip"] not in clips:
            clips.append(r["clip"])
    configs = sorted({(r["beams"], r["threads"]) for r in rows if r["kind"] == "clip"})
    eloads = {t: [r["median_ms"] for r in pick("engine", "tiny", threads=t)] for t in (1, 4)}
    eload = {t: med(v) for t, v in eloads.items() if v}
    print("engine floor (process start + load, 0.2 s of silence): " +
          ", ".join(f"{t} thread(s) {v:.1f} ms" for t, v in sorted(eload.items())))
    for t in (1, 4):
        loads = [r["load_ms"] for r in rows if r["kind"] == "load" and r["threads"] == t]
        if loads:
            print(f"our load (read + Whistle::from_cact), {t} thread(s): median {med(loads):.1f} ms, "
                  f"min {min(loads):.1f} ms over {len(loads)} loads")
    head = ["clip", "sec"]
    for b, t in configs:
        head.append(f"b{b} t{t}")
    head += ["enc t4", "eng t1 raw", "eng t1 net", "eng t4 raw", "eng t4 net", "eng ttft t1", "eng tps t1"]
    print("\nmedian wall ms per transcription (median of the per-round medians)")
    print(" | ".join(head))
    for clip in clips:
        first = pick("clip", clip)[0]
        cells = [clip, f"{first['seconds']:.1f}"]
        for b, t in configs:
            v = pick("clip", clip, beams=b, threads=t)
            cells.append(f"{med(r['median_ms'] for r in v):.0f}" if v else "-")
        enc = pick("clip", clip, threads=4)
        cells.append(f"{med(r['encoder_ms'] for r in enc):.0f}" if enc else "-")
        for t in (1, 4):
            v = pick("engine", clip, threads=t)
            raw = med(r["median_ms"] for r in v) if v else None
            cells += [f"{raw:.0f}" if raw else "-",
                      f"{max(raw - eload.get(t, 0), 0):.0f}" if raw else "-"]
        v = pick("engine", clip, threads=1)
        cells.append(f"{med(r['ttft_ms'] for r in v):.0f}" if v else "-")
        cells.append(f"{med(r['decode_tps'] for r in v):.0f}" if v else "-")
        print(" | ".join(cells))
    print("\ntext agreement with the engine (clips with identical text, of", len(clips), ")")
    for b, t in configs:
        same = 0
        for clip in clips:
            ours = pick("clip", clip, beams=b, threads=t)
            eng = pick("engine", clip, threads=1)
            if ours and eng and ours[0]["text"].strip() == eng[0]["text"].strip():
                same += 1
        print(f"  beams {b}, threads {t}: {same}")


def wasm_table(results):
    order = ["native", "node-simd128", "node-scalar", "wasmtime-simd128", "wasmtime-scalar"]
    runs = {}
    for path in sorted(glob.glob(os.path.join(results, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            runs[os.path.splitext(os.path.basename(path))[0]] = [
                json.loads(line) for line in f if line.startswith('{"kind":"clip"')]
    names = [n for n in order if n in runs] + sorted(set(runs) - set(order))
    for beams in sorted({r["beams"] for rows in runs.values() for r in rows}):
        print(f"\nbeams {beams}, 1 thread: median wall ms per transcription (encoder-only ms)")
        print(" | ".join(["clip", "sec"] + names + ["node simd/native", "simd speedup (node)"]))
        clips = [r for r in runs.get("native", next(iter(runs.values()))) if r["beams"] == beams]
        for ref in clips:
            cell = {n: next((r for r in runs[n] if r["beams"] == beams and r["clip"] == ref["clip"]),
                            None) for n in names}
            row = [ref["clip"], f"{ref['seconds']:.1f}"]
            row += [f"{c['median_ms']:.0f} ({c['encoder_ms']:.0f})" if c else "-"
                    for c in cell.values()]
            ns, nat, sc = cell.get("node-simd128"), cell.get("native"), cell.get("node-scalar")
            row.append(f"{ns['median_ms'] / nat['median_ms']:.1f}x" if ns and nat else "-")
            row.append(f"{sc['median_ms'] / ns['median_ms']:.2f}x" if ns and sc else "-")
            print(" | ".join(row))
        for n in names[1:]:
            same = sum(1 for ref in clips for r in runs[n] if r["beams"] == beams
                       and r["clip"] == ref["clip"] and r["text"] == ref["text"])
            print(f"  {n}: text identical to native on {same} of {len(clips)} clips")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"clips": make_clips, "engine": engine, "table": table, "wasmtable": wasm_table}[cmd](*args)
