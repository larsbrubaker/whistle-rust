#!/usr/bin/env bash
# Benchmark whistle-rust against the closed Needle engine CLI on the same clips and machine
# (docs/design/performance.md, "Against the engine"). The engine is only ever run and timed.
#
# Builds crates/whistle-rust/examples/bench.rs in release, writes a ~25-30 s clip (speech clips
# joined by 0.3 s gaps) and a 0.2 s silent clip to a temp dir (no audio is committed), then runs
# ROUNDS rounds. Each round runs, in turn: our example at beams 1 and 3 with threads 1 and 4
# (RUNS timed calls per clip, one warm-up), then the engine at --threads 1 and 4 (RUNS
# invocations per clip, wall time per process including start and model load). The engine on the
# silent clip is the floor to subtract. Prints the table; raw JSON lines stay in the temp dir.
#
#   scripts/bench.sh            RUNS=10 ROUNDS=3 by default; override through the environment
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-10}
ROUNDS=${ROUNDS:-3}
PY=oracle/.venv/bin/python
ENGINE=oracle/.engine/macos-arm64/needle
MODEL=models/whistle.cact
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cargo build --release -p whistle-rust --example bench
BENCH=target/release/examples/bench
$PY -I scripts/bench_tools.py clips "$tmp"
clips=(test-data/clips/*_*.wav "$tmp/long.wav")
echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), commit $(git rev-parse --short HEAD)"
echo "load average at start: $(uptime | sed 's/.*load averages*: //')"
for round in $(seq "$ROUNDS"); do
  echo "round $round of $ROUNDS"
  for beams in 1 3; do
    for threads in 1 4; do
      "$BENCH" --model "$MODEL" --runs "$RUNS" --beams "$beams" --threads "$threads" "${clips[@]}" \
        >>"$tmp/ours.jsonl"
    done
  done
  for threads in 1 4; do
    $PY -I scripts/bench_tools.py engine "$ENGINE" "$MODEL" "$threads" "$RUNS" \
      "$tmp/tiny.wav" "${clips[@]}" >>"$tmp/engine.jsonl"
  done
done
echo "load average at end: $(uptime | sed 's/.*load averages*: //')"
$PY -I scripts/bench_tools.py table "$tmp"
