#!/usr/bin/env bash
# Benchmark whistle-rust against the closed Needle engine CLI on the same clips and machine.
# The engine is only ever run and timed.
#
# Builds crates/whistle-rust/examples/bench.rs in release, writes a ~25-30 s clip (speech clips
# joined by 0.3 s gaps) and a 0.2 s silent clip to a temp dir (no audio is committed), then runs
# ROUNDS rounds. Each round runs, in turn: our example at beams 1 and 3 with threads 1 and 4
# (RUNS timed calls per clip, one warm-up), then the engine at --threads 1 and 4 (RUNS
# invocations per clip, wall time per process including start and model load). The engine on the
# silent clip is the floor to subtract. Prints the table; raw JSON lines stay in the temp dir.
#
#   scripts/bench.sh            RUNS=10 ROUNDS=3 by default; override through the environment
#
# Like for like: the engine's wall time includes process start and model load, ours is a warm
# call on a loaded model. The engine on 0.2 s of silence (gated out before decoding) costs about
# 10.4 ms at 1 thread and 12.6 ms at 4, the floor "net" subtracts; our own load (read plus
# Whistle::from_cact) is about 4 ms and is not in our columns. The CLI exposes no beam count;
# its text equals ours at beams 3 on 8 of the 9 clips and at beams 1 on 7, so ours at beams 3
# is the fair comparison (whistle::DEFAULT_BEAMS).
#
# Measured 2026-10-24 on an Apple M5 (10 cores), 24 GB, release, RUNS=5 ROUNDS=2, load average
# 9.5-11 (about +-15% on any cell), with the beams stepped one at a time (before
# DecoderState::step_batch): median ms per transcription; b = beams, t = threads, enc t4 =
# features plus encoder at 4 threads; eng ttft / tps = the engine's own ttft_ms / decode_tps at
# 1 thread.
#
#   clip         sec  b1t1 b1t4 b3t1 b3t4 enc t4 | eng t1 raw/net  t4 raw/net  ttft  tps
#   de_anna      2.6    99   94  186  209    40 |       59/48       53/40     11  572
#   en_daniel    3.4   101   98  176  193    51 |       59/49       47/34     13  581
#   en_samantha  3.1   102   92  177  196    46 |       48/38       46/33     12  600
#   es_monica    2.6   107   89  205  229    36 |       55/45       53/41     11  600
#   fr_thomas    3.0   111   97  234  272    41 |       61/51       65/52     16  442
#   it_alice     2.3   108   80  161  186    37 |       61/50       41/29     13  452
#   nl_xander    2.1    88   74  146  169    36 |       51/40       40/27     11  521
#   pl_zosia     2.5    93   65  179  196    38 |       77/66       55/42     14  427
#   long        26.6   873  694 1831 1856   338 |     392/381     250/237    152  351
#
# So greedy is about 2x the engine on 3 s clips (roughly 40 ms of ours is the encoder, against
# the engine's 11-16 ms to first token) and 2.2-2.8x on 26.6 s, where our encoder alone exceeds
# the engine's whole 4-thread call; 3 beams were 3.5-7x. The engine's threads cut its long clip
# 1.6x, ours 1.25x at one beam: the decoder step is single-threaded.
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
