#!/usr/bin/env bash
# Single-threaded wasm benchmark and profile. The browser build has no threads, so one wasm
# thread is the performance floor and what performance work is chosen by. Builds crates/whistle-rust/examples/bench.rs for wasm32-wasip1 twice, with simd128 and
# scalar, plus the native release build, then times all of them at one thread on the 8 speech
# clips and the ~26.6 s long clip scripts/bench_tools.py writes (temp dir, nothing committed):
# native, Node (V8, via scripts/wasm-bench.mjs) and wasmtime, at beams 1 and 3, in ROUNDS rounds
# that interleave the runtimes. Prints the table (each cell the median of the per-round medians)
# and keeps the raw JSON lines under target/wasm-bench-results/<timestamp>/.
#
#   scripts/wasm-bench.sh            ROUNDS=3 rounds of RUNS=3 timed calls per clip (one warm-up
#                                    each); override either via env
#   scripts/wasm-bench.sh profile    node --cpu-prof on a simd128 build with debug info: a 3 s
#                                    clip at beams 1 and 3, the long clip at beams 1; summarized
#                                    by scripts/wasm_profile.py, profiles kept in target/wasm-prof
#
# The bench example needs no cfg changes for wasm32-wasip1: std time and file I/O work under
# WASI, and the library already runs one thread on wasm32. What the runs show (2026-10-24, M5,
# under heavy load, so as guides): simd128 is several times faster than scalar wasm everywhere
# (Node 3-10x, wasmtime 2-4x), so simd128 is the build to ship; Node simd128 runs at about
# 1.05-1.1x native single-thread on the 3 s clips.
#
# The profile counts only samples under the timed transcribe_wav calls; the stage comes from
# the call stack, the kind from the innermost recognisable frame, and self time includes what is
# inlined (the simd128 dot product, index decode and lookup all sit inside
# PackedMatrix::rows_into, the panel kernel inside linear_panels_into). Measured 2026-10-24 at
# load average 22-27 (shares are a guide), en_samantha at beams 1 and 3 and the 26.6 s clip at 1:
#
#   bucket                               3 s b1   3 s b3   26.6 s b1
#   decoder step                          36.4%    57.8%    37.1%
#     rotated packed kernel (rows_into)   27.6%    41.7%    25.5%
#   encoder                               50.8%    34.0%    52.6%
#     dense kernel (linear_panels_into)   25.8%    17.5%    25.6%
#     Hadamard MLP                         9.4%     6.4%     9.6%
#     band decode (packed CQ, per call)    6.4%     3.7%     0.7%
#     self-attention                       1.0%     1.0%     8.4%
#   decoder setup (cross-attention)        8.5%     5.5%     6.5%
#   front end + VAD                        4.0%     2.4%     3.7%
#     mel/log                              0.8%     0.3%     0.6%
#     FFT                                  2.3%     1.6%     2.4%
#
# Top functions on the 3 s clip at beams 1: linear_panels_into 32.3%, rows_into 27.6%,
# band_into 7.5%, kron_apply 6.7%, fake_quant_rows 3.0%, HadamardMlp::forward 2.9%,
# Fft::forward 2.3%, PackedMatrix::project 2.1%. The rotated kernel's per-16-weight work is 4
# plane swizzles and 8 byte interleaves for the lookup against 4 loads, 4 multiplies and 4 adds
# per row; cutting it further needs fewer bytes per value or more rows per decode.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-3}
ROUNDS=${ROUNDS:-3}
MODEL=models/whistle.cact
NODE=${NODE:-node}
WASMTIME=${WASMTIME:-wasmtime}
WASM=wasm32-wasip1/release/examples/bench.wasm
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

rustup target list --installed | grep -qx wasm32-wasip1 || rustup target add wasm32-wasip1
build() { # target dir, rustflags, extra env...
  local dir=$1 flags=$2
  shift 2
  env "$@" RUSTFLAGS="$flags" cargo build --release --target wasm32-wasip1 -p whistle-rust \
    --example bench --target-dir "$dir"
}
python3 -I scripts/bench_tools.py clips "$tmp"
# The worktree's models/ may be a symlink out of the preopened directory: preopen its target.
models_dir=$(cd models && pwd -P)
node_run() { "$NODE" --no-warnings scripts/wasm-bench.mjs --preopen "models=$models_dir" \
  --preopen "$tmp=$tmp" "$@"; }
wasmtime_run() { local wasm=$1; shift; "$WASMTIME" run --dir . --dir "$models_dir::models" \
  --dir "$tmp::$tmp" "$wasm" "$@"; }

echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), commit $(git rev-parse --short HEAD)"
echo "node $("$NODE" --version), $("$WASMTIME" --version)"
echo "load average at start: $(uptime | sed 's/.*load averages*: //')"

if [ "${1:-}" = profile ]; then
  build target/wasm-bench-prof '-C target-feature=+simd128' CARGO_PROFILE_RELEASE_DEBUG=1
  out=target/wasm-prof
  rm -rf "$out" && mkdir -p "$out"
  wasm=target/wasm-bench-prof/$WASM
  for spec in "short-b1 1 test-data/clips/en_samantha.wav" "short-b3 3 test-data/clips/en_samantha.wav" \
    "long-b1 1 $tmp/long.wav"; do
    read -r name beams clip <<<"$spec"
    "$NODE" --no-warnings --cpu-prof --cpu-prof-interval 200 --cpu-prof-dir "$out/$name" \
      scripts/wasm-bench.mjs --preopen "models=$models_dir" --preopen "$tmp=$tmp" "$wasm" \
      --runs "$RUNS" --beams "$beams" --threads 1 "$clip"
    echo "== $name: $clip, beams $beams"
    python3 -I scripts/wasm_profile.py "$out/$name"/*.cpuprofile
  done
  echo "load average at end: $(uptime | sed 's/.*load averages*: //')"
  exit 0
fi

build target/wasm-bench '-C target-feature=+simd128'
build target/wasm-bench-scalar ''
cargo build --release -p whistle-rust --example bench
clips=(test-data/clips/*_*.wav "$tmp/long.wav")
results=target/wasm-bench-results/$(date +%Y%m%d-%H%M%S)
mkdir -p "$results"
runtimes=(native node-simd128 node-scalar wasmtime-simd128 wasmtime-scalar)
run_one() { # runtime, bench args...
  local name=$1
  shift
  case $name in
    native) target/release/examples/bench "$@" ;;
    node-simd128) node_run target/wasm-bench/$WASM "$@" ;;
    node-scalar) node_run target/wasm-bench-scalar/$WASM "$@" ;;
    wasmtime-simd128) wasmtime_run target/wasm-bench/$WASM "$@" ;;
    wasmtime-scalar) wasmtime_run target/wasm-bench-scalar/$WASM "$@" ;;
  esac
}
# Load drifts over a run, so each round runs every runtime once per beam count, and the starting
# runtime rotates from round to round: no runtime always runs first or last, and every cell is
# the median of ROUNDS per-round medians rather than one sample of whatever load it ran under.
for ((round = 0; round < ROUNDS; round++)); do
  for beams in 1 3; do
    args=(--model "$MODEL" --runs "$RUNS" --beams "$beams" --threads 1 "${clips[@]}")
    for ((i = 0; i < ${#runtimes[@]}; i++)); do
      name=${runtimes[$(((i + round) % ${#runtimes[@]}))]}
      run_one "$name" "${args[@]}" >>"$results/$name.jsonl"
    done
  done
  echo "round $((round + 1)) of $ROUNDS done, load average $(uptime | sed 's/.*load averages*: //')"
done
echo "load average at end: $(uptime | sed 's/.*load averages*: //')"
echo "raw JSON lines: $results"
python3 -I scripts/bench_tools.py wasmtable "$results"
