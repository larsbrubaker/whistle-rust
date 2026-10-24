#!/usr/bin/env bash
# Runs the whistle-rust test suite (unit and integration tests, release build) on wasm32-wasip1
# under wasmtime, once with simd128 and once scalar, so the token contract and every stage bound
# are checked on the code the browser runs (docs/design/performance.md "Single-threaded wasm").
#
#   scripts/wasm-test.sh             both builds
#   scripts/wasm-test.sh simd128     one build (simd128 or scalar); extra args go to the test
#                                    binaries, e.g. `scripts/wasm-test.sh scalar contract`
#
# Tests find fixtures through CARGO_MANIFEST_DIR, an absolute host path baked in at compile time,
# so the repo is preopened at that same absolute path, and each git-ignored data directory
# (models/, needle-reference/, test-data/fixtures-local/, oracle/.engine), which a worktree links
# from the main checkout, is preopened at the path the tests use: WASI does not follow a symlink
# out of a preopened directory. A panic aborts a wasm test binary (panic = abort), so a crashed
# binary is rerun with the crashed test skipped until it completes; every failure is listed at
# the end and the script exits non-zero if there was any.
set -uo pipefail
cd "$(dirname "$0")/.."
repo=$(pwd -P)
WASMTIME=${WASMTIME:-wasmtime}
modes=(simd128 scalar)
if [ "${1:-}" = simd128 ] || [ "${1:-}" = scalar ]; then
  modes=("$1")
  shift
fi
rustup target list --installed | grep -qx wasm32-wasip1 || rustup target add wasm32-wasip1

dirs=(--dir "$repo::$repo")
for d in models needle-reference test-data/fixtures-local oracle/.engine; do
  [ -e "$d" ] && dirs+=(--dir "$(cd "$d" && pwd -P)::$repo/$d")
done

failures=()
for mode in "${modes[@]}"; do
  flags=''
  [ "$mode" = simd128 ] && flags='-C target-feature=+simd128'
  echo "== wasm32-wasip1 $mode: building tests"
  bins=$(RUSTFLAGS="$flags" cargo test --release --target wasm32-wasip1 -p whistle-rust \
    --target-dir "target/wasm-test-$mode" --no-run --message-format=json 2>/dev/null |
    python3 -I -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["profile"]["test"]:
        print(m["executable"])')
  if [ -z "$bins" ]; then
    echo "== $mode: build FAILED"
    failures+=("$mode: build")
    continue
  fi
  for bin in $bins; do
    name=$(basename "$bin" .wasm | sed 's/-[0-9a-f]*$//')
    skips=()
    while :; do
      log=$(mktemp)
      "$WASMTIME" run -W max-memory-size=4294967296 "${dirs[@]}" "$bin" "$@" ${skips[@]+"${skips[@]}"} \
        --test-threads=1 2>&1 | tee "$log" | grep -E '^test |^test result|panicked|^  left|^ right'
      status=${PIPESTATUS[0]}
      if grep -q '^test result' "$log"; then
        if [ "$status" -ne 0 ]; then
          while read -r f; do failures+=("$f"); done < <(sed -n '/^failures:$/,/^test result/p' "$log" |
            grep -E '^    [A-Za-z0-9_:]+$' | sed "s|^ *|$mode $name: |" | sort -u)
        fi
        rm -f "$log"
        break
      fi
      crashed=$(grep -E '^test [^ ]+ \.\.\. ' "$log" | tail -1 | sed -E 's/^test ([^ ]+) .*/\1/')
      rm -f "$log"
      if [ -z "$crashed" ] || [[ " ${skips[*]-} " == *" $crashed "* ]]; then
        failures+=("$mode $name: binary aborted outside a test")
        break
      fi
      echo "== $mode $name: $crashed aborted the binary; rerunning without it"
      failures+=("$mode $name: $crashed (aborted)")
      skips+=(--skip "$crashed" --exact)
    done
  done
done

if [ ${#failures[@]} -ne 0 ]; then
  echo "WASM TESTS FAILED:"
  printf '  %s\n' "${failures[@]}"
  exit 1
fi
echo "WASM TESTS OK (${modes[*]})"
