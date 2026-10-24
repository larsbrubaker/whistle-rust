#!/usr/bin/env bash
# The browser demo's gate (demo/, its own Cargo workspace; CLAUDE.md "Commands"). The library's
# gate is scripts/gates.sh and never builds the demo, so the library needs no GUI toolkit; this
# one needs the sibling ../agg-gui, ../agg-rust and ../tess2-rust checkouts, wasm-pack and bun.
#
# Steps, each timed; every step runs even after an earlier one fails:
#   fmt      cargo fmt --check on the demo workspace
#   clippy   cargo clippy, native (all targets) and wasm32 (the two wasm crates), -D warnings
#   tests    cargo test on the demo workspace (the UI's state and layout tests)
#   wasm     demo/web/build-wasm.sh (the UI module, and the worker module with simd128)
#   tsc      tsc --noEmit on demo/web
#   bundle   bun run build (vite) on demo/web
set -uo pipefail
cd "$(dirname "$0")/.."
manifest=demo/Cargo.toml
failed=()
step() {
  local name=$1
  shift
  local start=$SECONDS
  echo "== $name: $*"
  if "$@"; then
    echo "== $name ok ($((SECONDS - start)) s)"
  else
    echo "== $name FAILED ($((SECONDS - start)) s)"
    failed+=("$name")
  fi
}
step fmt cargo fmt --manifest-path "$manifest" --all --check
step clippy cargo clippy --manifest-path "$manifest" --workspace --all-targets -- -D warnings
step clippy-wasm cargo clippy --manifest-path "$manifest" -p whistle-demo-web -p whistle-demo-worker \
  --target wasm32-unknown-unknown -- -D warnings
step tests cargo test --manifest-path "$manifest" --workspace
step wasm demo/web/build-wasm.sh
in_web() { (cd demo/web && "$@"); }
step deps in_web bun install --frozen-lockfile
step tsc in_web bunx tsc --noEmit -p .
step bundle in_web bun run build
if [ ${#failed[@]} -ne 0 ]; then
  echo "DEMO GATES FAILED: ${failed[*]}"
  exit 1
fi
echo "DEMO GATES OK"
