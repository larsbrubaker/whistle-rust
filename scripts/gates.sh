#!/usr/bin/env bash
# The full gate run for whistle-rust: the one command the orchestrator runs on main after a merge
# and an implementer runs before committing (CLAUDE.md "Testing framework" and "Commands").
#
# Steps, in order, each timed; every step runs even after an earlier one fails, so one run
# reports everything that is broken:
#   fmt       cargo fmt --check
#   clippy    cargo clippy --workspace --all-targets -- -D warnings
#   tests     cargo nextest run --workspace (cargo test when nextest is not installed)
#   doctests  cargo test --workspace --doc
#   wasm      cargo build -p whistle-rust --target wasm32-unknown-unknown (the library must
#             build for the browser; GATES_WASM=0 skips it)
#   package   scripts/check-package.sh (the packaged crate builds, documents and compiles its
#             doctests on its own, under 1 MB; a few seconds; GATES_PACKAGE=0 skips it)
set -uo pipefail
cd "$(dirname "$0")/.."
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
step fmt cargo fmt --check
step clippy cargo clippy --workspace --all-targets -- -D warnings
if cargo nextest --version >/dev/null 2>&1; then
  step tests cargo nextest run --workspace
else
  step tests cargo test --workspace
fi
step doctests cargo test --workspace --doc
if [ "${GATES_WASM:-1}" != 0 ]; then
  step wasm cargo build -p whistle-rust --target wasm32-unknown-unknown
fi
if [ "${GATES_PACKAGE:-1}" != 0 ]; then
  step package scripts/check-package.sh
fi
if [ ${#failed[@]} -ne 0 ]; then
  echo "GATES FAILED: ${failed[*]}"
  exit 1
fi
echo "GATES OK"
