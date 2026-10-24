#!/usr/bin/env bash
# Builds the demo's two wasm modules with wasm-pack into demo/web/pkg/:
#   pkg/ui/      demo/whistle-demo-web, the agg-gui UI on the page's main thread (src/main.ts)
#   pkg/worker/  demo/whistle-demo-worker, the model in the transcription worker (src/worker.ts)
# The worker is built with simd128, which every current browser runs (Chrome 91+, Firefox 89+,
# Safari 16.4+) and the library's kernels use. The UI is built without it: rustc 1.99's LLVM
# fails to compile agg-gui with simd128 on (demo/README.md, "simd128 and the UI module"), and
# the UI does no heavy arithmetic.
# Usage: demo/web/build-wasm.sh [--dev]
set -euo pipefail
cd "$(dirname "$0")/.."
profile=--release
if [ "${1:-}" = "--dev" ]; then
  profile=--dev
fi
wasm-pack build whistle-demo-web "$profile" \
  --target web --out-dir ../web/pkg/ui --no-typescript
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" \
  wasm-pack build whistle-demo-worker "$profile" \
  --target web --out-dir ../web/pkg/worker --no-typescript
