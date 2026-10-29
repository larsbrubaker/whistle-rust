#!/usr/bin/env bash
# Builds the demo's two wasm modules with wasm-pack into demo/web/pkg/:
#   pkg/ui/      demo/whistle-demo-web, the agg-gui UI on the page's main thread (src/main.ts)
#   pkg/worker/  demo/whistle-demo-worker, the model in the transcription worker (src/worker.ts)
# The worker is built with simd128, which every current browser runs (Chrome 91+, Firefox 89+,
# Safari 16.4+) and the library's kernels use. The UI is built without it, and loses nothing
# that matters since it does no heavy arithmetic: with simd128 on, rustc 1.99's LLVM aborts
# compiling agg-gui ("Cannot select: v4f32 = WebAssemblyISD::CONVERT_LOW_U", in
# color_wheel_picker's PreviewSwatches::paint). The loop vectorizer turns a per-pixel choice
# between two [u8; 4] colours into a vselect of packed u32s, takes the alpha byte with a shift by
# 24 and converts it to f32; the wasm backend folds that into a CONVERT_LOW_U node it has no
# v4f32 pattern for. Minimal trigger (rustc --crate-type cdylib --target wasm32-unknown-unknown
# -C opt-level=3 -C target-feature=+simd128):
#   #[no_mangle]
#   pub fn alphas(w: u32, old_rgba: [u8; 4], new_rgba: [u8; 4], out: &mut [u8]) {
#       let mid = w / 2;
#       for col in 0..w {
#           let rgba = if col < mid { old_rgba } else { new_rgba };
#           let t = rgba[3] as f32 / 255.0;
#           out[col as usize] = (rgba[0] as f32 * t).round().clamp(0.0, 255.0) as u8;
#       }
#   }
# It is an LLVM backend bug, not agg-gui's, and RUSTFLAGS apply to every crate in a build, so
# the UI module is built without simd128 rather than patching agg-gui.
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
