# whistle-rust browser demo

Live speech-to-text in a web page, on the whistle-rust library: press Record and speak, or
pick or drop a WAV file. Deployed at
[larsbrubaker.github.io/whistle-rust](https://larsbrubaker.github.io/whistle-rust/).

This directory is its own Cargo workspace, so the library's workspace, CI test job and crates.io
package never depend on the GUI toolkit.

| path | what |
|---|---|
| `whistle-demo/` | the agg-gui UI and its state (rlib; unit and layout tests run natively) |
| `whistle-demo-web/` | the UI's wasm on agg-gui-web-shell, on the page's main thread |
| `whistle-demo-worker/` | the transcription Web Worker's wasm: `Engine` wraps `Whistle` and `Stream` |
| `web/` | the page (bun + vite): `main.ts` glue, `worker.ts`, `model.ts` (download, SHA-256, Cache API), `microphone.ts` (AudioWorklet, ScriptProcessor fallback) |

The model runs only in the worker, one thread, so the UI never blocks; the worker's wasm holds
no GUI code (about 450 KB). The page fetches `whistle.cact` from beside itself (`?model=URL`
overrides it; Hugging Face is the fallback), checks its size and SHA-256 against
`whistle::model_file`, and caches it in the Cache API under its checksum.

## Build and run

Needs [wasm-pack](https://rustwasm.github.io/wasm-pack/), [bun](https://bun.sh) and the
sibling checkouts `../agg-gui`, `../agg-rust` and `../tess2-rust` next to this repository
(github.com/larsbrubaker/<name>).

```bash
scripts/fetch-weights.sh                                   # models/whistle.cact
ln -sf ../../../models/whistle.cact demo/web/public/whistle.cact
demo/web/build-wasm.sh                                     # demo/web/pkg/ui and pkg/worker
(cd demo/web && bun install && bun run dev)
scripts/demo-gates.sh                                      # the demo's full check
```

## The transcript's scroll view

The transcript sits in a `ScrollView` stuck to the bottom, so the newest words stay in view
while streaming; its content is a `Rebuilder` that rebuilds the text when the state changes.
The column inside the `Rebuilder` is deliberately **not** top-anchored. A `ScrollView`
measures its content at an unbounded height and lays it out again at the measured height only
when one of the content's direct children lies outside that box; a top-anchored column behind a
`Rebuilder` is not a direct child, so its labels stay near y = 1e308, off screen. The
minimal repro is `rebuilder_hides_a_top_anchored_column_from_scroll_view_relayout` in
`whistle-demo/src/ui_tests.rs` (`ScrollView > Rebuilder > FlexColumn::with_top_anchor(true) >
Label`); it fails once agg-gui handles the case, and then the workaround can go.

## simd128 and the UI module

The worker module is built with `-C target-feature=+simd128`; the UI module is not. With
simd128 on, rustc 1.99.0 (LLVM) aborts compiling agg-gui:

```
rustc-LLVM ERROR: Cannot select: v4f32 = WebAssemblyISD::CONVERT_LOW_U ...
  v4i32 = WebAssemblyISD::VEC_SHR_U ..., Constant:i32<24>
    v4i32 = vselect ...
In function: ...color_wheel_picker::preview_swatches::PreviewSwatches ... paint
```

The loop vectorizer turns a per-pixel choice between two `[u8; 4]` colours into a `vselect` of
packed `u32`s, takes the alpha byte with a shift by 24 and converts it to `f32`; the wasm
backend folds that into a `CONVERT_LOW_U` node for which it has no `v4f32` pattern. Minimal
trigger (`rustc --crate-type cdylib --target wasm32-unknown-unknown -C opt-level=3
-C target-feature=+simd128`):

```rust
#[no_mangle]
pub fn alphas(w: u32, old_rgba: [u8; 4], new_rgba: [u8; 4], out: &mut [u8]) {
    let mid = w / 2;
    for col in 0..w {
        let rgba = if col < mid { old_rgba } else { new_rgba };
        let t = rgba[3] as f32 / 255.0;
        out[col as usize] = (rgba[0] as f32 * t).round().clamp(0.0, 255.0) as u8;
    }
}
```

(Converting the alpha alone, `out[col] = rgba[3] as f32` into an `&mut [f32]`, compiles.) It is
an LLVM backend bug, not agg-gui's; RUSTFLAGS apply to every crate in a build, so the UI module
is built without simd128 rather than patching agg-gui. The UI does no heavy arithmetic, so it
loses nothing that matters; the model's kernels, in the worker, keep simd128.
