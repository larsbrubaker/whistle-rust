# whistle-rust demo

Live speech-to-text in a web page, on the whistle-rust library: press Record and speak, or
pick or drop a WAV file. Deployed at
[larsbrubaker.github.io/whistle-rust](https://larsbrubaker.github.io/whistle-rust/). The same
UI also runs in a desktop window ([below](#desktop-build)).

This directory is its own Cargo workspace, so the library's workspace, CI test job and crates.io
package never depend on the GUI toolkit.

| path | what |
|---|---|
| `whistle-demo/` | the agg-gui UI and its state (rlib; unit and layout tests run natively) |
| `whistle-demo-web/` | the UI's wasm on agg-gui-web-shell, on the page's main thread |
| `whistle-demo-worker/` | the transcription Web Worker's wasm: `Engine` wraps `Whistle` and `Stream` |
| `whistle-demo-native/` | the desktop build: the same UI on agg-gui-shell, the model on an engine thread |
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

## Desktop build

```bash
scripts/fetch-weights.sh                         # models/whistle.cact (optional, see below)
cd demo && cargo run -p whistle-demo-native --release
cargo run -p whistle-demo-native --release -- ../test-data/clips/en_samantha.wav
cargo run -p whistle-demo-native --release -- --help
```

(`.claude/launch.json` has it as `whistle-demo-native`.) The same whistle-demo UI in a native
window on agg-gui-shell: transcribe a WAV picked in the OS dialog, dropped on the window, opened
from the Finder or named on the command line. The parts:

| file | what |
|---|---|
| `engine.rs` | the engine thread, the desktop's Web Worker: owns the model and the live stream, answers `ToEngine` messages in order with timed `FromEngine` replies |
| `bridge.rs` | the UI's `DemoHost`: sends the UI's requests to the engine and applies the replies to `DemoState` |
| `capture.rs` | the live-audio seam: a `Capture` delivers mono chunks to a sink on its own thread; `WavCapture` plays a file through it |
| `model.rs` | where the model comes from, checked against `whistle::model_file`'s size and SHA-256 |
| `shell_host.rs` | the shell's hooks: apply what arrived on each idle pass; open what the OS asks to open |

**Model.** `--model FILE` loads only that file. Otherwise the checkout's
`models/whistle.cact` (what `scripts/fetch-weights.sh` writes), then the cache
(`~/Library/Caches`, `%LOCALAPPDATA%` or `$XDG_CACHE_HOME`, under
`whistle-rust/<sha256>/whistle.cact`), then a download of `whistle::model_file::MODEL_URL`
into the cache. Every copy is checked; a damaged cached copy is deleted, and a download that
fails the check is never cached. The download is ureq over rustls with the RustCrypto
provider (pure Rust).

**Threads.** The UI thread never runs the model. The library splits each pass's projections
over one worker thread per core but one, 1 to 4 (`--threads N` to choose).

**Microphone.** Not wired in yet: Record is shown unavailable with a note. A live source
plugs into the `Capture` trait (start with a sink, stop with every chunk delivered); the
bridge forwards its chunks to the engine's stream as the browser's microphone feeds its
worker. `--mic-from-wav FILE` plays a file through that path in real time (`--record` starts
it once the model is ready), so the live text, committed and pending, can be seen without a
microphone.

**Device rates and sessions.** Audio at a device's rate (44.1 or 48 kHz) goes through one
`whistle::audio::StreamResampler` per stream, in the engine here and in the browser's worker
alike, so 100 ms pushes come out exactly as the whole recording would: resampling each push
on its own put a seam at every boundary, enough to read "printing" as "granting". Each
recording and file is a numbered session; the engine drops audio for a stream that has ended
(a failed push ends its stream), and the bridge drops replies to any session but the current
one, so a late update from an old recording never shows in a new one.

**Scripted runs.** Every final transcript is also printed to stdout with the status line.
`--screenshot FILE.png` waits (up to 12 s) for the first transcript or error, saves the
window and exits:

```bash
cargo run -p whistle-demo-native --release -- \
  ../test-data/clips/en_samantha.wav --screenshot /tmp/whistle-native.png
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
