# whistle-rust porting plan — open work

Open work only, in landing order. Each item says what a caller can do once it lands and what it
needs. Delete an item when it lands; rewrite it to what is left when part of it lands. How the
model, the archive and the oracle work: `crates/whistle-rust/src/lib.rs`, the `oracle/` script
headers. The verification bar for every stage: CLAUDE.md "Tolerance policy".

## 6. Performance

Transcription is fast enough for push-to-talk on every target, and the weights stay packed (the
headers of `kernels/packed.rs` and `tests/transcribe/timing.rs`: ~41 MB resident after binding
natively, ~59 MB on wasm simd128 with its byte codes; a 3 s clip in ~82 ms natively and ~87 ms on
Node simd128 at one beam, one thread); what is left.

**Priority: single-threaded wasm32 (simd128).** The browser build has no threads, and phones and
slower computers on the web are where transcription is slowest, so performance work is chosen
and measured for one thread on wasm first; native threads are a bonus.

- Re-measure on a quiet machine (beams 1 and 3, native and Node simd128): the vectorised
  `expf`/`sigmoid`/`silu` (`math::expf_lanes`, NEON/SSE2/simd128, bit-identical to scalar on all
  2^32 inputs) should win back the deterministic math's ~10% native / ~8% Node cost; the first
  numbers were taken at load ~40. CI's x86_64 jobs (Linux, Windows) confirm the SSE2 and AVX2
  paths bit for bit.
- Optimise for one wasm thread, by the Node (V8) profile (`scripts/wasm-bench.sh profile`, after
  the byte codes and mel spans): the encoder's dense panel kernel is now the largest single
  function (~26-32% of a 3 s clip at one beam), the decoder's rotated kernel 28% at one beam and
  42% at three, the Hadamard MLP ~9%, decoder setup ~8%. Rerun the full `scripts/wasm-bench.sh`
  table on a quiet machine, and measure in the browser through the demo (item 8).
- The 3 s encoder alone is 5-12% slower than the dense build (it decodes ~19M packed weights per
  call); cut that if the encoder dominates a real workload.
- Close the gap to the engine (`scripts/bench.sh`, whose header holds the comparison
  with the engine): about 2x slower at one beam and 3.5-7x at the default three before the beams'
  decoder steps were batched (`DecoderState::step_batch`, bit-identical; 20-30% faster at 3-5
  beams on one thread natively, wasm not yet measured cleanly). Next: what a batched step's cost
  is made of, then the encoder (~40 ms on a 3 s clip against the engine's 11-16 ms to first
  token).
- Smaller wins not yet tried: a persistent thread pool so single-row projections can split; a
  transposed panel layout or an f32/bf16 copy of the decoder weights (trading memory for time);
  checking that `kron_apply` vectorises under simd128; reusing the Hadamard MLP's scratch; and
  finding which clip's wasm text differs from native.

## 7. Designed decode features

The behaviours only the closed engine has, designed here and held to the engine transcripts. Beam
search (default 3 beams), keyword biasing, word timestamps, windowing and streaming are in
(`src/transcribe/search.rs`, `keywords.rs`, `repeat.rs`, `align.rs`, `window.rs`, `stream.rs`);
what is left:

- Streaming quality: a word committed and then read differently by a later pass stays as
  committed (no retraction), so live text can keep an early misreading ("Q20" for "cube 20",
  "matter" for "Matterhackers" parts). A re-read with other characters is no longer committed
  again when its words lie in time where the committed ones were (`realign::re_read`), but a
  pass word that straddles the committed end still is: "Q20" committed, then "20mm" read as one
  word over "20" and the new "mm", streams "Q20 20mm" (en_samantha, 0.02 hiss, 7500 samples of
  leading silence; `tests/streaming/offsets.rs` sweeps this), and a merged re-read can swallow
  part of the next word. Per-update cost grows with the utterance (up to a 30 s decode per
  step); re-measure on wasm and consider bounding it.
- Keyword biasing at 5+ beams finds fewer keywords than at 1 or 3 (partial keyword paths crowd
  the beam): per-beam-count tuning or a cap on hypotheses inside a keyword.
- Garbage audio still decodes to confident nonsense, sometimes in the wrong language (the
  repetition guard only stops loops). Measured at one beam: the 8 contract clips have a
  language-tag probability of at least 0.995 and a mean token log-probability between −0.04
  and −0.39; of 48 garbled variants, the 8 decoded in the wrong language all had tag
  probability ≤ 0.87 and mean log-probability ≤ −0.55, but some right-language garbage is as
  low, and pure noise reads "Thank you." at 0.74 / −0.01. A rule such as "emit nothing below
  0.9 and −0.5" fits these numbers; it needs real microphone and accented recordings to set,
  a choice between emitting nothing and exposing a confidence on `Transcript`, and a rule for
  streams (language forced after the first commit).
- More clips (long audio, noise, real microphone recordings) via `oracle/clips.json`; the
  alignment heads were fitted on the 8 contract clips and hold on 3 held-out clips so far.

## 8. Integration

- The browser demo (`demo/`, agg-gui: live microphone transcript, a WAV file path, the model
  downloaded, checksummed and cached, transcription in a Web Worker) is in, and deploys to
  https://larsbrubaker.github.io/whistle-rust/ on every push to main. Still unverified by hand:
  live microphone streaming (per-update time and real-time factor in the browser), the
  word-timestamps display, the Hugging Face fallback, the file picker.
- The desktop demo (`demo/whistle-demo-native`, agg-gui-shell) transcribes WAV files and
  records from the microphone through agg-mic. Still unverified by hand: a real recording on
  each OS (the tests use a fake agg-mic backend). Left:
  - macOS permission for a bundled build: an app bundle with `NSMicrophoneUsageDescription` in
    its Info.plist and, if it is signed with the hardened runtime, the
    `com.apple.security.device.audio-input` entitlement. Today a `cargo run` binary borrows the
    launching terminal's or IDE's permission (the header of the demo's `mic.rs`).
  - Falling behind a live microphone: merging queued pushes saves nothing, since `Stream::push`
    runs one pass per second of audio whatever the chunking. At 1 beam one thread on an M5
    peaks at 446 ms per 1 s step, at 3 beams 965 ms (`tests/streaming/measure.rs`), so a slower
    machine, or 3 beams, can fall behind and lag more the longer the user speaks. Catching up
    needs the library to let a stream skip passes (one pass over all the audio queued, not one
    per second of it); measure on a slow machine first.
- MatterCAD Rust: the desktop Whistle provider lands in mattercad-rust (its roadmap, `ai.md`);
  the browser provider needs a Web Worker host like the demo's.

## 9. Release on crates.io

A Rust developer can `cargo add whistle-rust`. The name is free on crates.io (checked
2026-10-24); publishing runs from GitHub Actions with the `CARGO_REGISTRY_TOKEN` repository
secret (a 90-day token the maintainer rotates; the first publish needs one with the publish-new
scope), so the token never touches a development machine.

- The packaged crate must build on its own: today `lib.rs` includes `../../../README.md`, which
  is not in the package, so the published crate would fail to compile. Make the crate docs
  build from the package, and check that in CI (`cargo package` plus a build of the unpacked
  crate).
- Package only what callers need (`src/`, README, LICENSE, the third-party notices Apache-2.0
  requires); the repo-only tests, fixtures and oracle stay out. Document, and expose as
  constants, the pinned `whistle.cact` download URL and its SHA-256, since the weights are not
  in the crate.
- CI (`ci.yml`): fmt, clippy and tests on macOS, Linux and Windows, fetching `whistle.cact`
  (pinned, checksummed, cached); this also runs the AVX2 kernels on real x86_64 hardware.
- `publish.yml` (`workflow_dispatch`, as tess2-rust does): tests, then `cargo publish`.

