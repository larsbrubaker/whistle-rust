# whistle-rust porting plan — open work

Open work only, in landing order. Each item says what a caller can do once it lands and what it
needs. Delete an item when it lands; rewrite it to what is left when part of it lands. How the
model, the archive and the oracle work: `docs/design/model.md`, `docs/design/oracle.md`. The
verification bar for every stage: CLAUDE.md "Tolerance policy".

## 6. Performance

Transcription is fast enough for push-to-talk on every target, and the weights stay packed
(`docs/design/performance.md`: ~41 MB resident after binding natively, ~59 MB on wasm simd128
with its byte codes; a 3 s clip in ~82 ms natively and ~87 ms on Node simd128 at one beam, one
thread); what is left.

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
  table on a quiet machine; then the browser itself (item 8's demo).
- The 3 s encoder alone is 5-12% slower than the dense build (it decodes ~19M packed weights per
  call); cut that if the encoder dominates a real workload.
- Close the gap to the engine (`scripts/bench.sh`, `docs/design/performance.md` "Against the
  engine"): about 2x slower at one beam and 3.5-7x at the default three before the beams'
  decoder steps were batched (`DecoderState::step_batch`, bit-identical; 20-30% faster at 3-5
  beams on one thread natively, wasm not yet measured cleanly). Next: what a batched step's cost
  is made of, then the encoder (~40 ms on a 3 s clip against the engine's 11-16 ms to first
  token).

## 7. Designed decode features

The behaviours only the closed engine has, designed here and held to the engine transcripts.
Beam search (default 3 beams), keyword biasing, word timestamps, windowing and streaming are in
(`docs/design/decoding.md`, `timestamps.md`, `windowing.md`, `streaming.md`); what is left:

- Streaming quality: a word committed and then read differently by a later pass stays as
  committed (no retraction), so live text can keep an early misreading ("matter" for
  "Matterhackers" parts, "make a the cube"); a merged re-read can swallow part of the next word.
  Per-update cost grows with the utterance (up to a 30 s decode per step); re-measure on wasm
  and consider bounding it.
- Keyword biasing at 5+ beams finds fewer keywords than at 1 or 3 (partial keyword paths crowd
  the beam): per-beam-count tuning or a cap on hypotheses inside a keyword.
- More clips (long audio, noise, real microphone recordings) via `oracle/clips.json`; the
  alignment heads were fitted on the 8 contract clips and hold on 3 held-out clips so far.

## 8. Integration

- wasm32: the library builds and runs in a Web Worker; the host downloads and caches
  `whistle.cact` from a pinned, self-hosted URL.
- MatterCAD Rust: a local Whistle transcription provider beside OpenAI
  (`mattercad-ai` `AiTranscriptionProvider`; MatterCAD's recorders already record 16 kHz mono
  PCM16). The step lands in mattercad-rust, following its CLAUDE.md.
- An agg-gui demo (microphone → live transcript), native and web.

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

