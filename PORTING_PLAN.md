# whistle-rust porting plan — open work

Open work only, in landing order. Each item says what a caller can do once it lands and what it
needs. Delete an item when it lands; rewrite it to what is left when part of it lands. How the
model, the archive and the oracle work: `docs/design/model.md`, `docs/design/oracle.md`. The
verification bar for every stage: CLAUDE.md "Tolerance policy".

## 6. Performance

Transcription is fast enough for push-to-talk on every target, and the weights stay packed
(`docs/design/performance.md`: ~41 MB resident after binding, a 3 s clip in ~75-85 ms on an M5
at one beam, ~2.2 ms per decoder token); what is left.

**Priority: single-threaded wasm32 (simd128).** The browser build has no threads, and phones and
slower computers on the web are where transcription is slowest, so performance work is chosen
and measured for one thread on wasm first; native threads are a bonus.

- A wasm benchmark: the `bench` example built for wasm32-wasip1 with simd128, run under Node
  (V8, the engine Chrome uses) and wasmtime, single-threaded, with a profile (`node --cpu-prof`)
  of where the time goes; then the browser itself once the Web Worker host exists (item 8).
- Optimise the single-thread hot paths that profile shows, in wasm terms: simd128 is 4-wide with
  no per-lane variable shift and no FMA we may use; `exp`/`ln` come from software libm on wasm;
  the encoder decodes ~19M packed weights per call on one thread.
- Run the AVX2 paths' bit-identity tests on x86_64 hardware (dense, few-row and packed kernels
  are checked by construction and asm only); simd128 ran under wasmtime.
- The 3 s encoder alone is 5-12% slower than the dense build (it decodes ~19M packed weights per
  call); cut that if the encoder dominates a real workload.
- Close the gap to the engine (`scripts/bench.sh`, `docs/design/performance.md` "Against the
  engine"): about 2x slower at one beam and 3.5-7x at the default three. Batch the beams'
  decoder steps into one multi-row projection per matrix (so the panel path and threads help),
  then the encoder (~40 ms on a 3 s clip against the engine's 11-16 ms to first token).

## 7. Designed decode features

The behaviours only the closed engine has, designed here and held to the engine transcripts.
Beam search (default 3 beams), keyword biasing and word timestamps are in
(`docs/design/decoding.md`, `docs/design/timestamps.md`); what is left:

- Windowing past 30 s, and streaming that commits words two passes agree on.
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

