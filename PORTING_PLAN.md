# whistle-rust porting plan — open work

Open work only, in landing order. Each item says what a caller can do once it lands and what it
needs. Delete an item when it lands; rewrite it to what is left when part of it lands. How the
model, the archive and the oracle work: `docs/design/model.md`, `docs/design/oracle.md`. The
verification bar for every stage: CLAUDE.md "Tolerance policy".

## 6. Performance

Transcription is fast enough for push-to-talk on every target, and the weights stay packed
(`docs/design/performance.md`: ~41 MB resident after binding, a 3 s clip in ~75-85 ms on an M5
at one beam, ~2.2 ms per decoder token); what is left:

- Run the AVX2 paths' bit-identity tests on x86_64 hardware (dense, few-row and packed kernels
  are checked by construction and asm only); simd128 ran under wasmtime.
- The 3 s encoder alone is 5-12% slower than the dense build (it decodes ~19M packed weights per
  call); cut that if the encoder dominates a real workload.
- Bench against the engine CLI on the same clips and record the numbers.

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
