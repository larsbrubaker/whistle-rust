# whistle-rust porting plan — open work

Open work only, in landing order. Each item says what a caller can do once it lands and what it
needs. Delete an item when it lands; rewrite it to what is left when part of it lands. How the
model, the archive and the oracle work: `docs/design/model.md`, `docs/design/oracle.md`. The
verification bar for every stage: CLAUDE.md "Tolerance policy".

## 4. Decoder

A caller can score a token sequence against encoded audio. Token embedding, Engram keys, values
and the gated add, and the decoder memory are in (`model::engram`, `model::decoder_input`);
what is left:

- Causal self-attention with the 3-tap QKV convs, cross-attention to the memory (8 KV heads,
  v width 512, keys-only mask over valid frames), the decoder block, the decoder mHC stack with
  the Engram add at its sites, the final norm, and the tied A8 output head; an incremental KV
  cache whose per-step logits equal the full recompute.
- Bound the header's n-gram orders (e.g. ≤ `max_seq_len`), so a hostile archive cannot make the
  hash loop arbitrarily slow.
- Tests: the block's sub-modules from en_samantha's committed decoder layer 0 and 3 internals;
  each decoder layer from the previous `lanes`; `decoder/out` and `decoder/logits`
  (en_samantha, de_anna), bound set by measurement.

## 5. Greedy transcription (first milestone)

A caller can call `transcribe(samples)` and get text and language.

- Greedy decode with language detection (step 0 restricted to tag tokens) or a forced language;
  `max_tokens` 320; cancellation checked between decoder steps.
- Tests: token streams identical to every clip's manifest; decoded text equals the manifest.

## 6. Performance

Transcription is fast enough for push-to-talk on every target.

- Keep CQ weights packed in memory (17 MB) and rotate activations by a fast Walsh–Hadamard
  transform per group instead of dequantizing; SIMD paths (NEON, AVX2, wasm simd128) behind one
  portable fallback with identical results; native worker threads.
- Bench against the engine CLI on the same clips; record the numbers in `docs/design/`.

## 7. Designed decode features

The behaviours only the closed engine has, designed here and held to the engine transcripts.

- Silence gate wiring: transcription runs `vad::speech_segments` first, returns an empty
  transcript when there is no speech (the reference decode says "Thank you." on `silence`; the
  engine returns ""), and decodes only the speech span. The detector's known false positives
  (intermittent beeps, typing, `docs/design/vad.md`) need a check on real microphone recordings
  before always-on use.
- Beam search (5 beams), keyword biasing, word timestamps from cross-attention, windowing past
  30 s, streaming that commits words two passes agree on.
- Tests: word error rate within noise of `test-data/engine/` per clip; timestamps against the
  engine's within a frame or two; more clips (long audio, noise) via `oracle/clips.json`.

## 8. Integration

- wasm32: the library builds and runs in a Web Worker; the host downloads and caches
  `whistle.cact` from a pinned, self-hosted URL.
- MatterCAD Rust: a local Whistle transcription provider beside OpenAI
  (`mattercad-ai` `AiTranscriptionProvider`; MatterCAD's recorders already record 16 kHz mono
  PCM16). The step lands in mattercad-rust, following its CLAUDE.md.
- An agg-gui demo (microphone → live transcript), native and web.
