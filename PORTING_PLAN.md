# whistle-rust porting plan — open work

Open work only, in landing order. Each item says what a caller can do once it lands and what it
needs. Delete an item when it lands; rewrite it to what is left when part of it lands. How the
model, the archive and the oracle work: `docs/design/model.md`, `docs/design/oracle.md`. The
verification bar for every stage: CLAUDE.md "Tolerance policy".

## 5. Greedy transcription (first milestone)

A caller can call `transcribe(samples)` and get text and language. The decoder scores a full
token sequence (`model::Decoder`), and its restricted argmax reproduces every clip's reference
token at every position; what is left:

- An incremental KV cache (self-attention k/v after norm, RoPE and KV quant; the raw q/k/v of
  the last two positions for the taps, `causal_taps(.., min(pos, 2), pos)`; Engram rows per
  step) whose per-step logits equal the full recompute.
- Greedy decode with language detection (step 0 restricted to tag tokens) or a forced language;
  at most `max_seq_len` tokens; cancellation checked between decoder steps.
- The silence gate: `vad::speech_segments` first; no speech gives an empty transcript, and only
  the speech span is decoded. The detector's known false positives (intermittent beeps,
  typing, `docs/design/vad.md`) need a check on real microphone recordings before always-on
  use.
- Tests: token streams identical to every speech clip's manifest; decoded text equals the
  manifest; `silence` gives an empty transcript.

## 6. Performance

Transcription is fast enough for push-to-talk on every target.

- Keep CQ weights packed in memory (17 MB) and rotate activations by a fast Walsh–Hadamard
  transform per group instead of dequantizing; SIMD paths (NEON, AVX2, wasm simd128) behind one
  portable fallback with identical results; native worker threads.
- Bench against the engine CLI on the same clips; record the numbers in `docs/design/`.

## 7. Designed decode features

The behaviours only the closed engine has, designed here and held to the engine transcripts.

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
