# whistle-rust porting plan — open work

Open work only, in landing order. Each item says what a caller can do once it lands and what it
needs. Delete an item when it lands; rewrite it to what is left when part of it lands. How the
model, the archive and the oracle work: `docs/design/model.md`, `docs/design/oracle.md`. The
verification bar for every stage: CLAUDE.md "Tolerance policy".

## 1. Archive, dequantization and tokenizer

A caller can load `whistle.cact` from bytes, see its geometry, and decode token ids to text.

- `cact` module: header (tag `0x05E12A84`, geometry, `kv_bits`, codebooks), directory, blobs,
  each record bounds-checked; a malformed or truncated archive is an `Err`, never a panic.
- `quant` module: CQ unpack at 1, 2, 3, 4 bits and ternary (record bits 5), group norms,
  128-point Walsh–Hadamard; fp16 → f32.
- Tokenizer: decode (byte-fallback pieces, `▁` → space, control pieces dropped) and encode
  (user-defined markers first, then score-greedy BPE, byte fallback), matching `RefTokenizer`.
- Tests: every tensor equals the oracle's `read_export` value (new fixture: a per-tensor
  checksum and a few full tensors, added to `oracle/dump.py` in this step); every clip's
  reference token stream decodes to its manifest text.

## 2. Audio front end

A caller can turn 16 kHz mono samples (or a PCM16/float WAV) into Whistle's features.

- WAV reader (PCM 8/16/24/32, float, any channel count downmixed) and a 16 kHz resampler for
  other rates; mel features per `whistle.features` with an in-crate 512-point real FFT.
- Tests: `samples` and `features` fixtures on every clip (exact-stage bound).

## 3. Encoder

A caller can compute the speech embedding (one 512-vector per 80 ms frame).

- Stem, ZC-RMSNorm, A8 fake quant (round half to even), Hadamard MLP, encoder self-attention
  (GQA, RoPE, q/k norm, fused-head scaling, sigmoid gate, KV A8), conv module, mHC lane stack
  with Sinkhorn, final norm.
- Tests: `stem/out`; each encoder layer from the previous layer's `lanes` (en_samantha,
  de_anna); `encoder/out` on every clip end to end, bound set by measurement.

## 4. Decoder

A caller can score a token sequence against encoded audio.

- Embedding scale, Engram hashing (wrapping u32), tables, key/value projections and taps,
  causal self-attention with QKV conv taps, cross-attention, gated sinusoidal memory, tied head;
  an incremental KV cache whose per-step logits equal the full recompute.
- Tests: `decoder/embedded`, `decoder/memory`, `decoder/engram_k`/`_v`; each decoder layer
  from the previous `lanes`; `decoder/logits` (en_samantha, de_anna).

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

- Silence gate: silence and steady noise return an empty transcript (the reference decode
  says "Thank you." on `silence`; the engine returns "").
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
