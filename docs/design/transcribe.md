# Transcription (`whistle::Whistle`)

`Whistle::from_cact(bytes)` parses the archive, binds the weights (`model::Weights`) and the
tokenizer, and looks up the seven language tags (`en de fr es it nl pl`, ids 8192–8198). It owns
everything, so one `Whistle` can be kept for the life of the program and shared across threads.
`transcribe(samples, &Options)` takes 16 kHz mono `f32`; `transcribe_wav(bytes, &Options)` reads
the WAV (`audio::read_wav`, downmixed to mono), resamples to 16 kHz when the file is at another
rate, then transcribes. A `Transcript` is the text, the language code, and the text token ids
(without BOS, the tag and EOS).

## The call, in order

1. **Options.** A forced language outside the seven codes is `Error::Input`, listing them. A
   WAV file `read_wav` cannot decode is `Error::Audio` (the reader's `AudioError`).
2. **Finite samples.** Any NaN or ±∞ sample is `Error::Input` naming its index, before anything
   else runs: one NaN would otherwise poison the features and the decoder would write 318
   tokens of nonsense.
3. **Early length limits**, on the length at 16 kHz (`len · 16000 / rate`) before resampling or
   detection, so an oversized input costs nothing: with `Off` or `Gate`, over 480,000 samples
   (30 s) is `Error::Input` naming the limit; with `GateAndTrim`, which may trim a long
   recording down to a short utterance, over `MAX_INPUT_SAMPLES` (10 minutes) is.
4. **Resampling** to 16 kHz when the WAV is at another rate; the original buffer is dropped.
5. **Voice-activity gate** (`docs/design/vad.md`), per `VadMode`:
   - `Off`: decode the samples as given.
   - `Gate`: no speech segment gives an empty transcript; otherwise decode the whole clip.
   - `GateAndTrim` (the default): no speech gives an empty transcript; otherwise decode only
     `first.start .. last.end`, the span from the first segment's start to the last segment's
     end (gaps between segments are kept, so the decoder sees the utterance whole).
6. **Limits after trimming.** More than 480,000 samples (30 s) is `Error::Input`: windowing
   past 30 s is not ported yet. Empty audio, or audio shorter than one 160-sample hop (no valid
   feature frame, so the decoder would attend to a fully masked memory), is an empty transcript,
   not an error.
7. The front end (`audio::features`), the encoder (`Encoder::forward_with`), and the decoder
   memory (`Weights::decoder_memory`), each once.
8. **Greedy decode** over the KV cache (below), following `whistle_finetune._score_quantised`:
   the sequence starts `[BOS]`; step `pos` feeds the token at `pos` and scores the next one.
   Step 0 picks among the language tags only (the reference sets every other score to −∞), or,
   with a forced language, feeds that tag without looking. Later steps pick among the text ids
   only (below 8192), so a tag is never emitted after step 0. Ties go to the lowest id, as
   `np.argmax`; a NaN score never beats a number (finite audio through finite weights gives
   finite scores, so only a corrupt archive could produce one). The decode stops when it picks
   EOS (1) or after `max_seq_len − 1` steps, so the sequence never exceeds `max_seq_len` (320)
   tokens. `from_cact` checks that the seven tags are exactly the last ids of the vocabulary,
   as the reference's `arange(vocab) >= text` mask assumes.
9. **Text.** The ids after the tag decode through the tokenizer. No text ids gives an empty
   transcript (language `None`); otherwise the language is the detected or forced tag's code.
   Forcing a language feeds its tag; it does not translate (English audio forced to `fr` still
   comes out in English, labelled `fr`).

`Cancel` is an `Arc<AtomicBool>` handle; clones share the flag, so a UI thread can keep one and
call `cancel()` while the decode runs elsewhere. A fired handle makes the call return
`Error::Cancelled` at the next check: after the input checks, after resampling, after the
detector, after the features, before each encoder layer (`Encoder::forward_with` takes the check
as a closure, so the encoder does not depend on this module), before each decoder step, and
after the decode. The worst-case latency is the longest gap between two checks: on 30 s of
audio (release build, Apple Silicon, measured 2026-10-24 after the kernel work in
`performance.md`, on a loaded machine) the features take about 50 ms and the detector 13 ms,
the stem before the first layer check 28–46 ms, and the encoder layers 40–55 ms each, so a
cancel returns within about 0.07 s; a decoder step is about 5 ms.

## The KV cache (`model::DecoderState`)

`DecoderState::new(weights, memory, max_positions)` computes every layer's cross-attention keys
and values over the memory once (`CrossAttention::memory_kv`) and one RoPE table over
`max_positions` (a position's angles do not depend on the table's length). `step(token)` runs
the one new position through the stack:

- **Self-attention** (`SelfAttention::forward_step`, per layer `KvCache`): the new row's raw q,
  k, v and gate projections join the last `min(pos, n_taps − 1)` raw rows (2 for Whistle) for
  the causal taps (`causal_taps(.., history, pos)`); q and k are normed and rotated at `pos`, q
  takes A8 and k, v the KV path; k and v are appended per KV head; the query attends to every
  cached key.
- **Cross-attention** on the new row against the fixed `CrossMemory`.
- **Engram**: each site's keys and pre-conv values for the new position come from the token
  prefix (`Engram::project_rows`: the n-gram hashes look back at most 3 tokens); the values are
  kept so the value conv (`Engram::value_conv`, taps 4 at dilation 3) can read back 9 positions.
- The **mHC step**, the final lane mean and norm, and the tied head on the new row only.

The cached step and the full pass (`Decoder::forward`) run the same functions:
`SelfAttention::attention_heads` (head split, norms, RoPE, quantization) is shared by the
encoder's bidirectional attention, the causal full pass and the step; `DecoderLayer::block_with`
is the block for both decoder paths, given their self-attention; `Engram::keys_values` is
`project_rows` then `value_conv` from position 0. Every operation is per row except the three
that read earlier positions: the QKV taps (from the cached raw rows), the Engram value conv
(from the cached projected values) and attention (over the cached keys and values). The first
two read the same rows in the same order as the full pass; in attention the keys the full pass masks take a large finite negative logit whose `exp` is exactly
0, which adds nothing to the softmax sum or the weighted values. So each step's scores equal the
full pass's row for that position **bit for bit** (`tests/transcribe/kv_cache.rs`, en_samantha
and de_anna, every position). A failed `step` with an out-of-vocabulary token or past the
decode's length leaves the state as it was; any other error leaves it part way through its
cache updates and unusable.

## Cost

Measured 2026-10-24 on en_samantha (3.09 s), release build, Apple Silicon, default threads:
the encoder 36 ms, the decoder setup (cross-attention memory for 8 layers) 4 ms, then about
4.5 ms per decoder token (on one thread whatever the setting); the whole default call (trimmed
to the speech span) 114 ms (661 ms before the kernel work; the breakdown, both thread counts
and what was done are in `performance.md`). Weights stay dequantized to f32 here, and each
self-attention step copies its cache into one contiguous block.
