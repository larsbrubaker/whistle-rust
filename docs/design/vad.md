# Voice-activity detection (`whistle::vad`)

Whistle's decoder, like most attention decoders, writes text on non-speech: the reference greedy
decode says "Thank you." on a second of digital silence, while the engine's documented behaviour
is an empty transcript for silence and steady noise. `whistle::vad` is the gate that runs before
decoding. It was designed here from that documented behaviour; there is no reference for it.

`speech_segments(samples)` returns the sorted, disjoint sample ranges of 16 kHz mono audio that
hold speech; `has_speech` is "any segment"; `speech_segments_with(&VadConfig, ..)` takes explicit
thresholds. It allocates only per-frame buffers, sized by the input. Any `VadConfig` is accepted
and never panics: a NaN field acts as its default, negative durations as 0, durations longer
than the input as the whole input, and ±∞ dB thresholds as "never blocks" / "never passes".

Results are bit-identical on every platform: the window, the band powers and the dB
conversions use f64 `cos`, `pow` and `log10` from the library's own deterministic math
(`src/math`, ported from musl via the Rust `libm` crate), not the platform libm, so a case
sitting right at a threshold comes out the same on macOS, Linux, Windows and wasm32. The
margins below keep the tests away from the thresholds all the same, so a deliberate change to
the features does not flip a clip unnoticed.

## Pipeline

1. **Frames** (`vad/features.rs`): 25 ms periodic-Hann frames at a 10 ms hop, 512-point FFT
   (the front end's `audio::fft`). Bins below `low_cut_hz` (80 Hz, rounded per 31.25 Hz bin,
   so the first kept bin is 93.75 Hz) are dropped: DC and rumble. The cut does not remove mains
   hum, since the window's main lobe leaves a 60 Hz fundamental only about 4 dB down in the
   kept bins; hum is rejected by the steadiness gates (steps 2 and 4). The rest is summed into six roughly octave-wide bands with tops at
   300, 600, 1200, 2400, 4000 and 8000 Hz, each in dB relative to a full-scale square wave.
2. **Noise floors** (`decide::band_floor`): per band, the minimum over ±`floor_window_s` (1 s) of
   the band energy power-averaged over `floor_smoothing_s` (50 ms), clamped at
   `absolute_floor_db − 20`. The minimum is a monotonic-deque sliding minimum, linear in the
   input whatever the window. A pause between words is always inside a 2 s window, so the floor
   is the background; steady noise is its own floor.
3. **Rise**: a frame's largest band-over-floor rise, over bands within 30 dB of its loudest band
   (fainter bands hold only window leakage, which wobbles without any sound behind it). A frame
   whose total power is under `absolute_floor_db` (−75) has no rise. Per-band rises are what let
   speech under broadband noise stand out: a formant concentrates power in one band, where the
   noise has only its share.
4. **Modulation** (`decide::modulation`): within ±`modulation_window_s / 2` (0.5 s window), the
   interquartile range of the 50 ms-smoothed total energy of the frames whose rise exceeds
   `margin_off_db`, or 0 when fewer than `min_modulation_span_s` (0.1 s) of such frames exist.
   Speech rises and falls at syllable rate; tones, hums, fans and hiss switching on do not. The
   interquartile range ignores the frames straddling an onset, and measuring the energy rather
   than the rise keeps the floor's step (when old silence leaves its window) from posing as
   modulation.
5. **Decision**: hysteresis on the rise (start above `margin_on_db` = 15, continue above
   `margin_off_db` = 3), with modulation ≥ `min_modulation_db` (2.5) on every speech frame.
6. **Runs**: gaps up to `merge_gap_s` (0.3 s) merged, runs under `min_speech_s` (0.1 s)
   dropped, then padded by `hangover_before_s` (0.1 s) and `hangover_after_s` (0.2 s), clamped
   to the input and merged again across `merge_gap_s`, so any two segments a caller sees are
   more than 0.3 s apart.

## Tests and measured margins

`crates/whistle-rust/tests/vad/`: no speech on the empty slice, `silence.wav`, white and pink
noise at −60/−40/−20 dBFS, 50 and 60 Hz hum with harmonics, a 1 kHz tone, a click, brown fan
noise, and a tone or hiss switching on after silence. Speech on every clip clean, at 10 dB SNR
under white noise and 30 dB quieter: segments cover ≥ 90% of the engine's word span
(`test-data/engine/`, used as data) and reach ≤ 0.5 s outside it. Trimming is pinned twice: a
second of digital silence each side is at most a third covered, and 2 s of white noise each
side (the noise also under the speech, at 10 and 20 dB SNR) is at most a quarter covered, with
segments more than `merge_gap_s` apart. Plus determinism, ordering, bounds, and every
`VadConfig` field at ±∞, NaN, ±1e30, −1 and 0 (`config.rs`).

`threshold_flip_points` (ignored; `--ignored --nocapture`) sweeps one threshold at a time from
the defaults and prints every value at which each case flips. With the defaults as shipped
(the step of each sweep in brackets):

| threshold | non-speech is rejected | speech is found | default |
| --- | --- | --- | --- |
| `margin_on_db` [0.5] | from 10.5 with modulation off (brown fan); at any value with it on | up to 23.5 (es_monica, 10 dB SNR) | 15 |
| `margin_off_db` [0.375] | at any value | up to 5.25 (nl_xander, 10 dB SNR) | 3 |
| `min_modulation_db` [0.25] | from 2.0 with the energy margins off (brown fan); at any value with them on | up to 3.0 (en_samantha, 10 dB SNR) | 2.5 |
| `absolute_floor_db` [1.5] | at any value | up to −67 (nl_xander, 30 dB quieter) | −75 |
| `min_modulation_span_s` [0.006] | at any value | up to 0.38 (it_alice, 30 dB quieter) | 0.1 |
| `merge_gap_s` [0.0125] | at any value | at any value in 0–1 s | 0.3 |

Non-speech is rejected by two independent gates, the rise (steady sound is its own floor) and the
modulation (steady sound is flat); a single threshold relaxed alone never lets it through, so its
row reports the flip point with the other gate switched off. `defaults_hold_their_measured_margins`
and `min_modulation_db_sits_between_fan_noise_and_noisy_speech` pin parts of this table.

**Hard to separate.**
- Modulation alone barely separates fan noise (rejected from 2.0 dB) from speech at 10 dB SNR
  (found up to 3.0 dB); the default sits midway, and the rise gate carries most of the work.
- Measured false positives, sounds that are loud, not steady and rise and fall like syllables
  (`known_limits.rs` pins them so a fix or a regression shows): an intermittent 1 kHz beep,
  150 ms on and 150 ms off, with or without a −50 dB noise bed (fully covered); a typing-like
  click train at about 8 per second (89.5% covered); white noise with 4 Hz amplitude modulation
  (fully covered at full modulation depth; shallower modulation is partly detected).
- Music and babble are not tested and will often read as speech, which is why a VAD or wake
  word is still recommended in front of an always-on microphone.
- The test clips are clean TTS; real microphones and rooms are not covered yet.
- The decode is chaotically sensitive to the lead-in the trim leaves: on a clip whose speech
  starts at its first sample the trim is the whole clip, while the same clip after silence keeps
  about 0.12 s of it (the pad plus frame rounding), and that can change the transcript
  (pl_zosia flips with 160, 320 or 1280 leading zeros; `crates/whistle-rust/tests/timestamps/options.rs`). Different,
  not worse; word times stay exact relative to the trimmed audio.
