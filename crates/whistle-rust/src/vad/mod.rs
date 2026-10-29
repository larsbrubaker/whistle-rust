//! Voice-activity detection: which parts of 16 kHz mono audio contain speech. Transcription runs
//! it first so silence and steady noise give an empty transcript instead of the decoder's
//! hallucinated "Thank you.", and so leading and trailing non-speech can be trimmed.
//!
//! `features` measures 25 ms frames at a 10 ms hop (energy in six sub-bands above the rumble
//! band); `decide` turns them into segments with an adaptive noise floor, a syllable-rate
//! modulation test, hysteresis, gap merging, a minimum duration and a hangover. The detector is
//! designed here from the engine's documented behaviour (an empty transcript for silence and
//! steady noise); there is no reference for it.
//!
//! Non-speech is rejected by two independent gates: the rise over each band's own noise floor
//! (steady sound is its own floor) and the syllable-rate modulation of the energy (steady sound
//! is flat). Relaxing one threshold alone never lets steady sound through; the margin each
//! default keeps from flipping a test case is measured by `threshold_flip_points` in
//! `tests/vad/margins.rs`, which carries the table.
//!
//! Results are bit-identical on every platform: the window, the band powers and the dB
//! conversions use the library's own deterministic `cos`, `pow` and `log10` (`crate::math`),
//! not the platform's libm, so a case sitting right at a threshold comes out the same on macOS,
//! Linux, Windows and wasm32. The detector allocates only per-frame buffers, sized by the input.
//!
//! Limits a caller should know:
//! - Sounds that are loud, unsteady and rise and fall like syllables read as speech: an
//!   intermittent beep, a typing-like click train, noise modulated at a few hertz
//!   (`tests/vad/known_limits.rs` pins how much of each is covered). Music and babble are not
//!   tested and will often read as speech, so an always-on microphone still wants a wake word
//!   or a push-to-talk in front of it.
//! - The tests run on clean synthesised speech with added noise; real microphones and rooms
//!   are not covered.
//! - The decode is sensitive to the lead-in the trim leaves. A clip whose speech starts at its
//!   first sample keeps all of it, while the same clip after silence keeps about 0.12 s of
//!   lead-in (the pad plus frame rounding), and that can change the transcript (pl_zosia flips
//!   with 160, 320 or 1280 leading zeros, `tests/timestamps/options.rs`): different, not
//!   worse, and word times stay exact relative to the trimmed audio.

mod decide;
mod features;

use std::ops::Range;

/// The detector's tunable thresholds. `Default` is the setting the tests and the measured
/// margins in `tests/vad/margins.rs` are for.
///
/// Every value is accepted and the detector never panics or allocates beyond the input's size:
/// a NaN field is replaced by its default; a duration (`*_s`) below zero counts as 0 and one
/// longer than the input counts as the whole input (so +∞ is allowed); a dB threshold may be
/// ±∞ (−∞ never blocks, +∞ never passes); `low_cut_hz` is clamped to 0–8000 Hz.
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    /// dB relative to full scale (a full-scale square wave is 0 dB, a full-scale sine −3 dB).
    /// A frame whose total power is below this is never speech, and no band's noise floor goes
    /// more than 20 dB below it. Default −75: digital silence and dither-level near-silence are
    /// never speech whatever the adaptive floor does (no non-speech test case depends on it),
    /// and a talker 30 dB quieter than the test clips is still found with the floor up to
    /// −67 dB, 8 dB of headroom.
    pub absolute_floor_db: f32,
    /// Seconds each side of a frame over which the noise floor is the quietest smoothed energy.
    /// Default 1.0: a 2 s window always contains a pause between words, yet follows a background
    /// that changes over a few seconds.
    pub floor_window_s: f32,
    /// Seconds of power-domain smoothing before the floor's minimum is taken. Default 0.05: it
    /// lifts the minimum of steady noise to within about a dB of its mean, so noise does not
    /// clear the margin on its own random peaks.
    pub floor_smoothing_s: f32,
    /// dB that a frame's best band must rise above its own noise floor to start a speech run.
    /// Default 15: with the modulation test off, steady noise (fan, pink) first passes below
    /// 10.5 dB, while every speech clip under white noise at 10 dB SNR is still found up to
    /// 23.5 dB.
    pub margin_on_db: f32,
    /// dB above the floor that keeps a run going, and that a frame must clear to count in the
    /// modulation measure. Default 3: well below `margin_on_db` so a run is not chopped by a soft
    /// consonant or a word's decay under noise (speech at 10 dB SNR loses coverage above 5.25).
    pub margin_off_db: f32,
    /// Seconds of audio, centred on each frame, over which energy modulation is measured.
    /// Default 0.5: two to four syllables at a normal speaking rate (2–8 Hz).
    pub modulation_window_s: f32,
    /// Seconds of frames above `margin_off_db` the modulation window must hold before it is
    /// measured at all. Default 0.1: a click or a single knock gives no modulation, so it is
    /// never speech.
    pub min_modulation_span_s: f32,
    /// Interquartile range, in dB, of the smoothed energy of the loud frames in the modulation
    /// window that speech must reach. Default 2.5: a tone, a hum, a fan or a hiss stays within
    /// about 2 dB of itself, including right after it switches on (fan noise reaches 2.0 with
    /// the energy margins off), while speech at 10 dB SNR is still found up to 3.0.
    pub min_modulation_db: f32,
    /// Hz below which FFT bins are ignored (rounded to the nearest 31.25 Hz bin; at the default
    /// the first kept bin is 93.75 Hz). Default 80: under every voice's pitch, above DC and
    /// rumble. It does not remove mains hum: the 25 ms Hann window's main lobe leaves a 60 Hz
    /// fundamental only about 4 dB down in the kept bins, so hum is rejected by the steadiness
    /// gates (its own noise floor and its flat envelope), not by this cut.
    pub low_cut_hz: f32,
    /// Seconds: a merged run shorter than this is dropped. Default 0.1, shorter than any word.
    pub min_speech_s: f32,
    /// Seconds: runs separated by at most this are merged into one segment. Default 0.3: the
    /// pause inside or between words of one phrase, short of a sentence break.
    pub merge_gap_s: f32,
    /// Seconds each segment is extended before its first speech frame. Default 0.1, so a soft
    /// onset (a breath, an unvoiced plosive) is not clipped.
    pub hangover_before_s: f32,
    /// Seconds each segment is extended after its last speech frame. Default 0.2, so decaying
    /// word endings and trailing fricatives are kept.
    pub hangover_after_s: f32,
}

impl Default for VadConfig {
    fn default() -> VadConfig {
        VadConfig {
            absolute_floor_db: -75.0,
            floor_window_s: 1.0,
            floor_smoothing_s: 0.05,
            margin_on_db: 15.0,
            margin_off_db: 3.0,
            modulation_window_s: 0.5,
            min_modulation_span_s: 0.1,
            min_modulation_db: 2.5,
            low_cut_hz: 80.0,
            min_speech_s: 0.1,
            merge_gap_s: 0.3,
            hangover_before_s: 0.1,
            hangover_after_s: 0.2,
        }
    }
}

impl VadConfig {
    /// This configuration with every NaN replaced by its default and `low_cut_hz` clamped to
    /// the spectrum; the remaining range handling (negative and oversized durations) is done
    /// where each value is used, against the input's size.
    fn sanitized(&self) -> VadConfig {
        let d = VadConfig::default();
        let or = |v: f32, default: f32| if v.is_nan() { default } else { v };
        VadConfig {
            absolute_floor_db: or(self.absolute_floor_db, d.absolute_floor_db),
            floor_window_s: or(self.floor_window_s, d.floor_window_s),
            floor_smoothing_s: or(self.floor_smoothing_s, d.floor_smoothing_s),
            margin_on_db: or(self.margin_on_db, d.margin_on_db),
            margin_off_db: or(self.margin_off_db, d.margin_off_db),
            modulation_window_s: or(self.modulation_window_s, d.modulation_window_s),
            min_modulation_span_s: or(self.min_modulation_span_s, d.min_modulation_span_s),
            min_modulation_db: or(self.min_modulation_db, d.min_modulation_db),
            low_cut_hz: or(self.low_cut_hz, d.low_cut_hz).clamp(0.0, 8000.0),
            min_speech_s: or(self.min_speech_s, d.min_speech_s),
            merge_gap_s: or(self.merge_gap_s, d.merge_gap_s),
            hangover_before_s: or(self.hangover_before_s, d.hangover_before_s),
            hangover_after_s: or(self.hangover_after_s, d.hangover_after_s),
        }
    }
}

/// The sample ranges of 16 kHz mono `samples` that contain speech, with the default settings:
/// sorted, non-overlapping, merged across short gaps and padded so word edges are kept. Empty
/// for silence, steady noise, tones and isolated clicks.
pub fn speech_segments(samples: &[f32]) -> Vec<Range<usize>> {
    speech_segments_with(&VadConfig::default(), samples)
}

/// `speech_segments` with explicit thresholds.
pub fn speech_segments_with(config: &VadConfig, samples: &[f32]) -> Vec<Range<usize>> {
    let config = &config.sanitized();
    let frames = features::frames(samples, config.low_cut_hz);
    decide::segments(&frames, samples.len(), config)
}

/// Whether 16 kHz mono `samples` contain any speech (with the default settings).
pub fn has_speech(samples: &[f32]) -> bool {
    !speech_segments(samples).is_empty()
}
