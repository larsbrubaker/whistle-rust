//! Streaming resampling: audio that arrives in pieces (a microphone at 44.1 or 48 kHz, in pushes
//! of any size) brought to another rate exactly as [`super::resample`] converts the whole clip.
//!
//! Resampling each piece on its own is wrong: [`super::resample`] counts the samples outside its
//! slice as zero and starts its phase at the slice, so every boundary is attenuated (a click per
//! push, enough to change a word) and every piece's length is rounded. [`StreamResampler`] keeps
//! the input it still needs and the index of its next output across pushes, and computes every
//! output with the one-shot resampler's filter (`resample.rs`'s `Filter`, the same weights
//! summed in the same order), so the pieces' outputs joined are the whole clip's output, bit
//! for bit, on every target. Each output waits for the input half a kernel past it (3 ms at
//! 48 kHz); [`StreamResampler::finish`] gives the rest, the input after the end counting as zero.
//!
//! Ported from agg-gui-whistle's `resample.rs` (github.com/larsbrubaker/agg-gui, by the same
//! author), which kept its own copy of this crate's filter.

use super::resample::Filter;
use super::{AudioError, SUPPORTED_RATES};

/// One rate conversion, fed in pushes (see the module docs).
pub struct StreamResampler {
    /// `None` for equal rates: samples pass through.
    filter: Option<Filter>,
    /// The input still needed; `held[0]` is input sample `base`.
    held: Vec<f32>,
    base: u64,
    /// Input samples pushed in all.
    received: u64,
    /// The next output sample's index.
    next: u64,
}

impl StreamResampler {
    /// A conversion from `from_rate` to `to_rate` Hz. Equal rates pass samples through; a rate
    /// outside [`SUPPORTED_RATES`] is [`AudioError::UnsupportedSampleRate`].
    pub fn new(from_rate: u32, to_rate: u32) -> Result<StreamResampler, AudioError> {
        for rate in [from_rate, to_rate] {
            if !SUPPORTED_RATES.contains(&rate) {
                return Err(AudioError::UnsupportedSampleRate(rate));
            }
        }
        Ok(StreamResampler {
            filter: (from_rate != to_rate).then(|| Filter::new(from_rate, to_rate)),
            held: Vec::new(),
            base: 0,
            received: 0,
            next: 0,
        })
    }

    /// Take `samples` and return every output whose input is all in.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        let Some(filter) = self.filter.as_mut() else {
            return samples.to_vec();
        };
        self.held.extend_from_slice(samples);
        self.received += samples.len() as u64;
        let mut out = Vec::new();
        // Output `next` needs input up to its centre plus the reach.
        while filter.centre(self.next) + filter.reach() < self.received as i64 {
            out.push(output(filter, &self.held, self.base, self.next));
            self.next += 1;
        }
        // Drop the input no output from `next` on reads.
        let needed = (filter.centre(self.next) - filter.reach()).max(0) as u64;
        if needed > self.base {
            let drop = ((needed - self.base) as usize).min(self.held.len());
            self.held.drain(..drop);
            self.base += drop as u64;
        }
        out
    }

    /// The end of the audio: the outputs still owed, `round(input * to / from)` in all, as
    /// [`super::resample`] makes for the whole input. Afterwards the resampler is as new: the
    /// next push starts another stretch of audio.
    pub fn finish(&mut self) -> Vec<f32> {
        let Some(filter) = self.filter.as_mut() else {
            return Vec::new();
        };
        let total = filter.output_len(self.received);
        let mut out = Vec::new();
        while self.next < total {
            out.push(output(filter, &self.held, self.base, self.next));
            self.next += 1;
        }
        self.held.clear();
        self.base = 0;
        self.received = 0;
        self.next = 0;
        out
    }
}

/// Output `i` from the held input (`held[0]` is input sample `base`; anything else is zero:
/// before the audio, after its end, or already dropped because no output needs it).
fn output(filter: &mut Filter, held: &[f32], base: u64, i: u64) -> f32 {
    // Input indices fit in i64: they count samples held in memory.
    let base = base as i64;
    filter.output(i, |at| {
        if at < base {
            return None;
        }
        held.get((at - base) as usize).copied()
    })
}
