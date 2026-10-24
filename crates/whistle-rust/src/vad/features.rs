//! The detector's per-frame measurements: 25 ms Hann-windowed frames at a 10 ms hop, each reduced
//! to its log energy in six sub-bands above the rumble band. Uses
//! the front end's radix-2 FFT (`audio::fft`); `decide` turns these into speech segments.

use crate::audio::fft::Fft;
use crate::audio::{HOP, SAMPLE_RATE};

/// Samples per analysis frame: 25 ms, long enough to resolve a 100 Hz voice's pitch harmonics.
pub(crate) const WIN: usize = 400;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
/// Added to every frame's mean power before the logarithm so digital silence gives a finite
/// −120 dB instead of −∞.
const POWER_EPS: f64 = 1e-12;

/// Upper edges, in Hz, of the sub-bands whose energies are tracked separately; the first band
/// starts at the configured low cut. Roughly octave-wide: a vowel's first formant, its second,
/// and the fricative range each own a band, so speech under broadband noise stands far above
/// that band's share of the noise even when the frame's total power barely rises.
pub(crate) const BAND_TOPS_HZ: [f32; BANDS] = [300.0, 600.0, 1200.0, 2400.0, 4000.0, 8000.0];
pub(crate) const BANDS: usize = 6;

/// What one frame looks like to the detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Frame {
    /// Mean power of the frame in each band of `BAND_TOPS_HZ`, in dB relative to a full-scale
    /// square wave (a full-scale sine reads −3 dB in the band holding it).
    pub(crate) band_db: [f32; BANDS],
}

/// Number of whole frames in `len` samples; a tail shorter than a hop is not framed (the
/// segment hangover covers it).
pub(crate) fn frame_count(len: usize) -> usize {
    if len < WIN {
        0
    } else {
        (len - WIN) / HOP + 1
    }
}

fn bin_of(hz: f32) -> usize {
    let bin = (f64::from(hz) * FFT as f64 / f64::from(SAMPLE_RATE)).round();
    // Clamp before the cast so an out-of-range configured frequency cannot select a bin outside
    // the spectrum.
    bin.clamp(0.0, (BINS - 1) as f64) as usize
}

/// Measures every frame of `samples`. Bins below `low_cut_hz` (DC and rumble) are ignored; the
/// window's main lobe still lets a 50/60 Hz fundamental into the first kept bins, so mains hum
/// is left to the steadiness gates in `decide`.
pub(crate) fn frames(samples: &[f32], low_cut_hz: f32) -> Vec<Frame> {
    let count = frame_count(samples.len());
    let fft = Fft::new(FFT);
    // Periodic Hann window, scaled so a stationary signal's power reads the same as unwindowed.
    let window: Vec<f64> = (0..WIN)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / WIN as f64).cos())
        .collect();
    let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>();
    let low = bin_of(low_cut_hz);
    let tops = BAND_TOPS_HZ.map(bin_of);
    let (mut re, mut im) = (vec![0.0f64; FFT], vec![0.0f64; FFT]);
    let mut out = Vec::with_capacity(count);
    for t in 0..count {
        let frame = &samples[t * HOP..t * HOP + WIN];
        re.fill(0.0);
        im.fill(0.0);
        for ((r, &s), &w) in re.iter_mut().zip(frame).zip(&window) {
            *r = f64::from(s) * w;
        }
        fft.forward(&mut re, &mut im);
        let mut bands = [0.0f64; BANDS];
        let mut which = 0;
        for b in low.max(1)..BINS {
            // One-sided spectrum: interior bins stand for their negative-frequency twin too.
            let twin = if b == BINS - 1 { 1.0 } else { 2.0 };
            let p = twin * (re[b] * re[b] + im[b] * im[b]);
            while which + 1 < BANDS && b > tops[which] {
                which += 1;
            }
            bands[which] += p;
        }
        // Parseval: Σ|X|² over the full spectrum = FFT · Σ(x·w)², so dividing by this gives the
        // windowed mean power normalised back to the unwindowed signal's.
        let scale = FFT as f64 * window_power;
        out.push(Frame {
            band_db: bands.map(|p| (10.0 * (p / scale + POWER_EPS).log10()) as f32),
        });
    }
    out
}
