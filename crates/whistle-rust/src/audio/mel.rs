//! Whistle's input features: the mel filter bank (`mel_matrix`, following upstream
//! `whistle.mel_matrix`) and the masked, per-bin normalized log-mel spectrogram (`features`,
//! following `whistle.features` for a single clip). Uses the radix-2 FFT in `fft`.

use super::fft::Fft;

/// Mel bins per frame (`WhistleConfig.mels`).
pub const MELS: usize = 80;
/// Samples between frames (`WhistleConfig.hop`): 10 ms at 16 kHz.
pub const HOP: usize = 160;
/// FFT bins per frame, `fft / 2 + 1`: the rows of `mel_matrix`.
pub const FFT_BINS: usize = FFT / 2 + 1;

const RATE: f64 = 16000.0;
const FFT: usize = 512;
const WIN: usize = 400;

/// The normalized log-mel features of one clip, row-major `[frames, MELS]`. Frames at and beyond
/// `valid_frames` are zero, as upstream masks them.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
}

fn to_mel(f: f64) -> f64 {
    if f < 1000.0 {
        f * 3.0 / 200.0
    } else {
        15.0 + crate::math::ln(f.max(1e-9) / 1000.0) * 27.0 / crate::math::ln(6.4)
    }
}

fn to_hz(m: f64) -> f64 {
    if m < 15.0 {
        m * 200.0 / 3.0
    } else {
        1000.0 * crate::math::exp((m - 15.0) * crate::math::ln(6.4) / 27.0)
    }
}

/// `np.linspace(0, stop, num)`: `i * step` with the last point exactly `stop`.
fn linspace(stop: f64, num: usize) -> Vec<f64> {
    let step = stop / (num - 1) as f64;
    (0..num)
        .map(|i| if i == num - 1 { stop } else { i as f64 * step })
        .collect()
}

/// The triangular mel filter bank, row-major `[FFT_BINS, MELS]` (257 × 80), area-normalized
/// (each filter scaled by 2 / its width in Hz). Computed in f64 and stored as f32, as upstream.
pub fn mel_matrix() -> Vec<f32> {
    let points: Vec<f64> = linspace(to_mel(RATE / 2.0), MELS + 2)
        .into_iter()
        .map(to_hz)
        .collect();
    let freqs = linspace(RATE / 2.0, FFT_BINS);
    let mut bank = Vec::with_capacity(FFT_BINS * MELS);
    for &f in &freqs {
        for m in 0..MELS {
            let (lo, mid, hi) = (points[m], points[m + 1], points[m + 2]);
            let rise = (f - lo) / (mid - lo);
            let fall = (hi - f) / (hi - mid);
            let weight = 0.0f64.max(rise.min(fall));
            bank.push((weight * 2.0 / (hi - lo)) as f32);
        }
    }
    bank
}

/// `np.hanning(WIN)` (symmetric), rounded to f32 as upstream stores it.
fn window() -> Vec<f64> {
    (0..WIN)
        .map(|k| {
            let n = 2.0 * k as f64 + 1.0 - WIN as f64;
            f64::from(
                (0.5 + 0.5 * crate::math::cos(std::f64::consts::PI * n / (WIN - 1) as f64)) as f32,
            )
        })
        .collect()
}

/// Whistle's features for one clip of 16 kHz mono samples: `1 + N / HOP` frames, of which
/// `N / HOP` are valid. The STFT zero-pads `WIN / 2` samples on both sides and frames every
/// `HOP` samples with the symmetric Hann window; each frame's power spectrum (512-point FFT, the
/// window's sum normalization undone) goes through `mel_matrix`, then `ln(x + 2^-24)`; each mel
/// bin is normalized by its mean and sample standard deviation over the valid frames,
/// `(x - mean) / (std + 1e-5)`, and the invalid frames are zeroed.
///
/// Upstream rejects clips shorter than one window (400 samples); here they are framed the same
/// way (the padding makes every frame well defined), so 0..159 samples give one all-zero frame.
///
/// The arithmetic is f64 from the f32 samples, window and filter bank up to the stored f32
/// features. Measured on the speech clips: an independent numpy f64 evaluation of upstream's
/// formula lands 1.654e-6..3.704e-6 (relative L2) from the oracle's features, and this port
/// lands at the same distances to about 4 digits. That gap is upstream's own f32 rounding, so
/// carrying f32 here would only add a second, independent rounding error on top of it.
pub fn features(samples: &[f32]) -> Features {
    let frames = 1 + samples.len() / HOP;
    let valid = samples.len() / HOP;
    let mel = mel_matrix();
    let window = window();
    let fft = Fft::new(FFT);
    let mut logmel = vec![0.0f64; valid * MELS];
    let (mut re, mut im) = (vec![0.0f64; FFT], vec![0.0f64; FFT]);
    let mut power = [0.0f64; FFT_BINS];
    let pad = WIN / 2;
    for t in 0..valid {
        re.fill(0.0);
        im.fill(0.0);
        for (k, &w) in window.iter().enumerate() {
            // Position k of frame t is padded-signal index t * HOP + k, sample index - pad.
            let at = (t * HOP + k).checked_sub(pad);
            if let Some(&s) = at.and_then(|i| samples.get(i)) {
                re[k] = f64::from(s) * w;
            }
        }
        fft.forward(&mut re, &mut im);
        for (b, p) in power.iter_mut().enumerate() {
            *p = re[b] * re[b] + im[b] * im[b];
        }
        let row = &mut logmel[t * MELS..(t + 1) * MELS];
        for (m, out) in row.iter_mut().enumerate() {
            let energy: f64 = (0..FFT_BINS)
                .map(|b| power[b] * f64::from(mel[b * MELS + m]))
                .sum();
            *out = crate::math::ln(energy + 2f64.powi(-24));
        }
    }
    let n = valid.max(1) as f64;
    let mut data = vec![0.0f32; frames * MELS];
    for m in 0..MELS {
        // Mean and variance are taken of the values relative to the first valid frame's: the
        // differences are small, so their sums lose less, and a constant bin (silence) gives
        // exactly zero deviations and therefore exactly zero features.
        let origin = if valid > 0 { logmel[m] } else { 0.0 };
        let shifted = |t: usize| logmel[t * MELS + m] - origin;
        let mean = (0..valid).map(shifted).sum::<f64>() / n;
        let var = (0..valid).map(|t| (shifted(t) - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let scale = var.sqrt() + 1e-5;
        for t in 0..valid {
            data[t * MELS + m] = ((shifted(t) - mean) / scale) as f32;
        }
    }
    Features {
        data,
        frames,
        valid_frames: valid,
    }
}
