//! Sample-rate conversion for the front end: a band-limited windowed-sinc (Kaiser) resampler, so a
//! WAV at 44.1 kHz, 48 kHz or any other rate can be brought to the 16 kHz `features` expects.
//! Upstream resamples with an external library; this is designed here from first principles.

use super::SUPPORTED_RATES;

/// Zero crossings of the low-pass kernel on each side, in samples of the lower of the two rates.
const HALF_TAPS: f64 = 48.0;
/// The kernel's cutoff as a fraction of the lower rate's Nyquist: the transition band ends just
/// below it, so content above the new Nyquist is attenuated by the Kaiser window's stop band.
const CUTOFF: f64 = 0.94;
/// Kaiser β for roughly 85 dB of stop-band attenuation.
const BETA: f64 = 8.6;
/// Above this many distinct output phases the kernel is evaluated per output sample instead of
/// tabulated (e.g. rates with no large common divisor).
const MAX_PHASES: u64 = 1024;

/// The zeroth-order modified Bessel function of the first kind, by its power series.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0f64, 1.0f64, x * x / 4.0);
    for k in 1..64 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

struct Kernel {
    /// Cutoff in cycles per input sample.
    fc: f64,
    /// Half-width of the kernel in input samples.
    half_width: f64,
    i0_beta: f64,
}

impl Kernel {
    /// Weights for input samples `first..first + len` around input position `x`, normalized to
    /// unit sum so a constant signal passes unchanged.
    fn weights(&self, x: f64, first: i64, len: usize, out: &mut Vec<f64>) {
        out.clear();
        for j in 0..len {
            let d = x - (first + j as i64) as f64;
            let u = d / self.half_width;
            let w = if u.abs() >= 1.0 {
                0.0
            } else {
                let arg = 2.0 * self.fc * d;
                let sinc = if arg == 0.0 {
                    1.0
                } else {
                    (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
                };
                sinc * bessel_i0(BETA * (1.0 - u * u).sqrt()) / self.i0_beta
            };
            out.push(w);
        }
        let sum: f64 = out.iter().sum();
        if sum != 0.0 {
            out.iter_mut().for_each(|w| *w /= sum);
        }
    }
}

/// Resamples mono `samples` from `from_rate` to `to_rate` Hz with a band-limited windowed-sinc
/// filter (cutoff just below the lower rate's Nyquist). The output has
/// `round(len * to_rate / from_rate)` samples; equal rates return the input unchanged. A rate
/// outside `SUPPORTED_RATES` (1 kHz to 768 kHz) returns no samples, which bounds the output to
/// 768 samples per input sample whatever a caller passes in.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    if !SUPPORTED_RATES.contains(&from_rate) || !SUPPORTED_RATES.contains(&to_rate) {
        return Vec::new();
    }
    if samples.is_empty() {
        return Vec::new();
    }
    let (from, to) = (u64::from(from_rate), u64::from(to_rate));
    // u64 holds len * rate for any slice that fits in memory and any u32 rate. The quotient can
    // exceed usize on 32-bit targets (wasm32) only for an output no 4 GB address space could hold
    // (over 768x a slice already in memory); that is answered with no samples rather than a
    // truncated length.
    let Ok(out_len) = usize::try_from((samples.len() as u64 * to + from / 2) / from) else {
        return Vec::new();
    };
    let lower = from.min(to) as f64;
    let kernel = Kernel {
        fc: CUTOFF * lower / 2.0 / from as f64,
        half_width: HALF_TAPS * from as f64 / lower,
        i0_beta: bessel_i0(BETA),
    };
    let reach = kernel.half_width.ceil() as i64;
    let len = (2 * reach + 1) as usize;
    let phases = to / gcd(from, to);
    // Output i sits at input position i * from / to; its fractional part repeats every `phases`.
    let table: Option<Vec<Vec<f64>>> = (phases <= MAX_PHASES).then(|| {
        (0..phases)
            .map(|p| {
                let x = (p * from % to) as f64 / to as f64;
                let mut w = Vec::new();
                kernel.weights(x, -reach, len, &mut w);
                w
            })
            .collect()
    });
    let mut scratch = Vec::new();
    (0..out_len as u64)
        .map(|i| {
            let whole = (i * from / to) as i64;
            let weights = match &table {
                Some(table) => &table[(i % phases) as usize],
                None => {
                    let x = (i * from % to) as f64 / to as f64;
                    kernel.weights(x, -reach, len, &mut scratch);
                    &scratch
                }
            };
            let first = whole - reach;
            let mut acc = 0.0f64;
            for (j, &w) in weights.iter().enumerate() {
                let at = first + j as i64;
                if at >= 0 {
                    if let Some(&s) = samples.get(at as usize) {
                        acc += f64::from(s) * w;
                    }
                }
            }
            acc as f32
        })
        .collect()
}
