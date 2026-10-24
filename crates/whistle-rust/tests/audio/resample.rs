// The resampler (`whistle::audio::resample`) by its properties, since upstream's resampling is an
// external library with no oracle: tones below the new Nyquist keep frequency and amplitude, tones
// above it are removed, and the output length is the rate ratio rounded.

use whistle::audio::resample;

fn tone(freq: f64, rate: u32, seconds: f64) -> Vec<f32> {
    let n = (f64::from(rate) * seconds) as usize;
    (0..n)
        .map(|i| {
            (0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate)).sin()) as f32
        })
        .collect()
}

/// A least-squares fit of `amp * sin(2π freq t)` alone over the middle of `x` (away from the
/// edges): the amplitude and the RMS of what is left. With no cosine term the fit is phase
/// pinned, so an output delayed or advanced against the input leaves a large residual.
fn fit(x: &[f32], freq: f64, rate: u32) -> (f64, f64) {
    let start = x.len() / 4;
    let mid = &x[start..3 * x.len() / 4];
    let basis =
        |i: usize| (2.0 * std::f64::consts::PI * freq * (start + i) as f64 / f64::from(rate)).sin();
    let (mut vs, mut ss) = (0.0f64, 0.0f64);
    for (i, &v) in mid.iter().enumerate() {
        vs += f64::from(v) * basis(i);
        ss += basis(i) * basis(i);
    }
    let amp = vs / ss;
    let residual = mid
        .iter()
        .enumerate()
        .map(|(i, &v)| (f64::from(v) - amp * basis(i)).powi(2))
        .sum::<f64>()
        / mid.len() as f64;
    (amp, residual.sqrt())
}

fn rms(x: &[f32]) -> f64 {
    let mid = &x[x.len() / 4..3 * x.len() / 4];
    (mid.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / mid.len() as f64).sqrt()
}

#[test]
fn tones_below_the_new_nyquist_keep_frequency_and_amplitude() {
    // 44101 Hz has no common divisor with 16 kHz, so it takes the per-sample kernel path (16000
    // phases); the others use the phase table.
    for from in [44100u32, 48000, 22050, 8000, 44101] {
        for freq in [440.0, 1000.0, 3000.0, 6000.0] {
            if freq >= f64::from(from.min(16000)) * 0.45 {
                continue;
            }
            let out = resample(&tone(freq, from, 0.5), from, 16000);
            let (amp, residual) = fit(&out, freq, 16000);
            let db = 20.0 * (amp / 0.5).log10();
            eprintln!("{from} Hz -> 16 kHz, {freq} Hz: gain {db:.4} dB, residual {residual:e}");
            assert!(db.abs() <= 0.01, "{from} Hz, {freq} Hz tone: gain {db} dB");
            assert!(
                residual <= 0.5 * 1e-3,
                "{from} Hz, {freq} Hz tone: residual {residual}"
            );
        }
    }
}

#[test]
fn tones_above_the_new_nyquist_are_attenuated_80_db() {
    for from in [44100u32, 48000, 44101] {
        for freq in [8500.0, 9000.0, 12000.0, 20000.0] {
            let out = resample(&tone(freq, from, 0.5), from, 16000);
            let db = 20.0 * (rms(&out) / (0.5 / 2f64.sqrt())).log10();
            eprintln!("{from} Hz -> 16 kHz, {freq} Hz: {db:.1} dB");
            assert!(db <= -80.0, "{from} Hz, {freq} Hz tone: {db} dB");
        }
    }
}

#[test]
fn output_length_is_the_rounded_rate_ratio() {
    for (len, from, to) in [
        (44100usize, 44100u32, 16000u32),
        (1001, 48000, 16000),
        (7, 44100, 16000),
        (333, 8000, 16000),
        (1, 44101, 16000),
        (0, 48000, 16000),
    ] {
        let expected = ((len as f64) * f64::from(to) / f64::from(from)).round() as usize;
        assert_eq!(
            resample(&vec![0.25; len], from, to).len(),
            expected,
            "{len} at {from} -> {to}"
        );
    }
    let same: Vec<f32> = (0..100).map(|i| i as f32).collect();
    assert_eq!(resample(&same, 16000, 16000), same);
}

#[test]
fn a_time_shift_fails_the_phase_pinned_fit() {
    // The fit itself must catch a delay: the 48 kHz -> 16 kHz output of a 3 kHz tone, shifted by
    // one output sample, leaves a residual far above the bound the resampler meets.
    let out = resample(&tone(3000.0, 48000, 0.5), 48000, 16000);
    let (_, aligned) = fit(&out, 3000.0, 16000);
    let (_, shifted) = fit(&out[1..], 3000.0, 16000);
    assert!(
        aligned <= 1e-5 && shifted > 0.1,
        "aligned {aligned}, shifted {shifted}"
    );
}

#[test]
fn rates_outside_the_supported_range_give_no_samples() {
    let input = vec![0.25f32; 100];
    for (from, to) in [
        (1u32, 16000u32),
        (999, 16000),
        (16000, 1),
        (768_001, 16000),
        (0, 16000),
        (16000, u32::MAX),
    ] {
        assert!(resample(&input, from, to).is_empty(), "{from} -> {to}");
    }
    assert_eq!(resample(&input, 1000, 16000).len(), 1600);
    assert_eq!(resample(&input, 768_000, 16000).len(), 2);
}
