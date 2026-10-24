// Unit tests for the private radix-2 FFT (`audio::fft`): it agrees with a direct DFT.

use super::Fft;

#[test]
fn fft_matches_a_direct_dft() {
    for size in [1usize, 2, 8, 64, 512] {
        let input: Vec<f64> = (0..size)
            .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
            .collect();
        let (mut re, mut im) = (input.clone(), vec![0.0; size]);
        Fft::new(size).forward(&mut re, &mut im);
        for k in 0..size {
            let (mut er, mut ei) = (0.0f64, 0.0f64);
            for (n, &x) in input.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * ((k * n) % size) as f64 / size as f64;
                er += x * a.cos();
                ei += x * a.sin();
            }
            assert!(
                (re[k] - er).abs() < 1e-9 && (im[k] - ei).abs() < 1e-9,
                "size {size} bin {k}"
            );
        }
    }
}
