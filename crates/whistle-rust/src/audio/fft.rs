//! The front end's FFT: an in-place iterative radix-2 complex FFT in f64, used by `mel` for the
//! 512-point transform of each real, windowed frame. f64 keeps the transform's own rounding far
//! below the f32 features it feeds (see `mel::features`).

/// A precomputed radix-2 FFT of one power-of-two size.
pub(crate) struct Fft {
    size: usize,
    /// e^{-2πik/size} for k < size/2.
    twiddles: Vec<(f64, f64)>,
}

impl Fft {
    /// `size` must be a power of two (the front end uses 512).
    pub(crate) fn new(size: usize) -> Fft {
        debug_assert!(size.is_power_of_two());
        let twiddles = (0..size / 2)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / size as f64;
                (angle.cos(), angle.sin())
            })
            .collect();
        Fft { size, twiddles }
    }

    /// Transforms `re`/`im` (each `size` long) in place: X[k] = Σ x[n] e^{-2πikn/size}.
    pub(crate) fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.size;
        debug_assert!(re.len() == n && im.len() == n);
        let bits = n.trailing_zeros();
        if bits == 0 {
            return;
        }
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = self.twiddles[k * stride];
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
    }
}

#[cfg(test)]
mod tests;
