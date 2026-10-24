//! The model's shared numeric primitives on row-major f32 slices: dense projection, A8 activation
//! fake quantization, ZC-RMSNorm, unit RMS, SiLU, sigmoid, GLU, softmax, Sinkhorn and RoPE. They
//! follow upstream's `architecture.py` and `quantize.py` operation for operation (same f32
//! arithmetic, same order within an element), so the stages built on them (`stem`, `hadamard`,
//! and the encoder and decoder blocks) reproduce the reference to f32 rounding.

/// `ZCRMSNorm`'s epsilon, and `_rms_unit`'s.
pub(crate) const NORM_EPS: f32 = 1e-6;

/// Activation fake-quantization width (`quantize.ACT_BITS`).
pub(crate) const ACT_BITS: u32 = 8;

/// `x [T, in] @ w[out, in]ᵀ -> [T, out]`: a Dense layer whose kernel the archive stores
/// transposed: each output one sequential f32 sum of products in input order (`kernels`
/// vectorizes across outputs, never within a sum), on the fastest SIMD path and the library's
/// worker threads.
pub(crate) fn linear(x: &[f32], w: &[f32], inputs: usize, outputs: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), inputs * outputs);
    debug_assert_eq!(x.len() % inputs.max(1), 0);
    super::kernels::linear_with(
        super::kernels::best_path(),
        super::kernels::threads(),
        x,
        w,
        inputs,
        outputs,
    )
}

/// Symmetric per-row fake quantization at `bits` (`quantize.fake_quant` with the group the
/// whole row): `scale = absmax / qmax` (1 for an all-zero row), `round_half_even(x / scale)`
/// clamped to `[-qmax - 1, qmax]`, times `scale`. The reference returns `x + (q - x)` (its
/// straight-through estimator), which is not always `q` in f32, so neither is this.
pub(crate) fn fake_quant_rows(x: &mut [f32], row: usize, bits: u32) {
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    for r in x.chunks_exact_mut(row) {
        let absmax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if absmax > 0.0 { absmax / qmax } else { 1.0 };
        for v in r.iter_mut() {
            let q = (*v / scale).round_ties_even().clamp(-qmax - 1.0, qmax) * scale;
            *v += q - *v;
        }
    }
}

/// A8 on each `row`-wide row (`architecture._aq` with `quantize.fake_quant_act`).
pub(crate) fn a8(x: &mut [f32], row: usize) {
    fake_quant_rows(x, row, ACT_BITS);
}

/// `ZCRMSNorm`: `(1 + scale) * x / sqrt(mean(x²) + 1e-6)` per row of `scale.len()`.
pub(crate) fn zc_rms_norm(x: &[f32], scale: &[f32]) -> Vec<f32> {
    let d = scale.len();
    let mut out = Vec::with_capacity(x.len());
    for r in x.chunks_exact(d) {
        let rms = (mean_square(r) + NORM_EPS).sqrt();
        out.extend(r.iter().zip(scale).map(|(&v, &s)| (1.0 + s) * v / rms));
    }
    out
}

/// `_rms_unit`: `x * rsqrt(mean(x²) + 1e-6)` per `row`-wide row.
pub(crate) fn rms_unit(x: &[f32], row: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for r in x.chunks_exact(row) {
        let inv = 1.0 / (mean_square(r) + NORM_EPS).sqrt();
        out.extend(r.iter().map(|&v| v * inv));
    }
    out
}

fn mean_square(r: &[f32]) -> f32 {
    r.iter().map(|v| v * v).sum::<f32>() / r.len() as f32
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `jax.nn.silu`: `x * sigmoid(x)`.
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `jax.nn.glu` over rows of `2 * half`: the first half times the sigmoid of the second.
pub(crate) fn glu(x: &[f32], half: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() / 2);
    for r in x.chunks_exact(2 * half) {
        let (a, b) = r.split_at(half);
        out.extend(a.iter().zip(b).map(|(&a, &b)| a * sigmoid(b)));
    }
    out
}

/// `jax.nn.softmax` in place over each `row`-wide row: subtract the max, exponentiate,
/// divide by the sum. Zero-wide rows have nothing to normalise, so that is a no-op.
pub(crate) fn softmax(x: &mut [f32], row: usize) {
    if row == 0 {
        return;
    }
    for r in x.chunks_exact_mut(row) {
        let max = r.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        for v in r.iter_mut() {
            *v = (*v - max).exp();
        }
        let sum: f32 = r.iter().sum();
        for v in r.iter_mut() {
            *v /= sum;
        }
    }
}

/// `jax.nn.logsumexp` of values read through an iterator: `max + log(Σ exp(v - max))`.
fn logsumexp(values: impl Iterator<Item = f32> + Clone) -> f32 {
    let max = values.clone().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    max + values.map(|v| (v - max).exp()).sum::<f32>().ln()
}

/// `_sinkhorn`: 20 rounds of row then column log-normalisation of each `n × n` matrix in
/// `logits`, then `exp`, giving doubly stochastic matrices.
pub(crate) fn sinkhorn(logits: &mut [f32], n: usize) {
    for m in logits.chunks_exact_mut(n * n) {
        for _ in 0..20 {
            for r in 0..n {
                let lse = logsumexp(m[r * n..(r + 1) * n].iter().copied());
                for v in &mut m[r * n..(r + 1) * n] {
                    *v -= lse;
                }
            }
            for c in 0..n {
                let lse = logsumexp((0..n).map(|r| m[r * n + c]));
                for r in 0..n {
                    m[r * n + c] -= lse;
                }
            }
        }
        for v in m.iter_mut() {
            *v = v.exp();
        }
    }
}

/// `precompute_rope_freqs`: `cos` and `sin` of `t · θ^(-2i / head_dim)` for positions
/// `t < seq_len` and `i < head_dim / 2`, each `[seq_len, head_dim / 2]` row-major.
pub(crate) struct Rope {
    pub(crate) half: usize,
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
}

impl Rope {
    pub(crate) fn new(head_dim: usize, seq_len: usize, theta: f32) -> Rope {
        let half = head_dim / 2;
        // freqs = 1 / theta ** (arange(0, head_dim, 2) / head_dim), all in f32.
        let freqs: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for t in 0..seq_len {
            for &f in &freqs {
                let angle = t as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Rope { half, cos, sin }
    }

    /// `apply_rope` on one head vector at position `t`, half-split layout: with `x1`, `x2` the
    /// two halves, `[x1·cos − x2·sin, x2·cos + x1·sin]`.
    pub(crate) fn apply(&self, x: &mut [f32], t: usize) {
        let h = self.half;
        let (cos, sin) = (&self.cos[t * h..(t + 1) * h], &self.sin[t * h..(t + 1) * h]);
        let (x1, x2) = x.split_at_mut(h);
        for i in 0..h {
            let (a, b) = (x1[i], x2[i]);
            x1[i] = a * cos[i] - b * sin[i];
            x2[i] = b * cos[i] + a * sin[i];
        }
    }
}

#[cfg(test)]
mod tests;
