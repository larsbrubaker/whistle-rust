//! The Hadamard MLP's forward pass (`architecture.HadamardMLP`) and the ZC-RMSNorm that feeds
//! it: on the width padded to `n` (a power of two),
//! `d4 · K3(d3 · P2(K2(silu(d2 · cond · P1(K1(d1 · x)) + b2))))`, each `K` the Kronecker product
//! of two learned factors applied as `aᵀ · z · b` to `z` reshaped `[a, b]` (`_kron_apply`), `P1`,
//! `P2` the archive's fixed permutations, and `cond = 1 + softmax(x · cond_v) · cond_u`. The
//! weights are the [`HadamardMlp`] bound in `weights`.

use super::ops::{silu_in_place, softmax, zc_rms_norm};
use super::weights::{HadaPerms, HadamardMlp, Matrix, Norm};

impl Norm {
    /// ZC-RMSNorm over rows of `scale.len()`: `(1 + scale) · x / sqrt(mean(x²) + 1e-6)`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        zc_rms_norm(x, &self.scale)
    }
}

/// `_kron_apply(z, a, b)` on one row: `z` as `[a.rows, b.rows]`, `out[k, l] =
/// Σ_ij z[i, j] · a[i, k] · b[j, l]`, contracted over `j` first.
fn kron_apply(z: &[f32], a: &Matrix, b: &Matrix, out: &mut [f32]) {
    let (na, nb) = (a.rows, b.rows);
    let (ad, bd) = (a.data(), b.data());
    // zb[i, l] = Σ_j z[i, j] · b[j, l]
    let mut zb = vec![0.0f32; na * nb];
    for i in 0..na {
        let zi = &z[i * nb..(i + 1) * nb];
        let row = &mut zb[i * nb..(i + 1) * nb];
        for (j, &zij) in zi.iter().enumerate() {
            let bj = &bd[j * nb..(j + 1) * nb];
            for (o, &bjl) in row.iter_mut().zip(bj) {
                *o += zij * bjl;
            }
        }
    }
    out.fill(0.0);
    for i in 0..na {
        let ai = &ad[i * na..(i + 1) * na];
        let zbi = &zb[i * nb..(i + 1) * nb];
        for (k, &aik) in ai.iter().enumerate() {
            let row = &mut out[k * nb..(k + 1) * nb];
            for (o, &v) in row.iter_mut().zip(zbi) {
                *o += aik * v;
            }
        }
    }
}

impl HadamardMlp {
    /// The MLP over rows of `d_model` (`cond_v.rows`), returning the same shape.
    pub fn forward(&self, x: &[f32], perms: &HadaPerms) -> Vec<f32> {
        let d = self.cond_v.rows;
        let rank = self.cond_v.cols;
        let n = self.d1.len();
        let (cond_v, cond_u) = (self.cond_v.data(), self.cond_u.data());
        let mut out = Vec::with_capacity(x.len());
        let (mut z, mut k) = (vec![0.0f32; n], vec![0.0f32; n]);
        let mut gain = vec![0.0f32; rank];
        for row in x.chunks_exact(d) {
            // cond = 1 + softmax(x · cond_v) · cond_u
            for (r, g) in gain.iter_mut().enumerate() {
                *g = row
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * cond_v[i * rank + r])
                    .sum();
            }
            softmax(&mut gain, rank);
            let cond: Vec<f32> = (0..n)
                .map(|j| {
                    let s: f32 = gain
                        .iter()
                        .enumerate()
                        .map(|(r, &p)| p * cond_u[r * n + j])
                        .sum();
                    1.0 + s
                })
                .collect();

            // Padding past d_model is zero.
            for (j, zj) in z.iter_mut().enumerate() {
                *zj = self.d1[j] * row.get(j).copied().unwrap_or(0.0);
            }
            let [(a1, b1), (a2, b2), (a3, b3)] = &self.factors;
            kron_apply(&z, a1, b1, &mut k);
            for (j, zj) in z.iter_mut().enumerate() {
                let v = k[perms.p1[j]];
                *zj = self.d2[j] * cond[j] * v + self.b2[j];
            }
            silu_in_place(&mut z);
            kron_apply(&z, a2, b2, &mut k);
            for (j, zj) in z.iter_mut().enumerate() {
                *zj = self.d3[j] * k[perms.p2[j]];
            }
            kron_apply(&z, a3, b3, &mut k);
            out.extend((0..d).map(|j| self.d4[j] * k[j]));
        }
        out
    }
}
