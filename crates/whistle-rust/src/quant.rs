//! Dequantization: fp16 to f32, and CQ ("Cactus Quants") records to row-major f32 matrices,
//! following upstream's `export._cq_unpack`. A CQ group dequantizes to
//! `(codebook[idx] * norm) @ H`, `H` the orthonormal Walsh–Hadamard matrix of the group size
//! (Sylvester order); `H` is symmetric, so each group is one fast Walsh–Hadamard transform.
//! Reads the validated [`Tensor`] records `cact` produces.

use crate::cact::{packed_row_bytes, Archive, DType, Tensor, TERNARY_RECORD_BITS};
use crate::error::Error;

/// The 3-level Lloyd–Max centroid magnitude of the ternary codebook (`quantize._TERNARY_CB`).
const TERNARY_CENTROID: f64 = 1.224_006_4;

/// IEEE 754 binary16 to f32, exact for every input (subnormals, infinities, NaN payloads).
pub fn fp16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = u32::from((half >> 10) & 0x1f);
    let mantissa = u32::from(half & 0x03ff);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: value = mantissa * 2^-24, exactly representable in f32.
            let magnitude = mantissa as f32 * f32::from_bits(0x3380_0000); // 2^-24
            return if sign != 0 { -magnitude } else { magnitude };
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// In-place orthonormal Walsh–Hadamard transform in Sylvester order: `values @ H` with
/// `H = Hadamard(n) / sqrt(n)`. `n` must be a power of two.
pub fn walsh_hadamard(values: &mut [f32]) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    // Scale first, as `H`'s entries are the f32 `±1/sqrt(n)` the reference multiplies by.
    let scale = (1.0 / (n as f64).sqrt()) as f32;
    for v in values.iter_mut() {
        *v *= scale;
    }
    let mut half = 1;
    while half < n {
        for block in values.chunks_exact_mut(half * 2) {
            let (lo, hi) = block.split_at_mut(half);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (x, y) = (*a, *b);
                *a = x + y;
                *b = x - y;
            }
        }
        half *= 2;
    }
}

impl Tensor<'_> {
    /// The tensor's values as row-major f32 (`[out, in]` for CQ, padding dropped). RAW
    /// tensors have no numeric value and return an error.
    pub fn to_f32(&self, archive: &Archive<'_>) -> Result<Vec<f32>, Error> {
        let bytes = self.bytes();
        match self.dtype() {
            DType::Fp16 => Ok(bytes
                .chunks_exact(2)
                .map(|b| fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()),
            DType::Fp32 => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            DType::Cq => dequantize_cq(self, archive),
            DType::Raw => Err(Error::Archive(
                "a RAW tensor has no numeric values".to_string(),
            )),
        }
    }
}

/// The unit codebook for a record width: stored for 2-4 bits, analytic for 1 bit and ternary.
fn codebook(archive: &Archive<'_>, bits: u32, group: usize) -> Result<Vec<f32>, Error> {
    let root = (group as f64).sqrt();
    match bits {
        1 => {
            let c = ((2.0 / std::f64::consts::PI).sqrt() / root) as f32;
            Ok(vec![-c, c])
        }
        TERNARY_RECORD_BITS => {
            let c = (TERNARY_CENTROID / root) as f32;
            Ok(vec![-c, 0.0, c])
        }
        _ => archive
            .codebook(bits)
            .map(<[f32]>::to_vec)
            .ok_or_else(|| Error::Archive(format!("no codebook for CQ width {bits}"))),
    }
}

/// Index `k` of an LSB-first bitstream of `width`-bit fields (width <= 4).
fn field(row: &[u8], k: usize, width: usize) -> usize {
    let bit = k * width;
    let (byte, shift) = (bit / 8, bit % 8);
    let mut word = usize::from(row[byte]);
    if shift + width > 8 {
        word |= usize::from(row[byte + 1]) << 8;
    }
    (word >> shift) & ((1 << width) - 1)
}

fn dequantize_cq(tensor: &Tensor<'_>, archive: &Archive<'_>) -> Result<Vec<f32>, Error> {
    // Shape, width, group and the blob size were validated by `Archive::parse`.
    let (out, in_dim) = (tensor.shape()[0], tensor.shape()[1]);
    let bits = tensor.bits();
    let group = tensor.group() as usize;
    let in_pad = in_dim.next_multiple_of(group);
    let groups = in_pad / group;
    let row_bytes = packed_row_bytes(in_pad, bits);
    let (packed, norms) = tensor.bytes().split_at(out * row_bytes);
    let cb = codebook(archive, bits, group)?;
    let width = if bits == TERNARY_RECORD_BITS {
        2
    } else {
        bits as usize
    };
    let mut values = Vec::with_capacity(out * in_dim);
    let mut padded = vec![0.0f32; in_pad];
    for r in 0..out {
        let row = &packed[r * row_bytes..(r + 1) * row_bytes];
        for (k, slot) in padded.iter_mut().enumerate() {
            let code = field(row, k, width);
            let idx = if bits == TERNARY_RECORD_BITS {
                // Crumb codes 3, 0, 1 are trits 0, 1, 2 (sign-extended -1, 0, +1).
                match code {
                    3 => 0,
                    0 => 1,
                    1 => 2,
                    _ => {
                        return Err(Error::Archive(format!(
                            "ternary crumb code 2 at row {r}, column {k}"
                        )))
                    }
                }
            } else {
                code
            };
            *slot = cb[idx];
        }
        for (g, chunk) in padded.chunks_exact_mut(group).enumerate() {
            let at = (r * groups + g) * 2;
            let norm = fp16_to_f32(u16::from_le_bytes([norms[at], norms[at + 1]]));
            for v in chunk.iter_mut() {
                *v *= norm;
            }
            walsh_hadamard(chunk);
        }
        values.extend_from_slice(&padded[..in_dim]);
    }
    Ok(values)
}

#[cfg(test)]
mod tests;
