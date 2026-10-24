//! Dequantization: fp16 to f32, and CQ ("Cactus Quants") records to row-major f32 matrices,
//! following upstream's `export._cq_unpack`. A CQ group dequantizes to
//! `(codebook[idx] * norm) @ H`, `H` the orthonormal Walsh–Hadamard matrix of the group size
//! (Sylvester order); `H` is symmetric, so each group is one fast Walsh–Hadamard transform.
//! Reads the validated [`Tensor`] records `cact` produces. [`CqRows`] keeps a CQ record packed
//! and dequantizes single rows on demand (the model's Engram tables).

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

/// The decoding parameters of one CQ record: its logical shape, padded row, width and
/// codebook. Shared by the whole-tensor path ([`Tensor::to_f32`]) and [`CqRows`].
#[derive(Debug, Clone)]
struct CqLayout {
    rows: usize,
    cols: usize,
    in_pad: usize,
    row_bytes: usize,
    bits: u32,
    group: usize,
    codebook: Vec<f32>,
}

impl CqLayout {
    fn new(tensor: &Tensor<'_>, archive: &Archive<'_>) -> Result<CqLayout, Error> {
        if tensor.dtype() != DType::Cq {
            return Err(Error::Archive(format!(
                "expected a CQ matrix, found {:?}",
                tensor.dtype()
            )));
        }
        // Shape, width, group and the blob size were validated by `Archive::parse`.
        let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
        let group = tensor.group() as usize;
        let in_pad = cols.next_multiple_of(group);
        Ok(CqLayout {
            rows,
            cols,
            in_pad,
            row_bytes: packed_row_bytes(in_pad, tensor.bits()),
            bits: tensor.bits(),
            group,
            codebook: codebook(archive, tensor.bits(), group)?,
        })
    }

    /// Where the per-group norms start in the blob (after every packed row).
    fn norms_at(&self) -> usize {
        self.rows * self.row_bytes
    }

    /// Row `r` padded to the group size (`padded.len() == in_pad`) from the record's `blob`:
    /// codebook values, times each group's norm, then the group's Walsh–Hadamard transform.
    fn decode(&self, blob: &[u8], r: usize, padded: &mut [f32]) -> Result<(), Error> {
        let (packed, norms) = blob.split_at(self.norms_at());
        let groups = self.in_pad / self.group;
        let row = &packed[r * self.row_bytes..(r + 1) * self.row_bytes];
        let width = if self.bits == TERNARY_RECORD_BITS {
            2
        } else {
            self.bits as usize
        };
        for (k, slot) in padded.iter_mut().enumerate() {
            let code = field(row, k, width);
            let idx = if self.bits == TERNARY_RECORD_BITS {
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
            *slot = self.codebook[idx];
        }
        for (g, chunk) in padded.chunks_exact_mut(self.group).enumerate() {
            let at = (r * groups + g) * 2;
            let norm = fp16_to_f32(u16::from_le_bytes([norms[at], norms[at + 1]]));
            for v in chunk.iter_mut() {
                *v *= norm;
            }
            walsh_hadamard(chunk);
        }
        Ok(())
    }
}

/// One CQ record kept packed, so a caller can dequantize a single row on demand: the Engram
/// tables (73,728 rows each) are only ever gathered by row, and as f32 they would cost about
/// 38 MB apiece. Holds the one owned copy of the record's blob (about 2.5 MB per table), so it
/// outlives the archive. Rows must be whole groups, which lets [`CqRows::row_into`] decode
/// straight into the caller's buffer without allocating.
#[derive(Debug, Clone)]
pub struct CqRows {
    layout: CqLayout,
    blob: Vec<u8>,
}

impl CqRows {
    /// The packed rows of a CQ tensor whose row length is a multiple of its group; any other
    /// dtype or a padded row is an error.
    pub fn from_tensor(tensor: &Tensor<'_>, archive: &Archive<'_>) -> Result<CqRows, Error> {
        let layout = CqLayout::new(tensor, archive)?;
        if layout.cols != layout.in_pad {
            return Err(Error::Archive(format!(
                "a row-addressed CQ matrix needs whole groups; rows of {} at group {}",
                layout.cols, layout.group
            )));
        }
        Ok(CqRows {
            layout,
            blob: tensor.bytes().to_vec(),
        })
    }

    /// The logical row count (`out`).
    pub fn rows(&self) -> usize {
        self.layout.rows
    }

    /// The row length (`in`).
    pub fn cols(&self) -> usize {
        self.layout.cols
    }

    /// Dequantizes row `r` into `out` (`cols` values), exactly as [`Tensor::to_f32`] would.
    /// Allocation-free.
    pub fn row_into(&self, r: usize, out: &mut [f32]) -> Result<(), Error> {
        if r >= self.layout.rows || out.len() != self.layout.cols {
            return Err(Error::Archive(format!(
                "CQ row {r} of {} rows into {} values (row length {})",
                self.layout.rows,
                out.len(),
                self.layout.cols
            )));
        }
        self.layout.decode(&self.blob, r, out)
    }

    /// Dequantizes row `r` as a new vector.
    pub fn row(&self, r: usize) -> Result<Vec<f32>, Error> {
        let mut out = vec![0.0f32; self.layout.cols];
        self.row_into(r, &mut out)?;
        Ok(out)
    }

    /// Every row, row-major: the whole matrix.
    pub fn to_f32(&self) -> Result<Vec<f32>, Error> {
        let mut values = vec![0.0f32; self.layout.rows * self.layout.cols];
        for (r, row) in values.chunks_exact_mut(self.layout.cols).enumerate() {
            self.layout.decode(&self.blob, r, row)?;
        }
        Ok(values)
    }
}

/// A whole CQ record, decoded from the archive's bytes in place (no copy of the blob).
fn dequantize_cq(tensor: &Tensor<'_>, archive: &Archive<'_>) -> Result<Vec<f32>, Error> {
    let layout = CqLayout::new(tensor, archive)?;
    let mut values = Vec::with_capacity(layout.rows * layout.cols);
    let mut padded = vec![0.0f32; layout.in_pad];
    for r in 0..layout.rows {
        layout.decode(tensor.bytes(), r, &mut padded)?;
        values.extend_from_slice(&padded[..layout.cols]);
    }
    Ok(values)
}

#[cfg(test)]
mod tests;
