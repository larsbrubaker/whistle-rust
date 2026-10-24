//! The decoder's inputs other than the Engram sites (`Whistle.decode`): the scaled token
//! embedding ([`Weights::embed`]) and the cross-attention memory built from the encoder output
//! ([`Weights::decoder_memory`]: the encoder output plus the gated sinusoidal position encoding,
//! fake-quantized to A8). The Engram keys and values are in `engram`; the decoder stack consumes
//! all of them.

use crate::error::Error;

use super::ops::{a8, sigmoid};
use super::weights::{Matrix, Weights};

/// The decoder's cross-attention memory: `[frames, d_model]`, of which the first
/// `valid_frames` are real audio (cross-attention masks the rest).
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderMemory {
    pub memory: Matrix,
    pub valid_frames: usize,
}

/// `whistle.sinusoids`: `[frames, dim]` with `ang[t, i] = t · exp(-ln(10000) / (dim/2 − 1) · i)`
/// for `i < dim/2`, then `[sin(ang), cos(ang)]`. The step is a float64 constant the reference
/// casts to f32 before multiplying (its arrays are f32), and so is this.
fn sinusoids(frames: usize, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let step = (-crate::math::ln(10000.0) / (half as f64 - 1.0)) as f32;
    let inv: Vec<f32> = (0..half)
        .map(|i| crate::math::expf(step * i as f32))
        .collect();
    let mut out = Vec::with_capacity(frames * dim);
    for t in 0..frames {
        let angles = inv.iter().map(|&f| t as f32 * f);
        out.extend(angles.clone().map(crate::math::sinf));
        out.extend(angles.map(crate::math::cosf));
    }
    out
}

impl Weights {
    /// The decoder input: `embedding[token] · √d_model` per token, `[T, d_model]`. A token id
    /// outside the vocabulary is [`Error::Input`].
    pub fn embed(&self, tokens: &[u32]) -> Result<Matrix, Error> {
        let d = self.embedding.cols;
        // `math.sqrt(d)` is a float64 the reference multiplies f32 values by as f32.
        let scale = (d as f64).sqrt() as f32;
        let mut data = Vec::with_capacity(tokens.len() * d);
        for &token in tokens {
            let row = usize::try_from(token)
                .ok()
                .filter(|&r| r < self.embedding.rows)
                .ok_or_else(|| {
                    Error::Input(format!(
                        "token {token} is outside the vocabulary of {}",
                        self.embedding.rows
                    ))
                })?;
            let at = data.len();
            data.resize(at + d, 0.0);
            self.embedding.row_into(row, &mut data[at..])?;
            for v in &mut data[at..] {
                *v *= scale;
            }
        }
        Ok(Matrix::dense(tokens.len(), d, data))
    }

    /// The decoder's memory from the encoder output (`encoder_out`, `[frames, d_model]`
    /// row-major, the first `valid` frames real): `A8(enc + σ(pe_gate) · sinusoids)`. An output
    /// whose length is not `frames · d_model`, no frames, or more valid frames than frames, is
    /// [`Error::Input`].
    pub fn decoder_memory(
        &self,
        encoder_out: &[f32],
        frames: usize,
        valid: usize,
    ) -> Result<DecoderMemory, Error> {
        let d = self.embedding.cols;
        if frames.checked_mul(d) != Some(encoder_out.len()) {
            return Err(Error::Input(format!(
                "the encoder output holds {} values, not {frames} frames of {d}",
                encoder_out.len()
            )));
        }
        if valid > frames {
            return Err(Error::Input(format!(
                "the encoder output claims {valid} valid frames of {frames}"
            )));
        }
        if frames == 0 {
            return Err(Error::Input(
                "the encoder output has no frames (no encoded audio to attend to)".into(),
            ));
        }
        let gate = sigmoid(self.pe_gate);
        let pe = sinusoids(frames, d);
        let mut data: Vec<f32> = encoder_out
            .iter()
            .zip(&pe)
            .map(|(&e, &p)| e + gate * p)
            .collect();
        a8(&mut data, d);
        Ok(DecoderMemory {
            memory: Matrix::dense(frames, d, data),
            valid_frames: valid,
        })
    }
}
