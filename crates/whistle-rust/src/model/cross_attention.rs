//! The decoder's gated cross-attention (`whistle.CrossAttention`, flash path, quant on): queries
//! from the A8 decoder stream, keys and values from the decoder memory (`decoder_input`, already
//! A8, so no A8 of its own), ZC-RMSNorm on q and k per head, no RoPE, A8 on q and the KV path at
//! the archive's `kv_bits`, attention over the valid memory frames, the sigmoid gate, A8 and
//! `out_proj`. The memory side ([`CrossAttention::memory_kv`]) depends only on the audio, so a
//! decoder computes it once per layer and reuses it for every step; the head arithmetic is
//! shared with self-attention (`attention`).

use crate::error::Error;

use super::attention::{attend_with, gate_out, split_heads, Heads, ProbSink, Projections};
use super::decoder_input::DecoderMemory;
use super::ops::{a8, fake_quant_rows, zc_rms_norm};
use super::weights::CrossAttention;

/// One layer's cross-attention keys and values over the memory, ready to attend:
/// k `[heads, frames, qk_dim]` (normed, KV-quantized), v `[heads, frames, v_dim]`
/// (KV-quantized), and the valid frame count the mask admits.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossMemory {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
}

impl CrossAttention {
    /// 8 query heads over 8 KV heads (no grouping): qk 48, v 64 per head.
    fn heads(&self) -> Heads {
        let qk_dim = self.q_norm.scale.len().max(1);
        let kv_heads = (self.k_proj.rows / qk_dim).max(1);
        Heads {
            heads: self.q_proj.rows / qk_dim,
            kv_heads,
            qk_dim,
            v_dim: self.v_proj.rows / kv_heads,
        }
    }

    fn check_width(&self, x: &[f32], what: &str) -> Result<usize, Error> {
        let d = self.q_proj.cols;
        if d == 0 || !x.len().is_multiple_of(d) {
            return Err(Error::Input(format!(
                "{what} holds {} values, not rows of {d}",
                x.len()
            )));
        }
        Ok(x.len() / d)
    }

    /// The four projections: q and the gate from `A8(x)` (`x` `[T, d_model]`, `cross_norm`'s
    /// output), k and v from `memory` (`[frames, d_model]`, used as given: the decoder memory is
    /// already A8). Rows of the wrong width are [`Error::Input`].
    pub fn projections(&self, x: &[f32], memory: &[f32]) -> Result<Projections, Error> {
        self.check_width(x, "the cross-attention input")?;
        self.check_width(memory, "the decoder memory")?;
        let mut xq = x.to_vec();
        a8(&mut xq, self.q_proj.cols);
        Ok(Projections {
            q: self.q_proj.project(&xq),
            k: self.k_proj.project(memory),
            v: self.v_proj.project(memory),
            gate: self.gate_proj.project(&xq),
        })
    }

    /// The query projection split into heads and normed (`q_norm`), `[heads, T, qk_dim]`.
    pub fn query_heads(&self, q: &[f32]) -> Vec<f32> {
        let h = self.heads();
        zc_rms_norm(&split_heads(q, h.heads, h.qk_dim), &self.q_norm.scale)
    }

    /// The key projection split into heads and normed (`k_norm`), `[kv_heads, frames, qk_dim]`.
    pub fn key_heads(&self, k: &[f32]) -> Vec<f32> {
        let h = self.heads();
        zc_rms_norm(&split_heads(k, h.kv_heads, h.qk_dim), &self.k_norm.scale)
    }

    /// This layer's keys and values over `memory`: the k and v projections, the head split,
    /// `k_norm`, and the KV path at `kv_bits` (8: per-head A8 over each head vector, 0: none).
    /// A memory with no frames, or more valid frames than frames, is [`Error::Input`].
    pub fn memory_kv(&self, memory: &DecoderMemory, kv_bits: u32) -> Result<CrossMemory, Error> {
        let m = memory.memory.data();
        let frames = self.check_width(&m, "the decoder memory")?;
        if frames == 0 {
            return Err(Error::Input(
                "the decoder memory has no frames (no encoded audio to attend to)".into(),
            ));
        }
        if memory.valid_frames > frames {
            return Err(Error::Input(format!(
                "{} valid memory frames of {frames}",
                memory.valid_frames
            )));
        }
        let h = self.heads();
        let mut k = self.key_heads(&self.k_proj.project(&m));
        let mut v = split_heads(&self.v_proj.project(&m), h.kv_heads, h.v_dim);
        if kv_bits > 0 {
            fake_quant_rows(&mut k, h.qk_dim, kv_bits);
            fake_quant_rows(&mut v, h.v_dim, kv_bits);
        }
        Ok(CrossMemory {
            k,
            v,
            frames,
            valid_frames: memory.valid_frames,
        })
    }

    /// Cross-attention on `x` (`[T, d_model]`, `cross_norm`'s output) against `mem`: A8 then the
    /// q and gate projections, `q_norm`, A8 on q, attention where every query sees the valid
    /// memory frames (the mask covers keys only), `· σ(gate)`, A8, then `out_proj`. Each row
    /// depends only on its own input, so a cached step calls this on one row.
    pub fn forward(&self, x: &[f32], mem: &CrossMemory) -> Result<Vec<f32>, Error> {
        let (out, gate) = self.attention(x, mem, None)?;
        Ok(gate_out(&self.out_proj, out, &gate))
    }

    /// [`CrossAttention::forward`] that also returns the attention probabilities of the query
    /// heads in `heads` (ascending) over the valid memory frames, `[heads, T, valid_frames]`,
    /// copied from inside the same call, so the output is bit-identical to `forward`'s.
    pub fn forward_with_weights(
        &self,
        x: &[f32],
        mem: &CrossMemory,
        heads: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let mut sink = ProbSink {
            heads: heads.to_vec(),
            keep: mem.valid_frames,
            data: Vec::new(),
        };
        let (out, gate) = self.attention(x, mem, Some(&mut sink))?;
        Ok((gate_out(&self.out_proj, out, &gate), sink.data))
    }

    /// The attention before the gate, `[T, heads · v_dim]`, and the gate projection, filling
    /// `sink` (when given) with the probabilities it mixed the values with.
    pub(crate) fn attention(
        &self,
        x: &[f32],
        mem: &CrossMemory,
        sink: Option<&mut ProbSink>,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let h = self.heads();
        if mem.k.len() != h.kv_heads * mem.frames * h.qk_dim
            || mem.v.len() != h.kv_heads * mem.frames * h.v_dim
        {
            return Err(Error::Input(format!(
                "cross-attention memory of {} keys and {} values is not {} frames",
                mem.k.len(),
                mem.v.len(),
                mem.frames
            )));
        }
        self.check_width(x, "the cross-attention input")?;
        let mut xq = x.to_vec();
        a8(&mut xq, self.q_proj.cols);
        let mut q = self.query_heads(&self.q_proj.project(&xq));
        a8(&mut q, h.qk_dim);
        let gate = self.gate_proj.project(&xq);
        let valid = mem.valid_frames;
        let out = attend_with(&q, &mem.k, &mem.v, h, |_, s| s < valid, sink);
        Ok((out, gate))
    }

    /// The query head count this layer attends with.
    pub fn head_count(&self) -> usize {
        self.heads().heads
    }

    /// The value width per head (the attention output is `[T, heads · v_dim]`).
    pub fn value_dim(&self) -> usize {
        self.heads().v_dim
    }
}

#[cfg(test)]
mod tests;
