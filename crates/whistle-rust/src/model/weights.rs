//! The model's weights bound by role: typed structs for the stem, the encoder and decoder layers,
//! their mHC lane stacks, the Engram sites and the shared tensors, built from a parsed archive by
//! [`Weights::from_archive`]. The archive's tensors carry no names, so `bind` derives each
//! index's role from the geometry and the audio manifest, in the order upstream's
//! `export._tensors` and `_audio_tensors` write them. Matrices are dequantized to row-major
//! `[out, in]` f32 ([`Matrix`]); the Engram tables stay packed ([`CqRows`]).

use crate::cact::{Archive, Geometry};
use crate::error::Error;
use crate::quant::CqRows;

use super::bind;

/// A row-major f32 matrix, `[rows, cols]` as the archive stores it (`[out, in]` for a Dense
/// kernel, so a projection is [`Matrix::project`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f32>,
}

impl Matrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// `x [T, cols] -> [T, rows]`: the Dense layer this `[out, in]` kernel stores.
    pub(crate) fn project(&self, x: &[f32]) -> Vec<f32> {
        super::ops::linear(x, &self.data, self.cols, self.rows)
    }
}

/// A ZC-RMSNorm's learned scale (`(1 + scale) · x / rms(x)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Norm {
    pub scale: Vec<f32>,
}

/// Self-attention projections. The decoder's has causal depthwise taps on Q, K and V; the
/// encoder's has none.
#[derive(Debug, Clone)]
pub struct SelfAttention {
    /// `[heads · qk_dim, d_model]`.
    pub q_proj: Matrix,
    /// `[kv_heads · qk_dim, d_model]`.
    pub k_proj: Matrix,
    /// `[kv_heads · v_dim, d_model]`.
    pub v_proj: Matrix,
    /// `[taps, width]` each, decoder only.
    pub taps: Option<QkvTaps>,
    /// `[qk_dim]`, per head.
    pub q_norm: Norm,
    pub k_norm: Norm,
    /// `[heads · v_dim, d_model]`.
    pub gate_proj: Matrix,
    /// `[d_model, heads · v_dim]`.
    pub out_proj: Matrix,
}

/// The decoder's causal Q, K, V depthwise convolution taps, `[taps, width]` each.
#[derive(Debug, Clone)]
pub struct QkvTaps {
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
}

/// Decoder cross-attention: queries from the decoder, keys and values from the memory.
#[derive(Debug, Clone)]
pub struct CrossAttention {
    /// `[heads · qk_dim, d_model]`.
    pub q_proj: Matrix,
    /// `[heads · qk_dim, d_model]`.
    pub k_proj: Matrix,
    /// `[heads · v_dim, d_model]`.
    pub v_proj: Matrix,
    /// `[heads · v_dim, d_model]`.
    pub gate_proj: Matrix,
    /// `[d_model, heads · v_dim]`.
    pub out_proj: Matrix,
    pub q_norm: Norm,
    pub k_norm: Norm,
}

/// One `HadamardMLP` (see `hadamard`): diagonals, Kronecker factors and the rank-8 gain.
#[derive(Debug, Clone)]
pub struct HadamardMlp {
    /// `[n]` each, `n` the width padded to a power of two.
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub b2: Vec<f32>,
    pub d3: Vec<f32>,
    pub d4: Vec<f32>,
    /// `(a, b)` per stage: `[ba, ba]` and `[bb, bb]`, `ba · bb = n`.
    pub factors: [(Matrix, Matrix); 3],
    /// `[d_model, rank]`, used as `x · cond_v`.
    pub cond_v: Matrix,
    /// `[rank, n]`.
    pub cond_u: Matrix,
}

/// The two fixed Hadamard MLP permutations (`hada_p1`, `hada_p2`): output `m` takes input
/// `p[m]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadaPerms {
    pub p1: Vec<usize>,
    pub p2: Vec<usize>,
}

/// One decoder layer (`DecoderBlock`).
#[derive(Debug, Clone)]
pub struct DecoderLayer {
    /// Before self-attention (`ZCRMSNorm_0`, exported as `norm_in`).
    pub norm_in: Norm,
    pub self_attn: SelfAttention,
    pub post_attn_norm: Norm,
    /// The raw gate logit; the block uses `sigmoid(attn_gate)`.
    pub attn_gate: f32,
    pub cross_norm: Norm,
    pub cross_attn: CrossAttention,
    pub post_cross_norm: Norm,
    pub cross_gate: f32,
    pub pre_hada_norm: Norm,
    pub hadamard_mlp: HadamardMlp,
}

/// One encoder layer (`EncoderBlock`).
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    pub pre_hada_norm_0: Norm,
    pub hadamard_mlp_0: HadamardMlp,
    /// Before self-attention (`ZCRMSNorm_0`).
    pub attn_norm: Norm,
    pub self_attn: SelfAttention,
    pub post_attn_norm: Norm,
    pub attn_gate: f32,
    pub conv_norm: Norm,
    /// `[2 · d_model, d_model]`.
    pub pw1: Matrix,
    /// The depthwise kernel, `[conv_kernel, d_model]`.
    pub dw: Matrix,
    pub conv_out_norm: Norm,
    /// `[d_model, d_model]`.
    pub pw2: Matrix,
    pub pre_hada_norm: Norm,
    pub hadamard_mlp: HadamardMlp,
}

/// An mHC lane stack's per-layer mixing parameters (`LaneStack`), `L` layers of `n` lanes.
#[derive(Debug, Clone)]
pub struct Mhc {
    pub lanes: usize,
    /// `[L]` each.
    pub a_pre: Vec<f32>,
    pub a_post: Vec<f32>,
    pub a_res: Vec<f32>,
    /// `[L, n]` each.
    pub b_pre: Vec<f32>,
    pub b_post: Vec<f32>,
    /// `[L, n, n]`.
    pub b_res: Vec<f32>,
    /// `[L · n, n · d_model]`: layer `l`'s rows `l·n..(l+1)·n` are `phi_pre[l]` transposed.
    pub phi_pre: Matrix,
    pub phi_post: Matrix,
    /// `[L · n², n · d_model]`.
    pub phi_res: Matrix,
}

/// One Engram site: hashed n-gram tables and their key/value projections.
#[derive(Debug, Clone)]
pub struct Engram {
    /// `[tables · slots, sub_dim]`, kept packed: only ever gathered by row. Table
    /// `t = order_index · hash_heads + head` (orders major, as `engram_indices` stacks them),
    /// and slot `idx` of table `t` is row `t · slots + idx` (`Engram.__call__` gathers
    /// `tables[arange(num_tables), indices]`).
    pub tables: CqRows,
    /// `[d_model, tables · sub_dim]`.
    pub key_proj: Matrix,
    pub value_proj: Matrix,
    /// The value convolution taps, `[conv_taps, d_model]`.
    pub taps: Matrix,
    /// The n-gram hashing geometry, checked against `Whistle.decode` when bound.
    pub hash: EngramHash,
}

/// How an Engram site hashes and convolves (`Whistle.decode`), from the archive header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramHash {
    /// The n-gram orders, one group of `heads` tables each (2 and 3).
    pub orders: Vec<u32>,
    /// Hash heads per order: `tables / orders`.
    pub heads: usize,
    /// Slots per table.
    pub slots: u32,
    /// The value convolution's dilation: `max(orders)`, as `Whistle` builds its Engrams.
    pub dilation: usize,
}

/// One stem stage after the first: a depthwise 3×3 conv then a pointwise dense.
#[derive(Debug, Clone)]
pub struct StemStage {
    /// `[9, channels]`, tap `3·i + j` for kernel row (time) `i`, column (mel) `j`.
    pub dw: Matrix,
    /// `[channels, channels]`.
    pub pw: Matrix,
}

/// The convolutional stem (`whistle.Stem`, forward in `stem`).
#[derive(Debug, Clone)]
pub struct Stem {
    /// Stage 1's 3×3 kernel from the single mel plane to every channel, `[9, channels]`.
    pub w: Matrix,
    pub stages: Vec<StemStage>,
    /// `[d_model, channels · mels_out]`.
    pub out: Matrix,
    /// The mel bins the stem takes (80).
    pub mels: usize,
}

/// The audio section's manifest (`audio.manifest`), the audio geometry the header lacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioManifest {
    pub version: u32,
    pub text: u32,
    pub enc_layers: u32,
    pub enc_lanes: u32,
    pub conv_kernel: u32,
    pub stem: u32,
    pub stem_stages: u32,
    pub mels: u32,
    pub rate: u32,
    pub fft: u32,
    pub win: u32,
    pub hop: u32,
}

/// Every weight of the model, bound by role.
#[derive(Debug, Clone)]
pub struct Weights {
    pub geometry: Geometry,
    pub audio: AudioManifest,
    /// `[vocab, d_model]`, also the tied output projection.
    pub embedding: Matrix,
    pub decoder: Vec<DecoderLayer>,
    pub decoder_mhc: Mhc,
    pub hada_perms: HadaPerms,
    pub engrams: Vec<Engram>,
    /// The decoder stack's final norm.
    pub final_norm: Norm,
    /// The raw gate logit on the decoder memory's position encoding.
    pub pe_gate: f32,
    pub encoder: Vec<EncoderLayer>,
    pub encoder_mhc: Mhc,
    pub encoder_final_norm: Norm,
    pub stem: Stem,
    /// The mel filterbank stored in the archive, `[fft / 2 + 1, mels]`.
    pub mel: Matrix,
    pub(crate) names: Vec<String>,
}

impl Weights {
    /// Binds every tensor of `archive` by role. Fails with [`Error::Archive`] when the tensor
    /// count, a dtype or a shape does not fit the geometry, or when the archive carries probe
    /// heads (unsupported: Whistle has none).
    pub fn from_archive(archive: &Archive<'_>) -> Result<Weights, Error> {
        bind::bind(archive)
    }

    /// The upstream export name of each archive tensor, by index (`export._tensors` then
    /// `_audio_tensors`, then `tokenizer`).
    pub fn tensor_names(&self) -> &[String] {
        &self.names
    }
}
