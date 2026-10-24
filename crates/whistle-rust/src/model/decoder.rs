//! The decoder (`whistle.DecoderBlock` in a `LaneStack`, run by `Whistle.decode` and
//! `Whistle.logits`): the scaled token embedding and the decoder memory (`decoder_input`), the
//! Engram keys and values (`engram`), eight blocks each mixed in by the mHC step (`mhc`), the
//! final lane mean and norm, and the tied A8 output head, giving [`Logits`]. Each block is the
//! Engram add at its site layers, gated causal self-attention (`attention`), gated
//! cross-attention (`cross_attention`) and a full-step Hadamard MLP (`hadamard`). The sub-steps
//! are public so each can be tested on its own; positions start at 0 here, and the pieces take
//! explicit history and positions so an incremental step can reuse them.

use crate::error::Error;

use super::cross_attention::CrossMemory;
use super::decoder_input::DecoderMemory;
use super::engram::engram_add;
use super::mhc::lane_mean_norm;
use super::ops::{a8, sigmoid, Rope};
use super::stem::Lanes;
use super::weights::{DecoderLayer, Matrix, Weights};

/// What one decoder call shares across its layers: the weights, the token count, the RoPE
/// table over positions `0..T` (`precompute_rope_freqs(qk_dim, T)`, as `Whistle.decode` builds
/// it; a position's angles do not depend on the table's length, so an incremental step may
/// build its own table over its maximum length), each layer's cross-attention keys and values
/// over the memory, and every Engram site's keys and values (`[sites, T, d_model]`).
pub struct DecoderContext<'w> {
    weights: &'w Weights,
    rope: Rope,
    cross: Vec<CrossMemory>,
    engram_k: Vec<f32>,
    engram_v: Vec<f32>,
    positions: usize,
}

impl<'w> DecoderContext<'w> {
    /// A context for the decoder input `tokens` (the whole sequence, BOS first) against
    /// `memory`. No tokens, more than `max_seq_len`, or a memory that does not fit (no frames
    /// included) is [`Error::Input`]; an Engram site past the last layer or an unsupported KV width is
    /// [`Error::Archive`].
    pub fn new(
        weights: &'w Weights,
        tokens: &[u32],
        memory: &DecoderMemory,
    ) -> Result<DecoderContext<'w>, Error> {
        let g = &weights.geometry;
        check_tokens(weights, tokens.len())?;
        let cross = cross_memories(weights, memory)?;
        let (mut engram_k, mut engram_v) = (Vec::new(), Vec::new());
        for e in &weights.engrams {
            let (k, v) = e.keys_values(tokens)?;
            engram_k.extend_from_slice(&k.data);
            engram_v.extend_from_slice(&v.data);
        }
        Ok(DecoderContext {
            weights,
            rope: Rope::new(g.qk_head_dim as usize, tokens.len(), g.rope_theta),
            cross,
            engram_k,
            engram_v,
            positions: tokens.len(),
        })
    }

    /// The token count `T`.
    pub fn positions(&self) -> usize {
        self.positions
    }

    fn check(&self, x: &[f32], what: &str) -> Result<(), Error> {
        let d = self.weights.embedding.cols;
        if self.positions.checked_mul(d) != Some(x.len()) {
            return Err(Error::Input(format!(
                "{what} holds {} values, not {} positions of {d}",
                x.len(),
                self.positions
            )));
        }
        Ok(())
    }
}

/// The checks every decoder call makes before it runs: the KV width is supported, `positions`
/// tokens fit `1..=max_seq_len` ([`Error::Input`] otherwise), and every Engram site is a layer
/// of the decoder with weights bound for it ([`Error::Archive`] otherwise).
pub(crate) fn check_tokens(weights: &Weights, positions: usize) -> Result<(), Error> {
    let g = &weights.geometry;
    super::bind::check_kv_bits(g.kv_bits)?;
    let max = g.max_seq_len as usize;
    if positions == 0 || positions > max {
        return Err(Error::Input(format!(
            "{positions} decoder tokens (1 to {max})"
        )));
    }
    let layers = weights.decoder.len();
    if let Some(&site) = g.engram_sites.iter().find(|&&l| l as usize >= layers) {
        return Err(Error::Archive(format!(
            "Engram site at layer {site} of a {layers}-layer decoder"
        )));
    }
    if g.engram_sites.len() != weights.engrams.len() {
        return Err(Error::Archive(format!(
            "{} Engram sites with {} bound",
            g.engram_sites.len(),
            weights.engrams.len()
        )));
    }
    Ok(())
}

/// Every layer's cross-attention keys and values over `memory`, computed once per decode.
pub(crate) fn cross_memories(
    weights: &Weights,
    memory: &DecoderMemory,
) -> Result<Vec<CrossMemory>, Error> {
    let kv_bits = weights.geometry.kv_bits;
    weights
        .decoder
        .iter()
        .map(|l| l.cross_attn.memory_kv(memory, kv_bits))
        .collect()
}

/// Layer `layer`'s Engram site flags (`LaneStack`): 1 for each site at this layer, else 0.
pub(crate) fn site_flags(weights: &Weights, layer: usize) -> Vec<f32> {
    let sites = &weights.geometry.engram_sites;
    sites
        .iter()
        .map(|&l| if l as usize == layer { 1.0 } else { 0.0 })
        .collect()
}

/// The scaled embedding of `tokens` broadcast to every mHC lane: the decoder stack's input.
pub(crate) fn input_lanes(weights: &Weights, tokens: &[u32]) -> Result<Lanes, Error> {
    let x = weights.embed(tokens)?;
    let n = weights.decoder_mhc.lanes;
    let mut data = Vec::with_capacity(x.data.len() * n);
    for row in x.data.chunks_exact(x.cols) {
        for _ in 0..n {
            data.extend_from_slice(row);
        }
    }
    Ok(Lanes {
        data,
        frames: x.rows,
        lanes: n,
        width: x.cols,
        valid_frames: x.rows,
    })
}

impl DecoderLayer {
    /// The self-attention module on `norm_in`'s output (`[T, d_model]`), positions `0..T`,
    /// returning `out_proj`'s output.
    pub fn self_attention(&self, x: &[f32], ctx: &DecoderContext<'_>) -> Result<Vec<f32>, Error> {
        ctx.check(x, "the self-attention input")?;
        self.self_attn
            .forward_causal(x, &ctx.rope, ctx.weights.geometry.kv_bits)
    }

    /// The cross-attention module of layer `layer` on `cross_norm`'s output, against the
    /// context's memory, returning `out_proj`'s output.
    pub fn cross_attention(
        &self,
        x: &[f32],
        layer: usize,
        ctx: &DecoderContext<'_>,
    ) -> Result<Vec<f32>, Error> {
        ctx.check(x, "the cross-attention input")?;
        let mem = ctx
            .cross
            .get(layer)
            .ok_or_else(|| Error::Input(format!("layer {layer} of {}", ctx.cross.len())))?;
        self.cross_attn.forward(x, mem)
    }

    /// The whole block of layer `layer` on its input `x` (`[T, d_model]`, the lane mix `u`):
    /// the Engram add when a site is at this layer, then
    /// `x1 = x + σ(attn_gate)·post_attn_norm(self_attn(norm_in(x)))`,
    /// `x2 = x1 + σ(cross_gate)·post_cross_norm(cross_attn(cross_norm(x1)))`,
    /// `out = x2 + hadamard_mlp(pre_hada_norm(x2))` (a full step, unlike the encoder's halves).
    pub fn block(
        &self,
        x: &[f32],
        layer: usize,
        ctx: &DecoderContext<'_>,
    ) -> Result<Vec<f32>, Error> {
        ctx.check(x, "the block input")?;
        let mem = ctx
            .cross
            .get(layer)
            .ok_or_else(|| Error::Input(format!("layer {layer} of {}", ctx.cross.len())))?;
        let engram = BlockEngram {
            keys: &ctx.engram_k,
            values: &ctx.engram_v,
            flags: site_flags(ctx.weights, layer),
        };
        self.block_with(x, ctx.weights, &engram, mem, |normed| {
            self.self_attention(normed, ctx)
        })
    }

    /// The block's arithmetic on rows `x` whatever produced its self-attention: the Engram add
    /// (`engram`, its keys and values for exactly these rows), `self_attn` on `norm_in(x)`
    /// (the whole causal sequence, or one cached step), cross-attention against `mem`, and the
    /// Hadamard MLP. [`DecoderLayer::block`] and the cached step share it.
    pub(crate) fn block_with(
        &self,
        x: &[f32],
        weights: &Weights,
        engram: &BlockEngram<'_>,
        mem: &CrossMemory,
        self_attn: impl FnOnce(&[f32]) -> Result<Vec<f32>, Error>,
    ) -> Result<Vec<f32>, Error> {
        // A layer with no site adds `0` in the reference, which leaves `x` as it is.
        let x = if engram.flags.contains(&1.0) {
            let d = weights.embedding.cols;
            engram_add(x, engram.keys, engram.values, &engram.flags, d)?
        } else {
            x.to_vec()
        };

        let attn = self_attn(&self.norm_in.forward(&x))?;
        let h = self.post_attn_norm.forward(&attn);
        let gate = sigmoid(self.attn_gate);
        let x: Vec<f32> = x.iter().zip(&h).map(|(x, h)| x + gate * h).collect();

        let cross = self.cross_attn.forward(&self.cross_norm.forward(&x), mem)?;
        let h = self.post_cross_norm.forward(&cross);
        let gate = sigmoid(self.cross_gate);
        let x: Vec<f32> = x.iter().zip(&h).map(|(x, h)| x + gate * h).collect();

        let m = self
            .hadamard_mlp
            .forward(&self.pre_hada_norm.forward(&x), &weights.hada_perms);
        Ok(x.iter().zip(&m).map(|(x, m)| x + m).collect())
    }

    /// Decoder layer `layer` (this layer's weights) on the lanes before it: the mHC step around
    /// [`DecoderLayer::block`], giving the lanes after it.
    pub fn forward(
        &self,
        lanes: &Lanes,
        layer: usize,
        ctx: &DecoderContext<'_>,
    ) -> Result<Lanes, Error> {
        if lanes.frames != ctx.positions {
            return Err(Error::Input(format!(
                "lanes of {} positions in a context of {}",
                lanes.frames, ctx.positions
            )));
        }
        ctx.weights
            .decoder_mhc
            .step(layer, lanes, |u| self.block(u, layer, ctx))
    }
}

/// The Engram inputs of one block call: every site's keys and values for the block's rows
/// (`[sites, rows, d_model]`) and this layer's site flags.
pub(crate) struct BlockEngram<'a> {
    pub(crate) keys: &'a [f32],
    pub(crate) values: &'a [f32],
    pub(crate) flags: Vec<f32>,
}

/// Next-token scores: `[positions, vocab]` row-major f32, row `t` scoring the token after
/// position `t` over the whole vocabulary (text tokens, then the language tags).
#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    data: Vec<f32>,
    positions: usize,
    vocab: usize,
}

impl Logits {
    pub fn positions(&self) -> usize {
        self.positions
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Every score, row-major.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The scores after position `t`, or None past the last position.
    pub fn row(&self, t: usize) -> Option<&[f32]> {
        (t < self.positions).then(|| &self.data[t * self.vocab..(t + 1) * self.vocab])
    }
}

/// The decoder over the bound weights: [`Decoder::forward`] scores a token sequence against
/// the decoder memory.
pub struct Decoder<'w> {
    weights: &'w Weights,
}

impl<'w> Decoder<'w> {
    pub fn new(weights: &'w Weights) -> Decoder<'w> {
        Decoder { weights }
    }

    /// `Whistle.logits` for the whole sequence `tokens` (BOS first) against `memory`:
    /// [`Decoder::hidden`] then [`Decoder::logits`].
    pub fn forward(&self, tokens: &[u32], memory: &DecoderMemory) -> Result<Logits, Error> {
        self.logits(&self.hidden(tokens, memory)?)
    }

    /// `Whistle.decode`: the scaled embedding broadcast to the lanes, every decoder layer, then
    /// [`Decoder::finish`], `[T, d_model]`. Errors are [`DecoderContext::new`]'s.
    pub fn hidden(&self, tokens: &[u32], memory: &DecoderMemory) -> Result<Matrix, Error> {
        let w = self.weights;
        let ctx = DecoderContext::new(w, tokens, memory)?;
        let mut lanes = input_lanes(w, tokens)?;
        for (l, layer) in w.decoder.iter().enumerate() {
            lanes = layer.forward(&lanes, l, &ctx)?;
        }
        self.finish(&lanes)
    }

    /// The stack's end on the last layer's lanes: the mean over the lanes, then the decoder's
    /// final ZC-RMSNorm, `[T, d_model]`.
    pub fn finish(&self, lanes: &Lanes) -> Result<Matrix, Error> {
        let width = self.weights.final_norm.scale.len();
        let expected = lanes
            .frames
            .checked_mul(lanes.lanes)
            .and_then(|v| v.checked_mul(width));
        if lanes.width != width
            || lanes.lanes != self.weights.decoder_mhc.lanes
            || expected != Some(lanes.data.len())
        {
            return Err(Error::Input(format!(
                "lanes hold {} values as {} frames of {} lanes × {}, not × {width}",
                lanes.data.len(),
                lanes.frames,
                lanes.lanes,
                lanes.width
            )));
        }
        Ok(Matrix {
            rows: lanes.frames,
            cols: width,
            data: lane_mean_norm(lanes, &self.weights.final_norm),
        })
    }

    /// The tied output head on the decoder's output: `A8(hidden) · embeddingᵀ` in f32 over the
    /// whole vocabulary. A hidden width other than `d_model` is [`Error::Input`].
    pub fn logits(&self, hidden: &Matrix) -> Result<Logits, Error> {
        let e = &self.weights.embedding;
        if hidden.cols != e.cols || hidden.rows.checked_mul(e.cols) != Some(hidden.data.len()) {
            return Err(Error::Input(format!(
                "decoder output [{}, {}] of {} values for a {}-wide embedding",
                hidden.rows,
                hidden.cols,
                hidden.data.len(),
                e.cols
            )));
        }
        let mut x = hidden.data.clone();
        a8(&mut x, e.cols);
        Ok(Logits {
            data: e.project(&x),
            positions: hidden.rows,
            vocab: e.rows,
        })
    }
}
