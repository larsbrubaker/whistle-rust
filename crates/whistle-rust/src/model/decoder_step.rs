//! Incremental decoding: [`DecoderState`] runs the decoder (`decoder`) one token at a time with
//! a KV cache, so a greedy decode costs one position per step instead of re-scoring its whole
//! prefix. The cross-attention keys and values are computed once (`cross_attention`); each
//! layer keeps its self-attention cache (`attention::KvCache`: the taps' raw history, the keys
//! after norm, RoPE and the KV path, the values after the KV path); each Engram site keeps its
//! projected values for the value conv (`engram`). Every step runs the same functions the full
//! pass does on the new row only (the block core, the mHC step, the final norm and the head),
//! so its logits equal [`super::Decoder::forward`]'s row for that position. A state is
//! [`Clone`] so a beam search can fork a hypothesis: the RoPE table and the cross-attention
//! memory are fixed for the decode and shared (`Arc`); only the growing caches are copied.

use std::sync::Arc;

use crate::error::Error;

use super::attention::KvCache;
use super::cross_attention::CrossMemory;
use super::decoder::{check_tokens, cross_memories, input_lanes, site_flags, BlockEngram};
use super::decoder_input::DecoderMemory;
use super::ops::Rope;
use super::weights::Weights;
use super::Decoder;

/// One Engram site's running state: every position's projected values before the value conv
/// (`[positions, d_model]`), which the conv reads back up to `(taps − 1) · dilation` positions.
#[derive(Clone)]
struct EngramCache {
    values: Vec<f32>,
}

/// A decode in progress against one decoder memory. [`DecoderState::step`] feeds the token at
/// the next position (BOS first) and returns the scores for the token after it. Cloning copies
/// the self-attention and Engram caches (they grow with the position) and shares the rest.
#[derive(Clone)]
pub struct DecoderState<'w> {
    weights: &'w Weights,
    rope: Arc<Rope>,
    cross: Arc<Vec<CrossMemory>>,
    layers: Vec<KvCache>,
    engrams: Vec<EngramCache>,
    tokens: Vec<u32>,
    max_positions: usize,
}

impl<'w> DecoderState<'w> {
    /// A decode against `memory` of at most `max_positions` tokens (1 to the archive's
    /// `max_seq_len`; the RoPE table covers exactly these). Builds every layer's cross-attention
    /// keys and values. Errors are [`super::DecoderContext::new`]'s.
    pub fn new(
        weights: &'w Weights,
        memory: &DecoderMemory,
        max_positions: usize,
    ) -> Result<DecoderState<'w>, Error> {
        check_tokens(weights, max_positions)?;
        let g = &weights.geometry;
        Ok(DecoderState {
            weights,
            rope: Arc::new(Rope::new(
                g.qk_head_dim as usize,
                max_positions,
                g.rope_theta,
            )),
            cross: Arc::new(cross_memories(weights, memory)?),
            layers: weights.decoder.iter().map(|_| KvCache::new()).collect(),
            engrams: weights
                .engrams
                .iter()
                .map(|_| EngramCache { values: Vec::new() })
                .collect(),
            tokens: Vec::with_capacity(max_positions),
            max_positions,
        })
    }

    /// The tokens fed so far.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The token count this decode can hold.
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Feeds `token` at position `tokens().len()` and returns the scores for the next token
    /// over the whole vocabulary (`Decoder::forward`'s row for this position). A token outside
    /// the vocabulary, or a step past `max_positions`, is [`Error::Input`] and leaves the state
    /// as it was. Any other error comes from inside the stack, part way through updating the
    /// caches, and leaves the state unusable: start a new one.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        let w = self.weights;
        let pos = self.tokens.len();
        if pos >= self.max_positions {
            return Err(Error::Input(format!(
                "decoder step {} past the decode's {} positions",
                pos + 1,
                self.max_positions
            )));
        }
        let mut lanes = input_lanes(w, &[token])?;
        self.tokens.push(token);

        // This position's Engram keys and convolved values, `[sites, 1, d_model]`.
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for (e, cache) in w.engrams.iter().zip(&mut self.engrams) {
            let (k, v) = e.project_rows(&self.tokens, pos)?;
            cache.values.extend_from_slice(&v);
            keys.extend_from_slice(&k);
            values.extend_from_slice(&e.value_conv(&cache.values, pos)?);
        }

        let kv_bits = w.geometry.kv_bits;
        for (l, layer) in w.decoder.iter().enumerate() {
            let mem = &self.cross[l];
            let cache = &mut self.layers[l];
            let rope = &self.rope;
            let engram = BlockEngram {
                keys: &keys,
                values: &values,
                flags: site_flags(w, l),
            };
            lanes = w.decoder_mhc.step(l, &lanes, |u| {
                layer.block_with(u, w, &engram, mem, |normed| {
                    layer.self_attn.forward_step(normed, cache, rope, kv_bits)
                })
            })?;
        }
        let decoder = Decoder::new(w);
        let logits = decoder.logits(&decoder.finish(&lanes)?)?;
        Ok(logits.data().to_vec())
    }
}
