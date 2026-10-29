//! Incremental decoding: [`DecoderState`] runs the decoder (`decoder`) one token at a time with
//! a KV cache, so a greedy decode costs one position per step instead of re-scoring its whole
//! prefix. The cross-attention keys and values are computed once (`cross_attention`); each
//! layer keeps its self-attention cache (`attention::KvCache`: the taps' raw history, the keys
//! after norm, RoPE and the KV path, the values after the KV path); each Engram site keeps its
//! projected values for the value conv (`engram`). Every step runs the same functions the full
//! pass does on the new row only (the block core, the mHC step, the final norm and the head),
//! so its logits equal [`super::Decoder::forward`]'s row for that position, bit for bit: only
//! the taps, the Engram value conv and attention read earlier positions, the first two read the
//! same cached rows in the same order, and the keys the full pass masks take a large finite
//! negative logit whose `exp` is exactly 0, adding nothing to the softmax. A state is
//! [`Clone`] so a beam search can fork a hypothesis: the RoPE table and the cross-attention
//! memory are fixed for the decode and shared (`Arc`); only the growing caches are copied.
//! [`DecoderState::step_batch`] advances several such forks by one token each in one pass: every
//! projection runs on all their new rows together, so a beam search reads each weight matrix
//! once per step rather than once per hypothesis, and each row's scores are bit-identical to
//! stepping that state alone.

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
        self.check_feed(token)?;
        let w = self.weights;
        let pos = self.tokens.len();
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

    /// The input checks of a step, made before anything changes: the next position fits the
    /// decode's `max_positions` and `token` is in the vocabulary ([`Error::Input`] otherwise).
    fn check_feed(&self, token: u32) -> Result<(), Error> {
        if self.tokens.len() >= self.max_positions {
            return Err(Error::Input(format!(
                "decoder step {} past the decode's {} positions",
                self.tokens.len() + 1,
                self.max_positions
            )));
        }
        let vocab = self.weights.embedding.rows;
        if usize::try_from(token).map_or(true, |t| t >= vocab) {
            return Err(Error::Input(format!(
                "token {token} is outside the vocabulary of {vocab}"
            )));
        }
        Ok(())
    }

    /// [`DecoderState::step`] on every state at once, `tokens[b]` fed to `states[b]`; returns
    /// each state's scores in order, bit-identical to stepping each alone. The states may be at
    /// different positions with different prefixes; they must be forks of one decode (sharing
    /// its weights and cross-attention memory), as a beam search's are, else they are stepped
    /// one at a time. The token embedding, the Engram projections, every attention and MLP
    /// projection, the mHC mixing and the head run on all `B` new rows together, so each weight
    /// matrix is read once; the Engram gather and value conv, the self-attention taps, cache and
    /// attention stay per state. A count mismatch, a token outside the vocabulary or a state at
    /// its `max_positions` is [`Error::Input`] and leaves every state as it was; any other error
    /// leaves them all unusable, as for `step`.
    pub fn step_batch(
        states: &mut [&mut DecoderState<'w>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, Error> {
        if states.len() != tokens.len() {
            return Err(Error::Input(format!(
                "{} tokens for {} decoder states",
                tokens.len(),
                states.len()
            )));
        }
        let Some(first) = states.first() else {
            return Ok(Vec::new());
        };
        let (w, rope, cross) = (first.weights, first.rope.clone(), first.cross.clone());
        let shared = states.iter().all(|s| {
            std::ptr::eq(s.weights, w)
                && Arc::ptr_eq(&s.rope, &rope)
                && Arc::ptr_eq(&s.cross, &cross)
        });
        // Every state's input checks before any state moves, on both paths below.
        for (s, &t) in states.iter().zip(tokens) {
            s.check_feed(t)?;
        }
        if states.len() == 1 || !shared {
            return states
                .iter_mut()
                .zip(tokens)
                .map(|(s, &t)| s.step(t))
                .collect();
        }
        let mut lanes = input_lanes(w, tokens)?;
        for (s, &t) in states.iter_mut().zip(tokens) {
            s.tokens.push(t);
        }

        // Every state's Engram keys and convolved values at its new position, `[sites, B,
        // d_model]`: the gather and the conv per state, the projections over all rows at once.
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for (i, e) in w.engrams.iter().enumerate() {
            let mut gathered = Vec::new();
            for s in states.iter() {
                gathered.extend_from_slice(&e.gather_rows(&s.tokens, s.tokens.len() - 1)?);
            }
            let (k, v) = e.project_gathered(gathered);
            keys.extend_from_slice(&k);
            let d = e.value_proj.rows;
            for (b, s) in states.iter_mut().enumerate() {
                let cache = &mut s.engrams[i];
                cache.values.extend_from_slice(&v[b * d..(b + 1) * d]);
                values.extend_from_slice(&e.value_conv(&cache.values, s.tokens.len() - 1)?);
            }
        }

        let kv_bits = w.geometry.kv_bits;
        for (l, layer) in w.decoder.iter().enumerate() {
            let mem = &cross[l];
            let mut caches: Vec<&mut KvCache> =
                states.iter_mut().map(|s| &mut s.layers[l]).collect();
            let engram = BlockEngram {
                keys: &keys,
                values: &values,
                flags: site_flags(w, l),
            };
            lanes = w.decoder_mhc.step(l, &lanes, |u| {
                layer.block_with(u, w, &engram, mem, |normed| {
                    layer
                        .self_attn
                        .forward_step_batch(normed, &mut caches, &rope, kv_bits)
                })
            })?;
        }
        let decoder = Decoder::new(w);
        let logits = decoder.logits(&decoder.finish(&lanes)?)?;
        Ok(logits
            .data()
            .chunks_exact(logits.vocab().max(1))
            .map(<[f32]>::to_vec)
            .collect())
    }
}
