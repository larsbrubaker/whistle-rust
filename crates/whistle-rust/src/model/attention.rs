//! Gated multi-head self-attention (`architecture.MultiHeadAttention` with `flash=True`, quant on)
//! as the encoder runs it: A8 on the input, the Q, K, V and gate projections, ZC-RMSNorm on q and
//! k per head, RoPE, A8 on q and on the KV path, grouped-query attention, the sigmoid gate, A8,
//! and `out_proj`. The pieces are separate so the decoder's tapped, causal variant
//! ([`SelfAttention::causal_taps`], `forward_causal`, and its cached one-position step
//! `forward_step` over a [`KvCache`]) reuses them, and so does cross-attention
//! (`cross_attention`); the weights are the [`SelfAttention`] bound in `weights`, the arithmetic
//! is `ops`.

use crate::error::Error;

use super::ops::{a8, fake_quant_rows, linear, sigmoid, softmax, zc_rms_norm, Rope};
use super::weights::{Matrix, SelfAttention};

/// The four projections of one sequence from its A8 input, each `[T, width]` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Projections {
    /// `[T, heads · qk_dim]`.
    pub q: Vec<f32>,
    /// `[T, kv_heads · qk_dim]`.
    pub k: Vec<f32>,
    /// `[T, kv_heads · v_dim]`.
    pub v: Vec<f32>,
    /// `[T, heads · v_dim]`, before the sigmoid.
    pub gate: Vec<f32>,
}

/// One decoder layer's self-attention cache for incremental decoding: the raw projections of
/// the last `n_taps − 1` positions (the taps' history), and every earlier position's key (after
/// norm, RoPE and the KV path) and value (after the KV path) per KV head, `[positions, dim]`
/// each. Filled one position at a time by [`SelfAttention::forward_step`].
#[derive(Debug, Clone)]
pub(crate) struct KvCache {
    history: Projections,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    positions: usize,
}

impl KvCache {
    pub(crate) fn new() -> KvCache {
        KvCache {
            history: Projections {
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                gate: Vec::new(),
            },
            k: Vec::new(),
            v: Vec::new(),
            positions: 0,
        }
    }
}

/// The attention's head geometry, derived from the bound weights.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Heads {
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) qk_dim: usize,
    pub(crate) v_dim: usize,
}

/// `jax.nn.dot_product_attention`'s stand-in for a masked logit (`_get_large_negative`):
/// finite, so a row with no allowed key still softmaxes to a uniform average as it does there.
const MASKED: f32 = -0.7 * f32::MAX;

/// The causal depthwise conv on one projection (`MultiHeadAttention` with `qkv_conv_taps`):
/// `Σ_j taps[j] · shift_right(z, j) · tap_ok_j` per channel, `taps` `[n_taps, width]`. `z` holds
/// `history` earlier rows (the raw projections of the positions just before, as a cached step
/// keeps them) followed by the rows to convolve; the output covers only the latter. A tap that
/// reads before row 0 contributes 0, which is `tap_ok_j` (position ≥ j) when the history starts
/// at position 0 or holds at least `n_taps − 1` rows. The sum starts at 0 and adds the taps in
/// order, as Python's `sum` does, so the f32 sums round the same way. Shapes that do not fit
/// are [`Error::Archive`] (the taps) or [`Error::Input`] (the rows).
pub(crate) fn causal_conv(z: &[f32], taps: &Matrix, history: usize) -> Result<Vec<f32>, Error> {
    let width = taps.cols;
    if width == 0 || !z.len().is_multiple_of(width) {
        return Err(Error::Archive(format!(
            "QKV taps of width {width} do not fit {} projected values",
            z.len()
        )));
    }
    let rows = z.len() / width;
    if history > rows {
        return Err(Error::Input(format!("{history} history rows of {rows}")));
    }
    let tap_values = taps.data();
    let mut out = vec![0.0f32; (rows - history) * width];
    for (i, o) in out.chunks_exact_mut(width).enumerate() {
        let r = history + i;
        for j in 0..taps.rows {
            let tap = &tap_values[j * width..(j + 1) * width];
            // Before row 0 the shifted row is padding times tap_ok: 0, which adds nothing.
            let Some(src) = r.checked_sub(j) else {
                continue;
            };
            let zs = &z[src * width..(src + 1) * width];
            for ((o, &w), &v) in o.iter_mut().zip(tap).zip(zs) {
                *o += w * v;
            }
        }
    }
    Ok(out)
}

/// `[T, heads · dim]` to `[heads, T, dim]` (the reference's `reshape` then `transpose(0, 2, 1, 3)`).
pub(crate) fn split_heads(x: &[f32], heads: usize, dim: usize) -> Vec<f32> {
    let frames = x.len() / (heads * dim).max(1);
    let mut out = vec![0.0f32; x.len()];
    for t in 0..frames {
        for h in 0..heads {
            let src = &x[(t * heads + h) * dim..(t * heads + h + 1) * dim];
            out[(h * frames + t) * dim..(h * frames + t + 1) * dim].copy_from_slice(src);
        }
    }
    out
}

impl SelfAttention {
    pub(crate) fn heads(&self) -> Heads {
        let qk_dim = self.q_norm.scale.len().max(1);
        let kv_heads = (self.k_proj.rows / qk_dim).max(1);
        Heads {
            heads: self.q_proj.rows / qk_dim,
            kv_heads,
            qk_dim,
            v_dim: self.v_proj.rows / kv_heads,
        }
    }

    /// A8 on `x` (`[T, d_model]`), then the Q, K, V and gate projections of that A8 input.
    pub fn projections(&self, x: &[f32]) -> Projections {
        let mut xq = x.to_vec();
        a8(&mut xq, self.q_proj.cols);
        Projections {
            q: self.q_proj.project(&xq),
            k: self.k_proj.project(&xq),
            v: self.v_proj.project(&xq),
            gate: self.gate_proj.project(&xq),
        }
    }

    /// The query projection split into heads, `[heads, T, qk_dim]`, and ZC-RMSNorm'd per head
    /// (`q_norm`, before RoPE).
    pub fn query_heads(&self, q: &[f32]) -> Vec<f32> {
        let h = self.heads();
        zc_rms_norm(&split_heads(q, h.heads, h.qk_dim), &self.q_norm.scale)
    }

    /// The key projection split into heads, `[kv_heads, T, qk_dim]`, and normed (`k_norm`).
    pub fn key_heads(&self, k: &[f32]) -> Vec<f32> {
        let h = self.heads();
        zc_rms_norm(&split_heads(k, h.kv_heads, h.qk_dim), &self.k_norm.scale)
    }

    /// The encoder's attention (no Q/K/V taps; the decoder's own forward applies its taps
    /// between [`SelfAttention::projections`] and the head split) over `x` (`[T, d_model]`,
    /// already normed): bidirectional, every query row attending to the keys at frames below
    /// `valid_frames`, RoPE at positions `0..T`. `kv_bits` is the archive's KV width: 8 is
    /// per-head A8, 0 none. Returns `out_proj`'s output, `[T, d_model]`.
    pub(crate) fn forward_bidirectional(
        &self,
        x: &[f32],
        rope: &Rope,
        valid_frames: usize,
        kv_bits: u32,
    ) -> Vec<f32> {
        let h = self.heads();
        let p = self.projections(x);
        let (q, k, v) = self.attention_heads(&p, rope, 0, kv_bits);
        let out = attend(&q, &k, &v, h, |_, s| s < valid_frames);
        self.gate_out(out, &p.gate)
    }

    /// The projections (`p`, tapped already in the decoder) made ready to attend, its rows at
    /// positions `start..`: the head split, `q_norm` and `k_norm`, RoPE, then A8 on q and the KV
    /// path at `kv_bits` on k and v (`maybe_quant_query`, `maybe_quant_kv`: per head and
    /// position, over the head dim). Returns q `[heads, T, qk_dim]`, k `[kv_heads, T, qk_dim]`
    /// and v `[kv_heads, T, v_dim]`. The whole-sequence paths and the cached decoder step share
    /// it, so a step computes each position exactly as the full pass does.
    fn attention_heads(
        &self,
        p: &Projections,
        rope: &Rope,
        start: usize,
        kv_bits: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.heads();
        let frames = p.q.len() / (h.heads * h.qk_dim).max(1);
        let mut q = self.query_heads(&p.q);
        let mut k = self.key_heads(&p.k);
        let mut v = split_heads(&p.v, h.kv_heads, h.v_dim);
        rotate(&mut q, rope, h.qk_dim, frames, start);
        rotate(&mut k, rope, h.qk_dim, frames, start);
        a8(&mut q, h.qk_dim);
        if kv_bits > 0 {
            fake_quant_rows(&mut k, h.qk_dim, kv_bits);
            fake_quant_rows(&mut v, h.v_dim, kv_bits);
        }
        (q, k, v)
    }

    /// The decoder's causal taps on Q, K and V (`q_taps`, `k_taps`, `v_taps`) over projections
    /// whose first `history` rows are earlier positions ([`causal_conv`]); the result covers the
    /// rows after them, the gate (untapped) included, the first at absolute position `start`.
    /// The history must be exactly the `min(start, n_taps − 1)` positions before `start`, so a
    /// cached step cannot convolve against too few or misaligned rows: another count, or more
    /// history than rows, is [`Error::Input`]. A whole sequence is `(p, 0, 0)`. Attention without
    /// taps (the encoder's) is [`Error::Archive`].
    pub fn causal_taps(
        &self,
        p: &Projections,
        history: usize,
        start: usize,
    ) -> Result<Projections, Error> {
        let taps = self
            .taps
            .as_ref()
            .ok_or_else(|| Error::Archive("decoder self-attention without QKV taps".into()))?;
        let needed = start.min(taps.q.rows.saturating_sub(1));
        if history != needed {
            return Err(Error::Input(format!(
                "{history} history rows before position {start}; the QKV taps need {needed}"
            )));
        }
        let gate_width = self.gate_proj.rows;
        let skip = history
            .checked_mul(gate_width)
            .filter(|&n| n <= p.gate.len());
        let skip = skip.ok_or_else(|| Error::Input(format!("{history} history rows")))?;
        Ok(Projections {
            q: causal_conv(&p.q, &taps.q, history)?,
            k: causal_conv(&p.k, &taps.k, history)?,
            v: causal_conv(&p.v, &taps.v, history)?,
            gate: p.gate[skip..].to_vec(),
        })
    }

    /// The decoder's causal self-attention over a whole sequence at positions `0..T` (`x`
    /// `[T, d_model]`, already normed): projections, [`SelfAttention::causal_taps`], the head
    /// split and norms, RoPE, A8 on q and the KV path at `kv_bits`, and attention where query `t`
    /// sees keys `0..=t` (`make_causal_mask`). The RoPE table must cover `T` positions. Returns
    /// `out_proj`'s output, `[T, d_model]`.
    pub(crate) fn forward_causal(
        &self,
        x: &[f32],
        rope: &Rope,
        kv_bits: u32,
    ) -> Result<Vec<f32>, Error> {
        let h = self.heads();
        let p = self.causal_taps(&self.projections(x), 0, 0)?;
        let (q, k, v) = self.attention_heads(&p, rope, 0, kv_bits);
        let out = attend(&q, &k, &v, h, |t, s| s <= t);
        Ok(self.gate_out(out, &p.gate))
    }

    /// One cached decoder step of [`SelfAttention::forward_causal`]: `x` is the normed input of
    /// the single new position `cache.positions`. Its raw projections join the cache's history
    /// for the taps, its key (after norm, RoPE and the KV path) and value join the cache, and its
    /// query attends to every cached key, which is exactly the causal row of the full pass: the
    /// keys the full pass masks contribute an exact 0 there. The RoPE table must cover the new
    /// position. Returns `out_proj`'s output for that row.
    pub(crate) fn forward_step(
        &self,
        x: &[f32],
        cache: &mut KvCache,
        rope: &Rope,
        kv_bits: u32,
    ) -> Result<Vec<f32>, Error> {
        let h = self.heads();
        if x.len() != self.q_proj.cols {
            return Err(Error::Input(format!(
                "a decoder step of {} values, not one row of {}",
                x.len(),
                self.q_proj.cols
            )));
        }
        if cache.positions >= rope.cos.len() / rope.half.max(1) {
            return Err(Error::Input(format!(
                "decoder position {} is past the cache's RoPE table",
                cache.positions
            )));
        }
        let new = self.projections(x);
        let history = cache.history.q.len() / self.q_proj.rows.max(1);
        let joined = Projections {
            q: [cache.history.q.as_slice(), &new.q].concat(),
            k: [cache.history.k.as_slice(), &new.k].concat(),
            v: [cache.history.v.as_slice(), &new.v].concat(),
            gate: [cache.history.gate.as_slice(), &new.gate].concat(),
        };
        let p = self.causal_taps(&joined, history, cache.positions)?;
        let (q, k, v) = self.attention_heads(&p, rope, cache.positions, kv_bits);
        // The taps reach back `n_taps − 1` positions; keep exactly that many raw rows.
        let keep = self.taps.as_ref().map_or(0, |t| t.q.rows.saturating_sub(1));
        let rows = (history + 1).min(keep);
        let tail = |z: &[f32], width: usize| z[z.len() - rows * width..].to_vec();
        cache.history = Projections {
            q: tail(&joined.q, self.q_proj.rows),
            k: tail(&joined.k, self.k_proj.rows),
            v: tail(&joined.v, self.v_proj.rows),
            gate: tail(&joined.gate, self.gate_proj.rows),
        };
        if cache.k.len() != h.kv_heads {
            cache.k = vec![Vec::new(); h.kv_heads];
            cache.v = vec![Vec::new(); h.kv_heads];
        }
        for g in 0..h.kv_heads {
            cache.k[g].extend_from_slice(&k[g * h.qk_dim..(g + 1) * h.qk_dim]);
            cache.v[g].extend_from_slice(&v[g * h.v_dim..(g + 1) * h.v_dim]);
        }
        cache.positions += 1;
        let (keys, values) = (cache.k.concat(), cache.v.concat());
        let out = attend(&q, &keys, &values, h, |_, _| true);
        Ok(self.gate_out(out, &p.gate))
    }

    /// `out · sigmoid(gate)`, A8, then `out_proj`.
    pub(crate) fn gate_out(&self, out: Vec<f32>, gate: &[f32]) -> Vec<f32> {
        gate_out(&self.out_proj, out, gate)
    }
}

/// `out · sigmoid(gate)`, A8, then `out_proj`: how self- and cross-attention both end.
pub(crate) fn gate_out(out_proj: &Matrix, mut out: Vec<f32>, gate: &[f32]) -> Vec<f32> {
    for (o, &g) in out.iter_mut().zip(gate) {
        *o *= sigmoid(g);
    }
    a8(&mut out, out_proj.cols);
    out_proj.project(&out)
}

/// RoPE on every head vector of `[heads, frames, dim]`: row `t` of each head at position
/// `start + t`, so a decoder step can rotate one cached position, or a sequence can use a table
/// longer than itself. The caller keeps `start + frames` within the table (`Rope::new`'s
/// `seq_len`); `Rope::apply` indexes it.
pub(crate) fn rotate(x: &mut [f32], rope: &Rope, dim: usize, frames: usize, start: usize) {
    debug_assert!(start + frames <= rope.cos.len() / rope.half.max(1));
    for (i, row) in x.chunks_exact_mut(dim).enumerate() {
        rope.apply(row, start + i % frames.max(1));
    }
}

/// Grouped-query attention as `jax.nn.dot_product_attention` computes the fused kernel's call:
/// q `[heads, Tq, qk_dim]`, k `[kv_heads, Tk, qk_dim]`, v `[kv_heads, Tk, v_dim]`. Query head
/// `h` reads KV head `h / (heads / kv_heads)` (JAX reshapes the query heads to
/// `[kv_heads, group]`). q is scaled by `sqrt(fused / qk_dim)` with `fused = max(qk_dim,
/// v_dim)`, the logits by `1 / sqrt(fused)` (the zero padding to `fused` adds nothing to a dot
/// product), keys where `allowed(t, s)` is false take [`MASKED`], and each row softmaxes in f32.
/// Returns `[Tq, heads · v_dim]`; with no keys (`Tk = 0`) there is nothing to average, so every
/// output is 0 (callers reject an empty memory before this).
pub(crate) fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    h: Heads,
    allowed: impl Fn(usize, usize) -> bool,
) -> Vec<f32> {
    attend_with(q, k, v, h, allowed, None)
}

/// Where [`attend_with`] copies the probabilities it mixes the values with: for each query head
/// in `heads` (ascending, as `attend` visits them), every query row's first `keep` keys, appended
/// to `data` as `[heads, Tq, keep]`. The word aligner reads the valid memory frames this way, so
/// the weights it aligns from are exactly the ones the model used.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProbSink {
    pub heads: Vec<usize>,
    pub keep: usize,
    pub data: Vec<f32>,
}

/// [`attend`], optionally copying each wanted head's probabilities into `sink` right after the
/// softmax. With `None` nothing else happens, so the output is bit-identical either way.
pub(crate) fn attend_with(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    h: Heads,
    allowed: impl Fn(usize, usize) -> bool,
    mut sink: Option<&mut ProbSink>,
) -> Vec<f32> {
    let fused = h.qk_dim.max(h.v_dim);
    let q_scale = (fused as f64 / h.qk_dim as f64).sqrt() as f32;
    let scale = (1.0 / (fused as f64).sqrt()) as f32;
    let tq = q.len() / (h.heads * h.qk_dim).max(1);
    let tk = k.len() / (h.kv_heads * h.qk_dim).max(1);
    let group = (h.heads / h.kv_heads.max(1)).max(1);
    let width = h.heads * h.v_dim;
    let mut out = vec![0.0f32; tq * width];
    if tk == 0 {
        return out;
    }
    let mut qs = vec![0.0f32; tq * h.qk_dim];
    for head in 0..h.heads {
        let g = head / group;
        let keys = &k[g * tk * h.qk_dim..(g + 1) * tk * h.qk_dim];
        let values = &v[g * tk * h.v_dim..(g + 1) * tk * h.v_dim];
        for (s, &x) in qs
            .iter_mut()
            .zip(&q[head * tq * h.qk_dim..(head + 1) * tq * h.qk_dim])
        {
            *s = x * q_scale;
        }
        // Every query's logit against every key at once: each is still one in-order dot
        // product (`ops::linear`), then the scale and the mask.
        let mut all = linear(&qs, keys, h.qk_dim, tk);
        let mut capture = sink.as_deref_mut().filter(|s| s.heads.contains(&head));
        for (t, logits) in all.chunks_exact_mut(tk).enumerate() {
            for (s, l) in logits.iter_mut().enumerate() {
                *l = if allowed(t, s) { *l * scale } else { MASKED };
            }
            softmax(logits, tk);
            if let Some(c) = capture.as_deref_mut() {
                c.data.extend_from_slice(&logits[..c.keep.min(tk)]);
            }
            let o = &mut out[t * width + head * h.v_dim..t * width + (head + 1) * h.v_dim];
            for (&p, vr) in logits.iter().zip(values.chunks_exact(h.v_dim)) {
                for (o, &x) in o.iter_mut().zip(vr) {
                    *o += p * x;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
