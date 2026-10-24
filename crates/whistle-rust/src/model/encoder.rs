//! The encoder (`whistle.EncoderBlock` in a `LaneStack`, run by `Whistle.encode`): features
//! through the stem (`stem`), the lanes, eight blocks each mixed in by the mHC step (`mhc`), and
//! the final lane mean and norm, giving the speech embedding [`EncoderOutput`]. Each block is a
//! half-step Hadamard MLP (`hadamard`), gated self-attention (`attention`), the conv module, and
//! a second half-step Hadamard MLP. The sub-steps are public so each can be tested on its own.

use crate::audio::Features;
use crate::error::Error;

use super::mhc::lane_mean_norm;
use super::ops::{a8, glu, silu, Rope};
use super::stem::Lanes;
use super::weights::{EncoderLayer, Weights};

/// What one encoder call shares across its layers: the weights, the sequence length (padded to
/// even), the valid frames and the RoPE table for positions `0..frames`.
pub struct EncoderContext<'w> {
    weights: &'w Weights,
    rope: Rope,
    frames: usize,
    valid_frames: usize,
}

impl<'w> EncoderContext<'w> {
    /// A context for `frames` encoder frames of which the first `valid_frames` are real. More
    /// valid frames than frames is [`Error::Input`]; a KV width other than 0 or 8 is
    /// [`Error::Archive`] (only the A8 KV path is ported; [`Weights::from_archive`] rejects
    /// such archives too).
    pub fn new(
        weights: &'w Weights,
        frames: usize,
        valid_frames: usize,
    ) -> Result<EncoderContext<'w>, Error> {
        if valid_frames > frames {
            return Err(Error::Input(format!(
                "{valid_frames} valid frames of {frames}"
            )));
        }
        // Binding already rejects these; `Weights` fields are public, so check again here.
        super::bind::check_kv_bits(weights.geometry.kv_bits)?;
        let qk_dim = weights.geometry.qk_head_dim as usize;
        Ok(EncoderContext {
            weights,
            rope: Rope::new(qk_dim, frames, weights.geometry.rope_theta),
            frames,
            valid_frames,
        })
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn valid_frames(&self) -> usize {
        self.valid_frames
    }

    fn check(&self, x: &[f32], width: usize, what: &str) -> Result<(), Error> {
        if self.frames.checked_mul(width) != Some(x.len()) {
            return Err(Error::Input(format!(
                "{what} holds {} values, not {} frames of {width}",
                x.len(),
                self.frames
            )));
        }
        Ok(())
    }
}

/// The encoder's output, the speech embedding: `frames × width` row-major (`width` is
/// `d_model`, 512), of which the first `valid_frames` are real; `frames` includes the pad to an
/// even count.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderOutput {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
}

impl EncoderLayer {
    fn width(&self) -> usize {
        self.attn_norm.scale.len()
    }

    /// The self-attention module on its normed input (`ZCRMSNorm_0`'s output, `[T, d_model]`),
    /// returning `out_proj`'s output.
    pub fn self_attention(&self, x: &[f32], ctx: &EncoderContext<'_>) -> Result<Vec<f32>, Error> {
        ctx.check(x, self.width(), "the attention input")?;
        Ok(self.self_attn.forward_bidirectional(
            x,
            &ctx.rope,
            ctx.valid_frames,
            ctx.weights.geometry.kv_bits,
        ))
    }

    /// The conv module's first step on `conv_norm`'s output: A8, then `pw1` (`[T, 2·d_model]`).
    pub fn conv_in(&self, normed: &[f32]) -> Vec<f32> {
        let mut x = normed.to_vec();
        a8(&mut x, self.pw1.cols);
        self.pw1.project(&x)
    }

    /// The conv module's middle on `pw1`'s output: GLU (the first half gated by the sigmoid of
    /// the second), frames past the valid ones zeroed, the depthwise conv over time ("SAME":
    /// `(k − 1) / 2` zero frames each side, cross-correlation), then `conv_out_norm`.
    pub fn conv_mix(&self, pw1_out: &[f32], ctx: &EncoderContext<'_>) -> Result<Vec<f32>, Error> {
        let d = self.width();
        ctx.check(pw1_out, 2 * d, "the pw1 output")?;
        let mut h = glu(pw1_out, d);
        for v in &mut h[ctx.valid_frames * d..] {
            *v = 0.0;
        }
        let k = self.dw.rows;
        let left = k.saturating_sub(1) / 2;
        let frames = ctx.frames;
        let dw = self.dw.data();
        let mut out = vec![0.0f32; h.len()];
        for t in 0..frames {
            let o = &mut out[t * d..(t + 1) * d];
            for tap in 0..k {
                let Some(src) = (t + tap).checked_sub(left).filter(|&s| s < frames) else {
                    continue;
                };
                let w = &dw[tap * d..(tap + 1) * d];
                let hs = &h[src * d..(src + 1) * d];
                for ((o, &w), &v) in o.iter_mut().zip(w).zip(hs) {
                    *o += v * w;
                }
            }
        }
        Ok(self.conv_out_norm.forward(&out))
    }

    /// The conv module's last step on `conv_out_norm`'s output: SiLU, A8, then `pw2`.
    pub fn conv_out(&self, normed: &[f32]) -> Vec<f32> {
        let mut x: Vec<f32> = normed.iter().map(|&v| silu(v)).collect();
        a8(&mut x, self.pw2.cols);
        self.pw2.project(&x)
    }

    /// The whole block on its input `x` (`[T, d_model]`, the lane mix `u`):
    /// `x1 = x + ½·mlp₀(norm(x))`, `x2 = x1 + σ(attn_gate)·norm(attn(norm(x1)))`,
    /// `x3 = x2 + conv(x2)`, `out = x3 + ½·mlp(norm(x3))`.
    pub fn block(&self, x: &[f32], ctx: &EncoderContext<'_>) -> Result<Vec<f32>, Error> {
        ctx.check(x, self.width(), "the block input")?;
        let perms = &ctx.weights.hada_perms;
        let m = self
            .hadamard_mlp_0
            .forward(&self.pre_hada_norm_0.forward(x), perms);
        let x: Vec<f32> = x.iter().zip(&m).map(|(x, m)| x + 0.5 * m).collect();

        let attn = self.self_attention(&self.attn_norm.forward(&x), ctx)?;
        let h = self.post_attn_norm.forward(&attn);
        let gate = super::ops::sigmoid(self.attn_gate);
        let x: Vec<f32> = x.iter().zip(&h).map(|(x, h)| x + gate * h).collect();

        let mid = self.conv_mix(&self.conv_in(&self.conv_norm.forward(&x)), ctx)?;
        let c = self.conv_out(&mid);
        let x: Vec<f32> = x.iter().zip(&c).map(|(x, c)| x + c).collect();

        let m = self
            .hadamard_mlp
            .forward(&self.pre_hada_norm.forward(&x), perms);
        Ok(x.iter().zip(&m).map(|(x, m)| x + 0.5 * m).collect())
    }

    /// Encoder layer `layer` (this layer's weights) on the lanes before it: the mHC step around
    /// [`EncoderLayer::block`], giving the lanes after it.
    pub fn forward(
        &self,
        lanes: &Lanes,
        layer: usize,
        ctx: &EncoderContext<'_>,
    ) -> Result<Lanes, Error> {
        if lanes.frames != ctx.frames || lanes.valid_frames != ctx.valid_frames {
            return Err(Error::Input(format!(
                "lanes of {} frames ({} valid) in a context of {} ({} valid)",
                lanes.frames, lanes.valid_frames, ctx.frames, ctx.valid_frames
            )));
        }
        ctx.weights
            .encoder_mhc
            .step(layer, lanes, |u| self.block(u, ctx))
    }
}

/// The encoder over the bound weights: [`Encoder::forward`] turns features into the speech
/// embedding.
pub struct Encoder<'w> {
    weights: &'w Weights,
}

impl<'w> Encoder<'w> {
    pub fn new(weights: &'w Weights) -> Encoder<'w> {
        Encoder { weights }
    }

    /// `Whistle.encode` from features: the stem, the frames padded to even and broadcast to the
    /// lanes, every encoder layer, then [`Encoder::finish`]. Inconsistent features are
    /// [`Error::Input`].
    pub fn forward(&self, features: &Features) -> Result<EncoderOutput, Error> {
        self.forward_with(features, &mut || false)
    }

    /// [`Encoder::forward`] asking `cancelled` before each encoder layer (after the stem); when
    /// it answers true the pass stops with [`Error::Cancelled`], so a caller can abandon a long
    /// clip within one layer's time.
    pub fn forward_with(
        &self,
        features: &Features,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<EncoderOutput, Error> {
        let w = self.weights;
        let lanes = w.stem.forward(features)?.to_lanes(w.encoder_mhc.lanes);
        let ctx = EncoderContext::new(w, lanes.frames, lanes.valid_frames)?;
        let mut lanes = lanes;
        for (l, layer) in w.encoder.iter().enumerate() {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            lanes = layer.forward(&lanes, l, &ctx)?;
        }
        self.finish(&lanes)
    }

    /// The stack's end on the last layer's lanes: the mean over the lanes, then the encoder's
    /// final ZC-RMSNorm.
    pub fn finish(&self, lanes: &Lanes) -> Result<EncoderOutput, Error> {
        let width = self.weights.encoder_final_norm.scale.len();
        let expected = lanes
            .frames
            .checked_mul(lanes.lanes)
            .and_then(|v| v.checked_mul(width));
        if lanes.width != width
            || lanes.lanes != self.weights.encoder_mhc.lanes
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
        Ok(EncoderOutput {
            data: lane_mean_norm(lanes, &self.weights.encoder_final_norm),
            frames: lanes.frames,
            valid_frames: lanes.valid_frames,
        })
    }
}
