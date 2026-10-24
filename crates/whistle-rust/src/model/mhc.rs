//! The mHC lane stack's per-layer mixing (`whistle._LaneBody`) and its end (`LaneStack`: the lane
//! mean then the final ZC-RMSNorm), shared by the encoder and the decoder: each layer reads the
//! [`Lanes`], picks its block input from them, runs the block (a closure, so either stack's block
//! fits), and writes the update back through Sinkhorn-mixed lanes. Parameters are the [`Mhc`]
//! bound in `weights`.

use crate::error::Error;

use super::ops::{a8, rms_unit, sigmoid, sinkhorn, zc_rms_norm};
use super::stem::Lanes;
use super::weights::{Mhc, Norm};

/// `nx · phi[row]ᵀ` for each of `count` rows starting at `first`, per token.
fn mix_logits(nx: &[f32], phi: &[f32], width: usize, first: usize, count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(nx.len() / width * count);
    for row in nx.chunks_exact(width) {
        for r in first..first + count {
            let p = &phi[r * width..(r + 1) * width];
            out.push(row.iter().zip(p).map(|(a, b)| a * b).sum());
        }
    }
    out
}

impl Mhc {
    /// The stack's layer count.
    pub fn layers(&self) -> usize {
        self.a_pre.len()
    }

    /// One `_LaneBody` step for `layer` over `lanes` (`[T, n, C]`). `block` maps the block input
    /// `u` (`[T, C]`) to the block's output; the update `block(u) − u` is what the lanes take.
    /// Layer `l`'s offsets favour lane `l mod n`: `pre_off = 8·onehot − 4`, `post_off =
    /// −4·(1 − onehot)`.
    /// Errors: a layer past the stack, lanes whose shape does not fit it, or a block output of
    /// the wrong length are [`Error::Input`]; the block's own error is passed through.
    pub fn step(
        &self,
        layer: usize,
        lanes: &Lanes,
        block: impl FnOnce(&[f32]) -> Result<Vec<f32>, Error>,
    ) -> Result<Lanes, Error> {
        let n = self.lanes;
        let (frames, c) = (lanes.frames, lanes.width);
        let nc = n * c;
        if layer >= self.layers() {
            return Err(Error::Input(format!(
                "layer {layer} of a {}-layer lane stack",
                self.layers()
            )));
        }
        if lanes.lanes != n || frames.checked_mul(nc) != Some(lanes.data.len()) {
            return Err(Error::Input(format!(
                "lanes hold {} values as {frames} frames of {} lanes × {c}, not {n} lanes",
                lanes.data.len(),
                lanes.lanes
            )));
        }
        if self.phi_pre.cols != nc {
            return Err(Error::Input(format!(
                "lanes are {nc} wide per frame; the stack mixes {}",
                self.phi_pre.cols
            )));
        }
        let x = &lanes.data;
        let mut nx = rms_unit(x, nc);
        a8(&mut nx, nc);
        let hot = layer % n;
        let (a_pre, a_post, a_res) = (self.a_pre[layer], self.a_post[layer], self.a_res[layer]);
        let b_pre = &self.b_pre[layer * n..(layer + 1) * n];
        let b_post = &self.b_post[layer * n..(layer + 1) * n];
        let b_res = &self.b_res[layer * n * n..(layer + 1) * n * n];

        let pre = mix_logits(&nx, &self.phi_pre.data, nc, layer * n, n);
        let mut u = vec![0.0f32; frames * c];
        for t in 0..frames {
            let ut = &mut u[t * c..(t + 1) * c];
            for i in 0..n {
                let off = if i == hot { 4.0 } else { -4.0 };
                let w = sigmoid(a_pre * pre[t * n + i] + b_pre[i] + off);
                let lane = &x[(t * n + i) * c..(t * n + i + 1) * c];
                for (o, &v) in ut.iter_mut().zip(lane) {
                    *o += w * v;
                }
            }
        }

        let out = block(&u)?;
        if out.len() != u.len() {
            return Err(Error::Input(format!(
                "a lane block returned {} values for {} inputs",
                out.len(),
                u.len()
            )));
        }
        let y: Vec<f32> = out.iter().zip(&u).map(|(o, u)| o - u).collect();

        let post = mix_logits(&nx, &self.phi_post.data, nc, layer * n, n);
        let mut res = mix_logits(&nx, &self.phi_res.data, nc, layer * n * n, n * n);
        for m in res.chunks_exact_mut(n * n) {
            for (v, &b) in m.iter_mut().zip(b_res) {
                *v = a_res * *v + b;
            }
        }
        sinkhorn(&mut res, n);

        let mut data = vec![0.0f32; x.len()];
        for t in 0..frames {
            let hres = &res[t * n * n..(t + 1) * n * n];
            let yt = &y[t * c..(t + 1) * c];
            for i in 0..n {
                let off = if i == hot { 0.0 } else { -4.0 };
                let hpost = 2.0 * sigmoid(a_post * post[t * n + i] + b_post[i] + off);
                let o = &mut data[(t * n + i) * c..(t * n + i + 1) * c];
                for j in 0..n {
                    let w = hres[i * n + j];
                    let lane = &x[(t * n + j) * c..(t * n + j + 1) * c];
                    for (o, &v) in o.iter_mut().zip(lane) {
                        *o += w * v;
                    }
                }
                for (o, &v) in o.iter_mut().zip(yt) {
                    *o += hpost * v;
                }
            }
        }
        Ok(Lanes {
            data,
            frames,
            lanes: n,
            width: c,
            valid_frames: lanes.valid_frames,
        })
    }
}

/// The stack's end: the mean over the lanes, then `norm`, `[T, C]`.
pub(crate) fn lane_mean_norm(lanes: &Lanes, norm: &Norm) -> Vec<f32> {
    let (n, c) = (lanes.lanes, lanes.width);
    let mut mean = vec![0.0f32; lanes.frames * c];
    for (m, frame) in mean.chunks_exact_mut(c).zip(lanes.data.chunks_exact(n * c)) {
        for lane in frame.chunks_exact(c) {
            for (o, &v) in m.iter_mut().zip(lane) {
                *o += v;
            }
        }
        for o in m.iter_mut() {
            *o /= n as f32;
        }
    }
    zc_rms_norm(&mean, &norm.scale)
}
