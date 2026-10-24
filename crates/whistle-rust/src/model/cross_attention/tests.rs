// Unit tests for the cross-attention's probability capture (`CrossAttention::attention` with a
// `ProbSink`): on decoder layer 0 of models/whistle.cact against a synthetic memory with masked
// frames, the captured probabilities rebuild the attention output before the gate (Σ p·v over
// the valid frames, in attend's order) bit for bit, and capturing leaves the output unchanged.

use std::path::Path;

use super::*;
use crate::cact::Archive;
use crate::model::weights::{Matrix, Weights};

/// A deterministic spread of values in [−1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

#[test]
fn captured_probabilities_rebuild_the_attention_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the workspace root");
    let bytes = std::fs::read(root.join("models/whistle.cact")).expect("whistle.cact");
    let w = Weights::from_archive(&Archive::parse(&bytes).unwrap()).unwrap();
    let cross = &w.decoder[0].cross_attn;
    let d = cross.q_proj.cols;
    let (frames, valid, rows) = (20, 15, 3);
    let memory = DecoderMemory {
        memory: Matrix::new(frames, d, values(frames * d, 1)).unwrap(),
        valid_frames: valid,
    };
    let mem = cross.memory_kv(&memory, w.geometry.kv_bits).unwrap();
    let x = values(rows * d, 2);
    let heads: Vec<usize> = (0..cross.head_count()).collect();
    let mut sink = ProbSink {
        heads: heads.clone(),
        keep: valid,
        data: Vec::new(),
    };
    let (out, gate) = cross.attention(&x, &mem, Some(&mut sink)).unwrap();
    let (plain, plain_gate) = cross.attention(&x, &mem, None).unwrap();
    assert!(out
        .iter()
        .zip(&plain)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(gate, plain_gate);

    let v_dim = cross.value_dim();
    let width = heads.len() * v_dim;
    assert_eq!(sink.data.len(), heads.len() * rows * valid);
    for &h in &heads {
        let values = &mem.v[h * frames * v_dim..(h + 1) * frames * v_dim];
        for t in 0..rows {
            let p = &sink.data[(h * rows + t) * valid..(h * rows + t + 1) * valid];
            let mut o = vec![0.0f32; v_dim];
            for (&p, vr) in p.iter().zip(values.chunks_exact(v_dim)) {
                for (o, &x) in o.iter_mut().zip(vr) {
                    *o += p * x;
                }
            }
            let want = &out[t * width + h * v_dim..t * width + (h + 1) * v_dim];
            assert!(
                o.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "head {h} row {t}"
            );
        }
    }
}
