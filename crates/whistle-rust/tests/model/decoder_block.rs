// The decoder block's sub-modules on en_samantha's committed decoder layer 0 and layer 3 (the
// first Engram site) internals, each from its own fixture input: causal self-attention (the
// projections, the 3-tap QKV conv then the per-head norms, the module output), cross-attention
// to the `decoder/memory` fixture (projections, norms, output), and the whole block from
// `block_in` to `block/0`. Quantized stages (CLAUDE.md "Tolerance policy"): each bound is set
// from the error measured when the decoder landed, written beside it.

use whistle::model::{engram_add, DecoderContext, DecoderMemory, Matrix, Projections};

use crate::shared::weights;
use crate::support::{relative_l2, Fixture};

/// The committed `decoder/memory` for a clip, with its valid frames, so each decoder stage
/// starts from the reference's own memory.
pub fn memory(f: &Fixture) -> DecoderMemory {
    let (shape, data) = f.f32("decoder/memory");
    let valid = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
    DecoderMemory {
        memory: Matrix::new(shape[0], shape[1], data).unwrap(),
        valid_frames: valid,
    }
}

fn check(what: &str, got: &[f32], want: &[f32], bound: f64, worst: &mut (f64, String)) {
    let err = relative_l2(got, want);
    println!("{what}: relative L2 {err:.3e}");
    assert!(err <= bound, "{what}: relative L2 {err:.3e} > {bound:.1e}");
    if err > worst.0 {
        *worst = (err, what.to_string());
    }
}

/// Self-attention, layers 0 and 3, from `ZCRMSNorm_0/0`. Measured (2026-10-24): projections
/// 1.8e-7 to 3.3e-7 (worst gate_proj, layer 0); taps then q/k norm 1.2e-7 to 2.2e-7; the module
/// output 4.73e-7 (layer 0) and 4.33e-7 (layer 3). f32 rounding with no flipped A8 step, so the
/// bounds admit none.
const SELF_PROJ_BOUND: f64 = 1e-6;
const SELF_NORM_BOUND: f64 = 1e-6;
const SELF_OUT_BOUND: f64 = 1e-6;

#[test]
fn decoder_self_attention_reproduces_layers_0_and_3() {
    let f = Fixture::load("en_samantha");
    let tokens = f.tokens();
    let mem = memory(&f);
    let ctx = DecoderContext::new(weights(), &tokens, &mem).unwrap();
    let mut worst = (0.0, String::new());
    for l in [0usize, 3] {
        let p = format!("decoder/layer{l:02}/block");
        let layer = &weights().decoder[l];
        let (_, x) = f.f32(&format!("{p}/ZCRMSNorm_0/0"));
        let proj = layer.self_attn.projections(&x);
        for (name, got) in [
            ("q_proj", &proj.q),
            ("k_proj", &proj.k),
            ("v_proj", &proj.v),
            ("gate_proj", &proj.gate),
        ] {
            let (_, want) = f.f32(&format!("{p}/self_attn/{name}/0"));
            check(
                &format!("{p} self_attn {name}"),
                got,
                &want,
                SELF_PROJ_BOUND,
                &mut worst,
            );
        }
        let tapped = layer.self_attn.causal_taps(&proj, 0, 0).unwrap();
        let (_, want) = f.f32(&format!("{p}/self_attn/q_norm/0"));
        let got = layer.self_attn.query_heads(&tapped.q);
        check(
            &format!("{p} self_attn taps + q_norm"),
            &got,
            &want,
            SELF_NORM_BOUND,
            &mut worst,
        );
        let (_, want) = f.f32(&format!("{p}/self_attn/k_norm/0"));
        let got = layer.self_attn.key_heads(&tapped.k);
        check(
            &format!("{p} self_attn taps + k_norm"),
            &got,
            &want,
            SELF_NORM_BOUND,
            &mut worst,
        );
        let (_, want) = f.f32(&format!("{p}/self_attn/0"));
        let got = layer.self_attention(&x, &ctx).unwrap();
        check(
            &format!("{p} self_attn"),
            &got,
            &want,
            SELF_OUT_BOUND,
            &mut worst,
        );
    }
    println!("worst {:.3e} ({})", worst.0, worst.1);
}

#[test]
fn qkv_taps_with_history_equal_the_tail_of_the_full_sequence() {
    let f = Fixture::load("en_samantha");
    let attn = &weights().decoder[0].self_attn;
    let (shape, x) = f.f32("decoder/layer00/block/ZCRMSNorm_0/0");
    let full = attn.projections(&x);
    let all = attn.causal_taps(&full, 0, 0).unwrap();
    let t = shape[0];
    // A cached step at position p keeps the raw projections of the two positions before it.
    let rows = |z: &[f32], from: usize, to: usize| -> Vec<f32> {
        let w = z.len() / t;
        z[from * w..to * w].to_vec()
    };
    for pos in [0usize, 1, 2, 5, t - 1] {
        let from = pos.saturating_sub(2);
        let part = Projections {
            q: rows(&full.q, from, pos + 1),
            k: rows(&full.k, from, pos + 1),
            v: rows(&full.v, from, pos + 1),
            gate: rows(&full.gate, from, pos + 1),
        };
        let step = attn.causal_taps(&part, pos - from, pos).unwrap();
        assert_eq!(step.q, rows(&all.q, pos, pos + 1), "q at {pos}");
        assert_eq!(step.k, rows(&all.k, pos, pos + 1), "k at {pos}");
        assert_eq!(step.v, rows(&all.v, pos, pos + 1), "v at {pos}");
        assert_eq!(step.gate, rows(&all.gate, pos, pos + 1), "gate at {pos}");
    }
    // A history that is not exactly the min(pos, 2) positions before is an input error.
    let input_err = |r: Result<Projections, whistle::Error>| {
        assert!(matches!(r, Err(whistle::Error::Input(_))), "{r:?}");
    };
    input_err(attn.causal_taps(&full, 0, 3));
    input_err(attn.causal_taps(&full, 1, 5));
    input_err(attn.causal_taps(&full, 2, 1));
    input_err(attn.causal_taps(&full, t + 1, t + 1));
    assert!(matches!(
        weights().encoder[0].self_attn.causal_taps(&full, 0, 0),
        Err(whistle::Error::Archive(_))
    ));
}

/// Cross-attention, layers 0 and 3, from `cross_norm/0` and the memory fixture. Measured
/// (2026-10-24): projections 2.5e-7 to 3.5e-7; q/k norm 3.1e-7 to 3.55e-7; the module output
/// 6.33e-7 (layer 0) and 4.28e-7 (layer 3). No flipped A8 step, so the bounds admit none.
const CROSS_PROJ_BOUND: f64 = 1e-6;
const CROSS_NORM_BOUND: f64 = 1e-6;
const CROSS_OUT_BOUND: f64 = 1e-6;

#[test]
fn decoder_cross_attention_reproduces_layers_0_and_3() {
    let f = Fixture::load("en_samantha");
    let tokens = f.tokens();
    let mem = memory(&f);
    let ctx = DecoderContext::new(weights(), &tokens, &mem).unwrap();
    let mut worst = (0.0, String::new());
    for l in [0usize, 3] {
        let p = format!("decoder/layer{l:02}/block");
        let layer = &weights().decoder[l];
        let (_, x) = f.f32(&format!("{p}/cross_norm/0"));
        let proj = layer.cross_attn.projections(&x, mem.memory.data()).unwrap();
        for (name, got) in [
            ("q_proj", &proj.q),
            ("k_proj", &proj.k),
            ("v_proj", &proj.v),
            ("gate_proj", &proj.gate),
        ] {
            let (_, want) = f.f32(&format!("{p}/cross_attn/{name}/0"));
            check(
                &format!("{p} cross_attn {name}"),
                got,
                &want,
                CROSS_PROJ_BOUND,
                &mut worst,
            );
        }
        let (_, want) = f.f32(&format!("{p}/cross_attn/q_norm/0"));
        let got = layer.cross_attn.query_heads(&proj.q);
        check(
            &format!("{p} cross_attn q_norm"),
            &got,
            &want,
            CROSS_NORM_BOUND,
            &mut worst,
        );
        let (_, want) = f.f32(&format!("{p}/cross_attn/k_norm/0"));
        let got = layer.cross_attn.key_heads(&proj.k);
        check(
            &format!("{p} cross_attn k_norm"),
            &got,
            &want,
            CROSS_NORM_BOUND,
            &mut worst,
        );
        let (_, want) = f.f32(&format!("{p}/cross_attn/0"));
        let got = layer.cross_attention(&x, l, &ctx).unwrap();
        check(
            &format!("{p} cross_attn"),
            &got,
            &want,
            CROSS_OUT_BOUND,
            &mut worst,
        );
    }
    println!("worst {:.3e} ({})", worst.0, worst.1);
}

/// The whole block from `block_in` to `block/0`, layers 0 and 3 (with the Engram add, its keys
/// and values computed from the tokens as the decoder does). Measured (2026-10-24): layer 0
/// 3.32e-7; layer 3 3.49e-3. Traced: with the fixture's `engram_k`/`engram_v` every layer-3
/// sub-step stays within 5.2e-7 and the block output is 1.8e-7 (pinned at 1e-6 by
/// `decoder_block_3_from_the_fixture_engram_reproduces_the_reference`); the port's own Engram
/// keys and values (3.2e-7 and 3.6e-7 from the fixture, `engram`) move `norm_in`'s output by
/// 2.0e-7, which flips A8 steps in self-attention (5.7e-3 there), and that carries through the
/// block. The reference is as sensitive: ±1-ulp noise on its own `norm_in` output flips layer-3
/// self-attention to 4.7e-3 or 5.7e-3 in 4 of 6 trials (review measurement, 2026-10-24). The
/// bound is twice the measured flip.
const BLOCK_BOUND: f64 = 7e-3;

#[test]
fn decoder_block_reproduces_layers_0_and_3() {
    let f = Fixture::load("en_samantha");
    let tokens = f.tokens();
    let mem = memory(&f);
    let ctx = DecoderContext::new(weights(), &tokens, &mem).unwrap();
    let mut worst = (0.0, String::new());
    for l in [0usize, 3] {
        let p = format!("decoder/layer{l:02}");
        let (_, x) = f.f32(&format!("{p}/block_in"));
        let (_, want) = f.f32(&format!("{p}/block/0"));
        let got = weights().decoder[l].block(&x, l, &ctx).unwrap();
        check(&format!("{p} block"), &got, &want, BLOCK_BOUND, &mut worst);
    }
    println!("worst {:.3e} ({})", worst.0, worst.1);
}

/// Layer 3's block composed from its public pieces with the fixture's `engram_k`/`engram_v`
/// (the Engram add, `norm_in`, self-attention, `post_attn_norm`, the gated residual,
/// `cross_norm`, cross-attention, `post_cross_norm`, the gated residual, `pre_hada_norm` and the
/// full Hadamard step), so every input matches the reference's and a real error cannot hide
/// under `BLOCK_BOUND`'s flip. Measured (2026-10-24): 1.77e-7 (f32 rounding, no flipped step).
const BLOCK_FROM_FIXTURE_ENGRAM_BOUND: f64 = 1e-6;

#[test]
fn decoder_block_3_from_the_fixture_engram_reproduces_the_reference() {
    let w = weights();
    let f = Fixture::load("en_samantha");
    let tokens = f.tokens();
    let mem = memory(&f);
    let ctx = DecoderContext::new(w, &tokens, &mem).unwrap();
    let layer = &w.decoder[3];
    let (_, x) = f.f32("decoder/layer03/block_in");
    let (_, k) = f.f32("decoder/engram_k");
    let (_, v) = f.f32("decoder/engram_v");
    let sigmoid = |g: f32| 1.0 / (1.0 + (-g).exp());
    let x = engram_add(&x, &k, &v, &[1.0, 0.0], 512).unwrap();
    let a = layer
        .self_attention(&layer.norm_in.forward(&x), &ctx)
        .unwrap();
    let h = layer.post_attn_norm.forward(&a);
    let g = sigmoid(layer.attn_gate);
    let x: Vec<f32> = x.iter().zip(&h).map(|(x, h)| x + g * h).collect();
    let c = layer
        .cross_attention(&layer.cross_norm.forward(&x), 3, &ctx)
        .unwrap();
    let h = layer.post_cross_norm.forward(&c);
    let g = sigmoid(layer.cross_gate);
    let x: Vec<f32> = x.iter().zip(&h).map(|(x, h)| x + g * h).collect();
    let m = layer
        .hadamard_mlp
        .forward(&layer.pre_hada_norm.forward(&x), &w.hada_perms);
    let got: Vec<f32> = x.iter().zip(&m).map(|(x, m)| x + m).collect();
    let (_, want) = f.f32("decoder/layer03/block/0");
    let err = relative_l2(&got, &want);
    println!("decoder/layer03 block from the fixture Engram: relative L2 {err:.3e}");
    assert!(
        err <= BLOCK_FROM_FIXTURE_ENGRAM_BOUND,
        "relative L2 {err:.3e}"
    );
}
