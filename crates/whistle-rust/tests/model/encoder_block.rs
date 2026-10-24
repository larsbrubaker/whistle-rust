// The encoder block's sub-modules against en_samantha's committed internals for encoder layers
// 0 and 7, each from its own fixture input (CLAUDE.md "Tolerance policy"): the attention
// projections, the q/k head norms, the whole attention module, the conv module's three steps,
// and the whole block from `block_in` to `block/0`. The head norms and the conv module's middle
// (GLU, valid mask, depthwise conv, norm) have no A8 step, so they are exact stages; the rest
// contain A8 and their bounds are set from the error measured when they landed.

use whistle::model::{EncoderContext, EncoderLayer};

use crate::shared::weights;
use crate::support::{relative_l2, Fixture};

const EXACT: f64 = 1e-6;

/// Q, K, V and gate projections from `ZCRMSNorm_0/0` (A8 then a dense layer). Measured
/// (2026-10-24): 2.2e-7 to 3.3e-7 on both layers, f32 rounding with no flipped A8 step, so the
/// bound admits no flip either.
const PROJ_BOUND: f64 = 1e-6;
/// The attention module, `ZCRMSNorm_0/0` to `self_attn/0`. Measured (2026-10-24): layer 7
/// 5.4e-7; layer 0 4.43e-4, all of it in query frame 19 (3.0e-3 there, every other frame below
/// 1e-5): one flipped A8 step on that frame's q or gated output, since a K or V flip would
/// reach every frame. That draw was lucky: the reference itself, fed this input with 3e-7
/// noise, reached 1.6e-3 on layer 0 when a K/V step flipped (review measurement, 2026-10-24),
/// so the bound is 2e-3 (raised from 1e-3 with the orchestrator's sign-off).
const ATTN_BOUND: f64 = 2e-3;
/// `pw1` from `conv_norm/0` (A8 then a dense layer). Measured (2026-10-24): 2.9e-7 and 3.1e-7,
/// no flipped step.
const PW1_BOUND: f64 = 1e-6;
/// `pw2` from `conv_out_norm/0` (SiLU, A8, a dense layer). Measured (2026-10-24): 2.8e-7 and
/// 3.0e-7, no flipped step.
const PW2_BOUND: f64 = 1e-6;
/// The whole block, `block_in` to `block/0`. Measured (2026-10-24): layer 7 4.2e-7; layer 0
/// 2.60e-3, spread over every frame. Traced: the port's attention input is 3.5e-7 from
/// `ZCRMSNorm_0/0`, enough to flip a few A8 steps of it, so `k_proj` and `v_proj` land 1e-4
/// off and every query frame sees the moved keys; the attention module itself is 1.9e-3 off from
/// that input and `post_attn_norm` of the fixture's attention is exact (1.8e-7). The reference
/// moves as much: its layer-0 block differs by 2.8e-3 between `jax.jit` and eager, and by
/// 2.6e-3 to 2.8e-3 under 3e-7 input noise (review measurement, 2026-10-24).
const BLOCK_BOUND: f64 = 5e-3;

const LAYERS: [usize; 2] = [0, 7];

fn context(f: &Fixture) -> EncoderContext<'static> {
    let (shape, _) = f.f32("encoder/layer00/block_in");
    let valid = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
    EncoderContext::new(weights(), shape[0], valid).unwrap()
}

fn check(what: &str, got: &[f32], want: &[f32], bound: f64) -> f64 {
    let err = relative_l2(got, want);
    println!("{what}: relative L2 {err:.3e}");
    assert!(err <= bound, "{what}: relative L2 {err:.3e} > {bound:.1e}");
    err
}

fn layer(l: usize) -> (&'static EncoderLayer, String) {
    (&weights().encoder[l], format!("encoder/layer{l:02}/block"))
}

#[test]
fn attention_projections_match_from_the_normed_input() {
    let f = Fixture::load("en_samantha");
    for l in LAYERS {
        let (layer, p) = layer(l);
        let (_, x) = f.f32(&format!("{p}/ZCRMSNorm_0/0"));
        let got = layer.self_attn.projections(&x);
        for (name, data) in [
            ("q_proj", &got.q),
            ("k_proj", &got.k),
            ("v_proj", &got.v),
            ("gate_proj", &got.gate),
        ] {
            let (_, want) = f.f32(&format!("{p}/self_attn/{name}/0"));
            check(&format!("{p} {name}"), data, &want, PROJ_BOUND);
        }
    }
}

#[test]
fn query_and_key_head_norms_are_exact() {
    let f = Fixture::load("en_samantha");
    for l in LAYERS {
        let (layer, p) = layer(l);
        let (_, q) = f.f32(&format!("{p}/self_attn/q_proj/0"));
        let (_, want) = f.f32(&format!("{p}/self_attn/q_norm/0"));
        check(
            &format!("{p} q_norm"),
            &layer.self_attn.query_heads(&q),
            &want,
            EXACT,
        );
        let (_, k) = f.f32(&format!("{p}/self_attn/k_proj/0"));
        let (_, want) = f.f32(&format!("{p}/self_attn/k_norm/0"));
        check(
            &format!("{p} k_norm"),
            &layer.self_attn.key_heads(&k),
            &want,
            EXACT,
        );
    }
}

#[test]
fn attention_module_matches_from_the_normed_input() {
    let f = Fixture::load("en_samantha");
    let ctx = context(&f);
    for l in LAYERS {
        let (layer, p) = layer(l);
        let (_, x) = f.f32(&format!("{p}/ZCRMSNorm_0/0"));
        let (_, want) = f.f32(&format!("{p}/self_attn/0"));
        let got = layer.self_attention(&x, &ctx).unwrap();
        check(&format!("{p} self_attn"), &got, &want, ATTN_BOUND);
    }
}

#[test]
fn conv_module_steps_match_from_their_inputs() {
    let f = Fixture::load("en_samantha");
    let ctx = context(&f);
    for l in LAYERS {
        let (layer, p) = layer(l);
        let (_, normed) = f.f32(&format!("{p}/conv_norm/0"));
        let (_, pw1) = f.f32(&format!("{p}/pw1/0"));
        check(
            &format!("{p} pw1"),
            &layer.conv_in(&normed),
            &pw1,
            PW1_BOUND,
        );
        let (_, mid) = f.f32(&format!("{p}/conv_out_norm/0"));
        let got = layer.conv_mix(&pw1, &ctx).unwrap();
        check(&format!("{p} conv_out_norm"), &got, &mid, EXACT);
        let (_, pw2) = f.f32(&format!("{p}/pw2/0"));
        check(&format!("{p} pw2"), &layer.conv_out(&mid), &pw2, PW2_BOUND);
    }
}

#[test]
fn whole_block_matches_from_its_input() {
    let f = Fixture::load("en_samantha");
    let ctx = context(&f);
    for l in LAYERS {
        let (layer, p) = layer(l);
        let (_, x) = f.f32(&format!("encoder/layer{l:02}/block_in"));
        let (_, want) = f.f32(&format!("{p}/0"));
        let got = layer.block(&x, &ctx).unwrap();
        check(&p, &got, &want, BLOCK_BOUND);
    }
}

#[test]
fn inconsistent_block_inputs_are_input_errors() {
    let ctx = EncoderContext::new(weights(), 4, 3).unwrap();
    let layer = &weights().encoder[0];
    for result in [
        layer.block(&[0.0; 512 * 3], &ctx),
        layer.self_attention(&[0.0; 512 * 5], &ctx),
        layer.conv_mix(&[0.0; 512 * 4], &ctx),
    ] {
        assert!(
            matches!(result, Err(whistle::Error::Input(_))),
            "{result:?}"
        );
    }
    assert!(matches!(
        EncoderContext::new(weights(), 4, 5),
        Err(whistle::Error::Input(_))
    ));
}

#[test]
fn an_encoder_context_rejects_an_unported_kv_width() {
    // Binding rejects such archives; `Weights` fields are public, so the encoder checks again.
    let mut w = weights().clone();
    w.geometry.kv_bits = 4;
    assert!(matches!(
        EncoderContext::new(&w, 4, 3),
        Err(whistle::Error::Archive(_))
    ));
}
