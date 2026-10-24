// Each encoder layer (block plus mHC lane step) from the previous layer's committed lanes
// (`encoder/lanes_in` for layer 0) on en_samantha and de_anna, and the whole encoder end to end
// from every clip's `features` to `encoder/out`. Both are quantized stages (CLAUDE.md
// "Tolerance policy"); the per-layer bound is set from the error measured when they landed, and
// the end-to-end bound is loose because upstream itself moves about 2% between execution modes
// (docs/design/oracle.md).

use whistle::audio::Features;
use whistle::model::{Encoder, EncoderContext, Lanes};

use crate::shared::weights;
use crate::support::{clip_names, relative_l2, Fixture};

/// One layer from the previous layer's fixture lanes, worst over layers and both clips.
/// Measured (2026-10-24): worst 2.28e-3 (en_samantha layer 0, the A8 flips traced in
/// `encoder_block`); en_samantha 2.3e-3, 1.7e-3, 3.9e-4, 1.2e-3, 1.6e-3, then layers 5-7 at
/// 1.7e-7 to 3.3e-7; de_anna 2.9e-4, 1.9e-7, 3.8e-4, 1.2e-4, 9.3e-4, 1.7e-7, 1.6e-7, 7.6e-4.
/// Layers with no flipped step match to f32 rounding.
const LAYER_BOUND: f64 = 5e-3;
/// `encoder/out` from `features`, worst clip. Measured (2026-10-24): de_anna 2.45e-2,
/// en_daniel 2.23e-2, en_samantha 2.27e-2, es_monica 1.90e-2, fr_thomas 2.28e-2, it_alice
/// 2.21e-2, nl_xander 2.11e-2, pl_zosia 2.74e-2 (worst), silence 2.14e-2; the valid frames alone
/// measure the same to within 4e-4. Per layer the port is at most 2.3e-3 from the reference, and
/// each flipped step moves every later layer's A8 inputs, so the flips compound over 8 layers.
/// The reference itself moves as much: under 3e-7 input noise its encoder output spans 1.9% to
/// 2.8% across the clips (pl_zosia 2.40% to 2.76%, and 2.41% under `jax.jit`; review
/// measurement, 2026-10-24).
const END_TO_END_BOUND: f64 = 4e-2;

fn lanes(f: &Fixture, name: &str, valid: usize) -> Lanes {
    let (shape, data) = f.f32(name);
    Lanes {
        data,
        frames: shape[0],
        lanes: shape[1],
        width: shape[2],
        valid_frames: valid,
    }
}

/// The mHC step alone (`Mhc::step` around a block that returns the fixture's `block/0`), from
/// the previous layer's lanes: the block input `u` against `block_in`, and the lanes after
/// against `layerNN/lanes`. The step has one A8 (on the unit-RMS lanes feeding the mixing
/// logits). Measured (2026-10-24): `u` 1.31e-7 (layer 0) and 1.58e-7 (layer 7), the lanes
/// 6.2e-8 and 1.32e-7: f32 rounding with no flipped step, so the bounds admit none.
const STEP_U_BOUND: f64 = 1e-6;
const STEP_OUT_BOUND: f64 = 1e-6;

#[test]
fn mhc_step_mixes_the_lanes_as_the_reference_does() {
    let f = Fixture::load("en_samantha");
    let valid = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
    for (l, before) in [(0usize, "encoder/lanes_in"), (7, "encoder/layer06/lanes")] {
        let p = format!("encoder/layer{l:02}");
        let input = lanes(&f, before, valid);
        let (_, block_out) = f.f32(&format!("{p}/block/0"));
        let mut seen = Vec::new();
        let got = weights()
            .encoder_mhc
            .step(l, &input, |u| {
                seen = u.to_vec();
                Ok(block_out.clone())
            })
            .unwrap();
        let (_, block_in) = f.f32(&format!("{p}/block_in"));
        let u_err = relative_l2(&seen, &block_in);
        let want = lanes(&f, &format!("{p}/lanes"), valid);
        let out_err = relative_l2(&got.data, &want.data);
        println!("{p} mhc step: u relative L2 {u_err:.3e}, lanes {out_err:.3e}");
        assert!(u_err <= STEP_U_BOUND, "{p} u: {u_err:.3e}");
        assert!(out_err <= STEP_OUT_BOUND, "{p} lanes: {out_err:.3e}");
    }
}

#[test]
fn every_encoder_layer_matches_from_the_previous_layers_lanes() {
    let mut worst = (0.0f64, String::new());
    for clip in ["en_samantha", "de_anna"] {
        let f = Fixture::load(clip);
        let valid = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
        let mut input = lanes(&f, "encoder/lanes_in", valid);
        let ctx = EncoderContext::new(weights(), input.frames, valid).unwrap();
        for (l, layer) in weights().encoder.iter().enumerate() {
            let name = format!("encoder/layer{l:02}/lanes");
            let want = lanes(&f, &name, valid);
            let got = layer.forward(&input, l, &ctx).unwrap();
            assert_eq!(
                (got.frames, got.lanes, got.width),
                (want.frames, want.lanes, want.width)
            );
            let err = relative_l2(&got.data, &want.data);
            println!("{clip} {name}: relative L2 {err:.3e}");
            if err > worst.0 {
                worst = (err, format!("{clip} {name}"));
            }
            input = want;
        }
    }
    println!("worst layer relative L2 {:.3e} ({})", worst.0, worst.1);
    assert!(worst.0 <= LAYER_BOUND, "{:.3e} on {}", worst.0, worst.1);
}

#[test]
fn encoder_reproduces_the_reference_output_on_every_clip() {
    let encoder = Encoder::new(weights());
    let mut worst = (0.0f64, String::new());
    for clip in clip_names() {
        let f = Fixture::load(&clip);
        let (shape, data) = f.f32("features");
        let features = Features {
            data,
            frames: shape[0],
            valid_frames: f.manifest["feature_frames"].as_u64().unwrap() as usize,
        };
        let out = encoder.forward(&features).unwrap();
        let (want_shape, want) = f.f32("encoder/out");
        assert_eq!([out.frames, 512], [want_shape[0], want_shape[1]], "{clip}");
        let valid = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
        assert_eq!(out.valid_frames, valid, "{clip}");
        let err = relative_l2(&out.data, &want);
        let n = valid * 512;
        let valid_err = relative_l2(&out.data[..n], &want[..n]);
        println!("{clip}: encoder/out relative L2 {err:.3e} (valid frames {valid_err:.3e})");
        if err > worst.0 {
            worst = (err, clip.clone());
        }
    }
    println!("worst encoder relative L2 {:.3e} ({})", worst.0, worst.1);
    assert!(
        worst.0 <= END_TO_END_BOUND,
        "{:.3e} on {}",
        worst.0,
        worst.1
    );
}

#[test]
fn mismatched_lanes_are_input_errors() {
    let ctx = EncoderContext::new(weights(), 4, 3).unwrap();
    let layer = &weights().encoder[0];
    let good = Lanes {
        data: vec![0.0; 4 * 4 * 512],
        frames: 4,
        lanes: 4,
        width: 512,
        valid_frames: 3,
    };
    assert!(layer.forward(&good, 0, &ctx).is_ok());
    let short = Lanes {
        data: vec![0.0; 10],
        ..good.clone()
    };
    let other_valid = Lanes {
        valid_frames: 2,
        ..good.clone()
    };
    for (lanes, l) in [(&short, 0), (&other_valid, 0), (&good, 8)] {
        let result = layer.forward(lanes, l, &ctx);
        assert!(
            matches!(result, Err(whistle::Error::Input(_))),
            "{result:?}"
        );
    }
    let encoder = Encoder::new(weights());
    assert!(matches!(
        encoder.finish(&short),
        Err(whistle::Error::Input(_))
    ));
}
