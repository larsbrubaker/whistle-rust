// The stem (`whistle.Stem`) on every clip from that clip's `features` fixture, and the encoder's
// input step (pad to even, broadcast to the lanes) from `stem/out`.
//
// The stem ends in A8, so it is a quantized stage (CLAUDE.md "Tolerance policy"): its bound is
// set from the error measured when it landed.
//
// Measured (2026-10-24): eight clips at 3.3e-7 to 4.1e-7 (f32 rounding), en_samantha at 1.32e-4,
// all of it in one frame (frame 13 at 9.6e-4, every other frame below 1e-5). The cause is one A8
// flip, confirmed by rebuilding the stem from upstream's ops: in frame 13, flattened element
// 1090 has x / scale = 17.500001907 in JAX, one ulp above the tie, so it rounds to 18, while
// this port's f32 operation order lands it on the tie or below, and it rounds to 17. Adding
// that one step back brings en_samantha to 3.8e-7 like the other clips.

use whistle::audio::Features;

use crate::shared::weights;
use crate::support::{clip_names, relative_l2, Fixture};

/// Relative L2 against `stem/out`, worst clip. Measured when this landed: 1.32e-4
/// (en_samantha); the bound leaves room for about one more flipped step, not for a real error.
const STEM_BOUND: f64 = 3e-4;

#[test]
fn stem_reproduces_the_reference_on_every_clip() {
    let mut worst = (0.0f64, String::new());
    for clip in clip_names() {
        let f = Fixture::load(&clip);
        let (shape, data) = f.f32("features");
        let valid = f.manifest["feature_frames"].as_u64().unwrap() as usize;
        let features = Features {
            data,
            frames: shape[0],
            valid_frames: valid,
        };
        let out = weights().stem.forward(&features).unwrap();
        let (want_shape, want) = f.f32("stem/out");
        assert_eq!(
            [out.frames, out.width],
            [want_shape[0], want_shape[1]],
            "{clip}"
        );
        let encoder_frames = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
        assert_eq!(out.valid_frames, encoder_frames, "{clip}");
        let err = relative_l2(&out.data, &want);
        println!("{clip}: stem relative L2 {err:.3e}");
        if err > worst.0 {
            worst = (err, clip.clone());
        }
    }
    println!("worst stem relative L2 {:.3e} ({})", worst.0, worst.1);
    assert!(
        worst.0 <= STEM_BOUND,
        "stem relative L2 {:.3e} on {}",
        worst.0,
        worst.1
    );
}

#[test]
fn encoder_input_pads_to_even_and_broadcasts_to_the_lanes() {
    for clip in ["en_samantha", "de_anna"] {
        let f = Fixture::load(clip);
        let (shape, data) = f.f32("stem/out");
        let stem = whistle::model::StemOutput {
            data,
            frames: shape[0],
            width: shape[1],
            valid_frames: f.manifest["encoder_frames"].as_u64().unwrap() as usize,
        };
        let lanes = stem.to_lanes(weights().audio.enc_lanes as usize);
        let (want_shape, want) = f.f32("encoder/lanes_in");
        assert_eq!(
            vec![lanes.frames, lanes.lanes, lanes.width],
            want_shape,
            "{clip}"
        );
        assert_eq!(lanes.data, want, "{clip}: the lanes are exact copies");
    }
}

#[test]
fn inconsistent_features_are_input_errors() {
    let stem = &weights().stem;
    let bad = [
        // 10 frames of 80 bins need 800 values.
        Features {
            data: vec![0.0; 799],
            frames: 10,
            valid_frames: 10,
        },
        Features {
            data: vec![0.0; 800],
            frames: 10,
            valid_frames: 11,
        },
        Features {
            data: vec![0.0; 80],
            frames: usize::MAX,
            valid_frames: 0,
        },
    ];
    for f in &bad {
        match stem.forward(f) {
            Err(whistle::Error::Input(message)) => assert!(!message.is_empty()),
            other => panic!("{} values, {} frames: {other:?}", f.data.len(), f.frames),
        }
    }
    let ok = Features {
        data: vec![0.0; 800],
        frames: 10,
        valid_frames: 10,
    };
    let out = stem.forward(&ok).unwrap();
    assert_eq!((out.frames, out.valid_frames), (2, 2));
}
