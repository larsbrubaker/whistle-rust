// The decoder's inputs besides the Engram sites (`Whistle.decode`): the scaled token embedding
// against `decoder/embedded`, and the cross-attention memory (the encoder output plus the gated
// sinusoidal position encoding, then A8) against `decoder/memory` from each clip's
// `encoder/out`.
//
// The embedding is an exact stage (relative L2 <= 1e-6). The memory ends in A8, so its bound is
// set from the error measured when it landed (CLAUDE.md "Tolerance policy").

use crate::shared::weights;
use crate::support::{clip_names, relative_l2, Fixture};

const EXACT: f64 = 1e-6;

#[test]
fn embedding_reproduces_the_reference_on_every_clip() {
    for clip in clip_names() {
        let f = Fixture::load(&clip);
        let tokens = f.tokens();
        let (shape, want) = f.f32("decoder/embedded");
        let got = weights().embed(&tokens).unwrap();
        assert_eq!(vec![got.rows(), got.cols()], shape, "{clip}");
        let err = relative_l2(got.data(), &want);
        println!("{clip}: embedding relative L2 {err:.3e}");
        assert!(err <= EXACT, "{clip}: relative L2 {err:.3e}");
    }
}

#[test]
fn out_of_vocabulary_tokens_are_input_errors() {
    let vocab = weights().embedding.rows() as u32;
    for tokens in [vec![vocab], vec![2, u32::MAX]] {
        match weights().embed(&tokens) {
            Err(whistle::Error::Input(m)) => assert!(!m.is_empty()),
            other => panic!("{tokens:?}: {other:?}"),
        }
    }
    assert_eq!(weights().embed(&[vocab - 1]).unwrap().rows(), 1);
}

/// Relative L2 against `decoder/memory`, worst clip. Measured when this landed: 4.5e-8
/// (silence; the others 0 to 2.3e-8): no A8 step flips on any clip, so the bound is the exact
/// stages' 1e-6, which a single flipped step would exceed.
const MEMORY_BOUND: f64 = 1e-6;

#[test]
fn decoder_memory_reproduces_the_reference_on_every_clip() {
    let mut worst = (0.0f64, String::new());
    for clip in clip_names() {
        let f = Fixture::load(&clip);
        let (shape, enc) = f.f32("encoder/out");
        let valid = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
        let (want_shape, want) = f.f32("decoder/memory");
        let mem = weights().decoder_memory(&enc, shape[0], valid).unwrap();
        assert_eq!(
            vec![mem.memory.rows(), mem.memory.cols()],
            want_shape,
            "{clip}"
        );
        assert_eq!(mem.valid_frames, valid, "{clip}");
        let err = relative_l2(mem.memory.data(), &want);
        println!("{clip}: decoder memory relative L2 {err:.3e}");
        if err > worst.0 {
            worst = (err, clip.clone());
        }
    }
    println!(
        "worst decoder memory relative L2 {:.3e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= MEMORY_BOUND, "{:.3e} on {}", worst.0, worst.1);
}

#[test]
fn inconsistent_encoder_output_is_an_input_error() {
    let w = weights();
    for (len, frames, valid) in [(512 * 3 - 1, 3, 3), (512 * 3, 3, 4), (512, usize::MAX, 0)] {
        match w.decoder_memory(&vec![0.0; len], frames, valid) {
            Err(whistle::Error::Input(m)) => assert!(!m.is_empty()),
            other => panic!("{len} values, {frames} frames: {other:?}"),
        }
    }
}
