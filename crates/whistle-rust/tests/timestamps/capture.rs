// The capture pass (`model::Decoder::forward_with_cross_attention`) against the plain pass
// (`model::Decoder::forward`) on a fixture's token stream and committed decoder memory: the
// logits are bit-identical (the probabilities are copied from inside the same attention call),
// only the requested heads are kept, and every captured row is a distribution over the valid
// frames.

use whistle::cact::Archive;
use whistle::model::{Decoder, DecoderMemory, HeadFilter, Matrix, Weights};

use crate::support::archive::model_bytes;
use crate::support::Fixture;

#[test]
fn capturing_cross_attention_leaves_the_logits_bit_identical() {
    let bytes = model_bytes();
    let w = Weights::from_archive(&Archive::parse(&bytes).unwrap()).unwrap();
    let f = Fixture::load("en_samantha");
    let tokens = f.tokens();
    let (shape, data) = f.f32("decoder/memory");
    let mem = DecoderMemory {
        memory: Matrix::new(shape[0], shape[1], data).unwrap(),
        valid_frames: f.manifest["encoder_frames"].as_u64().unwrap() as usize,
    };
    let plain = Decoder::new(&w).forward(&tokens, &mem).unwrap();
    // Out of order, with an out-of-range pair: kept sorted, the bad pair dropped.
    let wanted = [(7, 3), (0, 1), (99, 0)];
    let (logits, cross) = Decoder::new(&w)
        .forward_with_cross_attention(&tokens, &mem, HeadFilter::Only(&wanted))
        .unwrap();
    assert_eq!(plain.data().len(), logits.data().len());
    let same = plain
        .data()
        .iter()
        .zip(logits.data())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same, "the capture pass changed the logits");
    assert_eq!(cross.heads, [(0, 1), (7, 3)]);
    assert_eq!(
        (cross.positions, cross.frames),
        (tokens.len(), mem.valid_frames)
    );
    assert_eq!(cross.data.len(), 2 * tokens.len() * mem.valid_frames);
    assert!(cross.row(1, 1, 0).is_none());
    for &(l, h) in &cross.heads {
        for t in 0..cross.positions {
            let sum: f32 = cross.row(l, h, t).unwrap().iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "layer {l} head {h} row {t}: {sum}"
            );
        }
    }
}
