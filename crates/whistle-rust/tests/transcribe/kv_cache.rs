// The cached decoder step (`model::DecoderState`) against the full recompute
// (`model::Decoder::forward`): fed the fixture's token stream one token at a time against the
// committed `decoder/memory`, every step's scores must equal the full pass's row for that
// position. Both paths run the same functions on each row, and the keys the full pass masks
// contribute an exact 0 to its softmax and weighted sum, so the match is bit for bit.

use whistle::model::{Decoder, DecoderMemory, DecoderState, Matrix};

use crate::shared::weights;
use crate::support::Fixture;

fn memory(f: &Fixture) -> DecoderMemory {
    let (shape, data) = f.f32("decoder/memory");
    let valid = f.manifest["encoder_frames"].as_u64().unwrap() as usize;
    DecoderMemory {
        memory: Matrix::new(shape[0], shape[1], data).unwrap(),
        valid_frames: valid,
    }
}

#[test]
fn cached_steps_score_every_position_exactly_as_the_full_recompute() {
    let w = weights();
    let max = w.geometry.max_seq_len as usize;
    for clip in ["en_samantha", "de_anna"] {
        let f = Fixture::load(clip);
        let tokens = f.tokens();
        let mem = memory(&f);
        let full = Decoder::new(w).forward(&tokens, &mem).unwrap();
        let mut state = DecoderState::new(w, &mem, max).unwrap();
        for (pos, &token) in tokens.iter().enumerate() {
            let step = state.step(token).unwrap();
            let want = full.row(pos).unwrap();
            let differing = step
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "{clip} position {pos}: {differing} scores differ"
            );
        }
        assert_eq!(state.tokens(), tokens.as_slice());
    }
}

#[test]
fn a_step_past_the_decodes_length_is_an_input_error() {
    let w = weights();
    let f = Fixture::load("en_samantha");
    let mem = memory(&f);
    let mut state = DecoderState::new(w, &mem, 2).unwrap();
    state.step(2).unwrap();
    state.step(8192).unwrap();
    let err = state.step(100).unwrap_err();
    assert!(matches!(err, whistle::Error::Input(_)), "{err}");
    let too_long = w.geometry.max_seq_len as usize + 1;
    assert!(matches!(
        DecoderState::new(w, &mem, too_long),
        Err(whistle::Error::Input(_))
    ));
}
