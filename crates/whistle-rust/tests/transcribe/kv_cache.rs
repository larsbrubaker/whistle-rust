// The cached decoder step (`model::DecoderState`) against the full recompute
// (`model::Decoder::forward`): fed the fixture's token stream one token at a time against the
// committed `decoder/memory`, every step's scores must equal the full pass's row for that
// position. Both paths run the same functions on each row, and the keys the full pass masks
// contribute an exact 0 to its softmax and weighted sum, so the match is bit for bit. The
// batched step (`DecoderState::step_batch`) against stepping each state alone: every kernel
// sums each output row on its own, so that match is bit for bit too.

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

/// `n` forks of one decode at different positions with different prefixes: fork `b` is fed
/// `b % 4` more fixture tokens than the shortest, and every third has one of them replaced.
fn forks<'w>(
    w: &'w whistle::model::Weights,
    mem: &DecoderMemory,
    tokens: &[u32],
    n: usize,
) -> Vec<DecoderState<'w>> {
    let mut root = DecoderState::new(w, mem, w.geometry.max_seq_len as usize).unwrap();
    for &t in &tokens[..2] {
        root.step(t).unwrap();
    }
    (0..n)
        .map(|b| {
            let mut s = root.clone();
            for (i, &t) in tokens[2..3 + b % 4].iter().enumerate() {
                let t = if b % 3 == 2 && i == 0 { t + 7 } else { t };
                s.step(t).unwrap();
            }
            s
        })
        .collect()
}

#[test]
fn a_batched_step_scores_every_state_exactly_as_stepping_it_alone() {
    let w = weights();
    let f = Fixture::load("en_samantha");
    let tokens = f.tokens();
    let mem = memory(&f);
    for n in [2, 3, 5, 8] {
        let mut batch = forks(w, &mem, &tokens, n);
        let mut alone = batch.clone();
        // Three steps, so the taps' history and the Engram conv read batched rows back.
        for step in 0..3 {
            let feed: Vec<u32> = (0..n).map(|b| tokens[6 + step + b % 2]).collect();
            let want: Vec<Vec<f32>> = alone
                .iter_mut()
                .zip(&feed)
                .map(|(s, &t)| s.step(t).unwrap())
                .collect();
            let mut refs: Vec<&mut DecoderState> = batch.iter_mut().collect();
            let got = DecoderState::step_batch(&mut refs, &feed).unwrap();
            assert_eq!(got.len(), n);
            for (b, (g, want)) in got.iter().zip(&want).enumerate() {
                let differing = g
                    .iter()
                    .zip(want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(g.len(), want.len());
                assert_eq!(differing, 0, "{n} states, step {step}, state {b}");
            }
        }
        for (a, b) in alone.iter().zip(&batch) {
            assert_eq!(a.tokens(), b.tokens());
        }
    }
}

#[test]
fn a_batched_step_with_a_bad_token_is_an_input_error_and_changes_nothing() {
    let w = weights();
    let f = Fixture::load("en_samantha");
    let tokens = f.tokens();
    let mem = memory(&f);
    let mut states = forks(w, &mem, &tokens, 3);
    let before: Vec<Vec<u32>> = states.iter().map(|s| s.tokens().to_vec()).collect();
    let mut refs: Vec<&mut DecoderState> = states.iter_mut().collect();
    let err = DecoderState::step_batch(&mut refs, &[5, u32::MAX, 5]).unwrap_err();
    assert!(matches!(err, whistle::Error::Input(_)), "{err}");
    let err = DecoderState::step_batch(&mut refs, &[5, 5]).unwrap_err();
    assert!(matches!(err, whistle::Error::Input(_)), "{err}");
    let after: Vec<Vec<u32>> = states.iter().map(|s| s.tokens().to_vec()).collect();
    assert_eq!(before, after);
}

#[test]
fn a_bad_token_in_a_batch_of_independent_decodes_changes_no_state() {
    // Two separate decodes share no memory, so the batch takes the one-at-a-time path; the
    // first must not step before the second's token is found to be outside the vocabulary.
    let w = weights();
    let f = Fixture::load("en_samantha");
    let mem = memory(&f);
    let max = w.geometry.max_seq_len as usize;
    let mut a = DecoderState::new(w, &mem, max).unwrap();
    let mut b = DecoderState::new(w, &mem, max).unwrap();
    let err = DecoderState::step_batch(&mut [&mut a, &mut b], &[5, u32::MAX]).unwrap_err();
    assert!(matches!(err, whistle::Error::Input(_)), "{err}");
    assert!(a.tokens().is_empty() && b.tokens().is_empty());
    // At the length limit too: `a` may not step when `b` cannot.
    let mut a = DecoderState::new(w, &mem, 2).unwrap();
    let mut b = DecoderState::new(w, &mem, 1).unwrap();
    b.step(5).unwrap();
    let err = DecoderState::step_batch(&mut [&mut a, &mut b], &[5, 5]).unwrap_err();
    assert!(matches!(err, whistle::Error::Input(_)), "{err}");
    assert!(a.tokens().is_empty());
}
