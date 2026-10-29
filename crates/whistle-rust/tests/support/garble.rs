// Garbled audio for the repetition-guard tests (`src/transcribe/repeat.rs`):
// a clip delivered the way a late audio callback (a browser ScriptProcessor on a busy main
// thread) delivers it, with blocks replayed, and a brute-force measure of the longest
// repetition in a token sequence that the tests check transcripts against, independent of the
// library's own detector.

/// The pseudo-random step (a 64-bit LCG, Knuth's MMIX constants): deterministic on every
/// target, so the garbled audio, and therefore its transcript, is the same everywhere.
fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

/// `samples` cut into blocks of `block` samples and played in order, except that before about
/// one block in four the previous block plays again (a callback that fired late and replayed
/// what it had), chosen by `seed`; at most 30 s, and the last partial block is dropped.
pub fn replayed_blocks(samples: &[f32], seed: u64, block: usize) -> Vec<f32> {
    let blocks = samples.len() / block;
    let mut state = seed;
    let mut out = Vec::new();
    let mut i = 0;
    while i < blocks && out.len() < 30 * 16_000 {
        let replay = next(&mut state).is_multiple_of(4);
        let j = if replay && i > 0 { i - 1 } else { i };
        out.extend_from_slice(&samples[j * block..(j + 1) * block]);
        if !replay {
            i += 1;
        }
    }
    out
}

/// The longest run of consecutive copies of one block anywhere in `tokens`, as (copies, block
/// length, tokens the run spans), by brute force over every start and block length; the most
/// copies wins, then the longest span.
pub fn longest_repetition(tokens: &[u32]) -> (usize, usize, usize) {
    let mut best = (1, 0, 0);
    for n in 1..=tokens.len() / 2 {
        for start in 0..tokens.len() {
            let mut copies = 1;
            while start + (copies + 1) * n <= tokens.len()
                && tokens[start..start + n] == tokens[start + copies * n..start + (copies + 1) * n]
            {
                copies += 1;
            }
            if (copies, copies * n) > (best.0, best.2) {
                best = (copies, n, copies * n);
            }
        }
    }
    best
}
