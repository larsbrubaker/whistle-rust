// Worker threads never change a result (`whistle::set_threads`): the encoder output and every
// decoder logit on en_samantha are bit-identical on one thread and on four, because threads
// split a projection by input rows, never inside a sum. The counts are set per calling thread
// (`model::with_threads`), so a parallel `cargo test` cannot race them.

use whistle::audio::{features, read_wav};
use whistle::model::{with_threads, Decoder, Encoder};

use crate::shared::{wav, weights};
use crate::support::Fixture;

fn encoder_and_logits_bits(threads: usize) -> (Vec<u32>, Vec<u32>) {
    with_threads(threads, || encoder_and_logits(threads))
}

fn encoder_and_logits(threads: usize) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(whistle::threads(), threads);
    let w = weights();
    let feats = features(&read_wav(&wav("en_samantha")).unwrap().samples);
    let enc = Encoder::new(w).forward(&feats).unwrap();
    let mem = w
        .decoder_memory(&enc.data, enc.frames, enc.valid_frames)
        .unwrap();
    let tokens = Fixture::load("en_samantha").tokens();
    let logits = Decoder::new(w).forward(&tokens, &mem).unwrap();
    let logit_bits = (0..tokens.len())
        .flat_map(|i| logits.row(i).unwrap().to_vec())
        .map(f32::to_bits)
        .collect();
    (enc.data.iter().map(|v| v.to_bits()).collect(), logit_bits)
}

#[test]
#[cfg_attr(
    target_arch = "wasm32",
    ignore = "wasm32 has no worker threads: whistle::threads() is always 1 there"
)]
fn one_and_four_worker_threads_give_bit_identical_encoder_output_and_logits() {
    let (enc1, logits1) = encoder_and_logits_bits(1);
    let (enc4, logits4) = encoder_and_logits_bits(4);
    assert_eq!(enc1, enc4, "encoder output");
    assert_eq!(logits1, logits4, "logits");
}
