// Each decoder layer (block plus mHC lane step) from the previous layer's committed lanes
// (`decoder/lanes_in` for layer 0) on en_samantha and de_anna; the decoder output
// (`decoder/out`, after the final norm) on every clip from its `tokens` and `decoder/memory`;
// the logits on en_samantha and de_anna (the clips that commit them); and, on every clip, that
// the greedy picks (step 0 among the language tags, later steps among the text tokens, as
// `oracle/dump.py` `greedy` restricts them) are the clip's next tokens. Quantized stages
// (CLAUDE.md "Tolerance policy"): each bound is set from the error measured when the decoder
// landed, written beside it.

use whistle::model::{Decoder, DecoderContext, DecoderMemory, Lanes, Matrix};

use crate::decoder_block::memory;
use crate::shared::weights;
use crate::support::{clip_names, relative_l2, Fixture};

fn lanes(f: &Fixture, name: &str) -> Lanes {
    let (shape, data) = f.f32(name);
    Lanes {
        data,
        frames: shape[0],
        lanes: shape[1],
        width: shape[2],
        valid_frames: shape[0],
    }
}

/// One layer from the previous layer's fixture lanes, worst over layers and both clips.
/// Measured (2026-10-24): worst 1.39e-4 (de_anna layer 2); de_anna layer 4 2.5e-5; every other
/// layer of both clips 7.5e-8 to 3.5e-7 (f32 rounding, no flipped step). The flips are A8
/// steps moved by f32 noise, as in the encoder; the bound leaves room for a few more.
/// Re-set when the CQ matrices became packed (`kernels::packed`: the FWHT moves to the
/// activations and the sum is grouped, so the rounding differs). Measured (2026-10-24): worst
/// 1.08e-3 (en_samantha layer 3), de_anna layer 0 3.9e-4, de_anna layers 6 and 7 7.5e-5 and
/// 7.3e-5, every other layer 8.3e-8 to 4.6e-7; tokens unchanged on every clip. The bound stays
/// under the reference's own 2.6–2.8e-3 between execution modes for one block (`oracle.md`).
const LAYER_BOUND: f64 = 2e-3;

#[test]
fn every_decoder_layer_matches_from_the_previous_layers_lanes() {
    let mut worst = (0.0f64, String::new());
    for clip in ["en_samantha", "de_anna"] {
        let f = Fixture::load(clip);
        let tokens = f.tokens();
        let mem = memory(&f);
        let ctx = DecoderContext::new(weights(), &tokens, &mem).unwrap();
        let mut input = lanes(&f, "decoder/lanes_in");
        for (l, layer) in weights().decoder.iter().enumerate() {
            let name = format!("decoder/layer{l:02}/lanes");
            let want = lanes(&f, &name);
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

/// `decoder/out` from `tokens` and `decoder/memory`, worst clip. Measured (2026-10-24):
/// de_anna 1.55e-2, en_daniel 6.3e-3, en_samantha 8.4e-3, es_monica 1.34e-2, fr_thomas
/// 1.20e-2, it_alice 1.05e-2, nl_xander 1.20e-2, pl_zosia 1.65e-2 (worst), silence 6.0e-7.
/// Per layer the port is at most 1.4e-4 from the reference, and each flipped A8 step moves
/// every later layer's inputs, so flips compound over 8 layers, as the encoder's do (2-3%).
const OUT_BOUND: f64 = 3e-2;

#[test]
fn decoder_output_reproduces_the_reference_on_every_clip() {
    let decoder = Decoder::new(weights());
    let mut worst = (0.0f64, String::new());
    for clip in clip_names() {
        let f = Fixture::load(&clip);
        let out = decoder.hidden(&f.tokens(), &memory(&f)).unwrap();
        let (shape, want) = f.f32("decoder/out");
        assert_eq!(vec![out.rows(), out.cols()], shape, "{clip}");
        let err = relative_l2(&out.data(), &want);
        println!("{clip}: decoder/out relative L2 {err:.3e}");
        if err > worst.0 {
            worst = (err, clip.clone());
        }
    }
    println!(
        "worst decoder/out relative L2 {:.3e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= OUT_BOUND, "{:.3e} on {}", worst.0, worst.1);
}

/// `decoder/logits` from `tokens` and `decoder/memory`, worst of the two clips. Measured
/// (2026-10-24): en_samantha 3.19e-3, de_anna 7.17e-3 (worst), following `decoder/out`
/// (8.4e-3, 1.55e-2) through one more A8 step and the tied head. The bound is about twice the
/// worst; the greedy picks are the contract and are checked exactly, on every clip, below.
const LOGITS_BOUND: f64 = 1.5e-2;

#[test]
fn logits_reproduce_the_reference_on_the_clips_that_have_them() {
    let decoder = Decoder::new(weights());
    let mut worst = (0.0f64, String::new());
    for clip in ["en_samantha", "de_anna"] {
        let f = Fixture::load(clip);
        let logits = decoder.forward(&f.tokens(), &memory(&f)).unwrap();
        let (shape, want) = f.f32("decoder/logits");
        assert_eq!(vec![logits.positions(), logits.vocab()], shape, "{clip}");
        let err = relative_l2(logits.data(), &want);
        println!("{clip}: decoder/logits relative L2 {err:.3e}");
        if err > worst.0 {
            worst = (err, clip.to_string());
        }
    }
    println!(
        "worst decoder/logits relative L2 {:.3e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= LOGITS_BOUND, "{:.3e} on {}", worst.0, worst.1);
}

/// Scoring each clip's `tokens` reproduces the reference's greedy choices: at every position
/// but the last (EOS), the best token among those greedy may pick (step 0 the language tags,
/// later steps the text tokens, as `oracle/dump.py` `greedy` restricts them) is the next token.
/// Smallest top-1 over top-2 margin per clip (logit units, position), measured (2026-10-24):
/// de_anna 0.975 (5), en_daniel 0.670 (14), en_samantha 0.638 (4), es_monica 0.411 (3),
/// fr_thomas 0.692 (11), it_alice 0.345 (1), nl_xander 0.685 (12), pl_zosia 0.186 (5, the
/// closest), silence 3.807 (1).
#[test]
fn restricted_argmax_is_the_next_token_on_every_clip() {
    let decoder = Decoder::new(weights());
    let text = 8192usize; // the first language tag (docs/design/model.md "Configuration")
    for clip in clip_names() {
        let f = Fixture::load(&clip);
        let tokens = f.tokens();
        let logits = decoder.forward(&tokens, &memory(&f)).unwrap();
        let mut margin = (f32::INFINITY, 0usize);
        for t in 0..tokens.len() - 1 {
            let row = logits.row(t).unwrap();
            let allowed = if t == 0 { text..row.len() } else { 0..text };
            let mut best: Vec<usize> = allowed.collect();
            best.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            assert_eq!(best[0] as u32, tokens[t + 1], "{clip} position {t}");
            let m = row[best[0]] - row[best[1]];
            if m < margin.0 {
                margin = (m, t);
            }
        }
        println!(
            "{clip}: smallest greedy margin {:.3} at position {}",
            margin.0, margin.1
        );
        assert!(logits.row(tokens.len()).is_none());
    }
}

#[test]
fn bad_decoder_inputs_are_input_errors() {
    let f = Fixture::load("en_samantha");
    let mem = memory(&f);
    let decoder = Decoder::new(weights());
    let max = weights().geometry.max_seq_len as usize;
    for tokens in [vec![], vec![2u32; max + 1], vec![2, 8199]] {
        let result = decoder.forward(&tokens, &mem);
        assert!(
            matches!(result, Err(whistle::Error::Input(_))),
            "{} tokens",
            tokens.len()
        );
    }
    let mut bad = mem.clone();
    bad.valid_frames = bad.memory.rows() + 1;
    assert!(matches!(
        decoder.forward(&[2, 8192], &bad),
        Err(whistle::Error::Input(_))
    ));
    // No encoded audio: rejected where the memory is built, and where a hand-built one is used.
    assert!(matches!(
        weights().decoder_memory(&[], 0, 0),
        Err(whistle::Error::Input(m)) if m.contains("no frames")
    ));
    let empty = DecoderMemory {
        memory: Matrix::new(0, 512, Vec::new()).unwrap(),
        valid_frames: 0,
    };
    for result in [
        decoder.forward(&[2], &empty).map(|_| ()),
        DecoderContext::new(weights(), &[2], &empty).map(|_| ()),
        weights().decoder[0]
            .cross_attn
            .memory_kv(&empty, 8)
            .map(|_| ()),
    ] {
        assert!(
            matches!(&result, Err(whistle::Error::Input(m)) if m.contains("no frames")),
            "{result:?}"
        );
    }
    let ctx = DecoderContext::new(weights(), &[2, 8192], &mem).unwrap();
    let good = Lanes {
        data: vec![0.0; 2 * 4 * 512],
        frames: 2,
        lanes: 4,
        width: 512,
        valid_frames: 2,
    };
    let layer = &weights().decoder[0];
    assert!(layer.forward(&good, 0, &ctx).is_ok());
    let short = Lanes {
        data: vec![0.0; 10],
        ..good.clone()
    };
    let three = Lanes {
        data: vec![0.0; 3 * 4 * 512],
        frames: 3,
        valid_frames: 3,
        ..good.clone()
    };
    for (lanes, l) in [(&short, 0), (&three, 0), (&good, 8)] {
        let result = layer.forward(lanes, l, &ctx);
        assert!(
            matches!(result, Err(whistle::Error::Input(_))),
            "{result:?}"
        );
    }
    assert!(matches!(
        decoder.finish(&short),
        Err(whistle::Error::Input(_))
    ));
    // The one-token sequence (BOS alone) is step 0 of a greedy decode.
    let one = decoder.forward(&[2], &mem).unwrap();
    assert_eq!((one.positions(), one.vocab()), (1, 8199));
}
