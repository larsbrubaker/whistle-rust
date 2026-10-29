// The cost of the beam search per beam count on en_daniel (3.3 s, 16 text tokens). Run it in
// release: `cargo test --release --test decoding -- --ignored --nocapture`. It asserts nothing
// about speed; it only checks every count transcribes. Measured 2026-10-24, release, Apple
// Silicon, one thread, ms per call (extra over greedy): 1 beam 207; 2 beams 305 (+98); 3 beams
// 402 (+195); 5 beams 670 (+463); 8 beams 894 (+687); 16 beams 1347 (+1140); 5 beams with 3
// keywords 442 (+235).

use std::time::Instant;

use whistle::Options;

use crate::clips::{beams, contract_clips, transcribe, whistle};

#[test]
#[ignore = "release measurement of the cost per beam count"]
fn timing_per_beam_count() {
    let clips = contract_clips();
    let clip = clips.iter().find(|c| c.name == "en_daniel").unwrap();
    transcribe(clip, &beams(1));
    let g = &whistle().weights().geometry;
    // Per position, per decoder layer: one key and one value per KV head.
    let kv_per_position = g.num_layers as usize
        * g.num_kv_heads as usize
        * (g.qk_head_dim + g.v_head_dim) as usize
        * 4;
    for n in [1, 2, 3, 5, 8, 16] {
        let runs = 3;
        let start = Instant::now();
        let mut tokens = 0;
        for _ in 0..runs {
            tokens = transcribe(clip, &beams(n)).tokens.len();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / runs as f64;
        println!(
            "beams {n:2}: {ms:7.1} ms per call ({tokens} tokens); self-attention cache {} KiB \
             per hypothesis at that length",
            kv_per_position * (tokens + 2) / 1024
        );
    }
    let keyed = Options {
        beams: 5,
        keywords: vec!["Siobhan".into(), "PETG".into(), "MatterHackers".into()],
        ..Options::default()
    };
    let start = Instant::now();
    transcribe(clip, &keyed);
    println!(
        "beams 5 with 3 keywords: {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
}
