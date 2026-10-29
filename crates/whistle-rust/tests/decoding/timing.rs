// The cost of the beam search per beam count on en_daniel (3.4 s, 16 text tokens). Run it in
// release: `cargo test --release -p whistle-rust --test decoding timing_per_beam_count --
// --ignored --nocapture`. It asserts nothing about speed; it only checks every count
// transcribes. Each count is timed over 7 calls and reported as their median, so a call slowed
// by other work on the machine does not move the figure.
//
// Measured 2026-10-29, release, Apple Silicon, the default kernel threads (up to 4), on a
// machine shared with other builds (load average 21-35 on 10 cores); ms per call (extra over
// greedy), the median of six runs of this test: 1 beam 164; 2 beams 228 (+58); 3 beams 229
// (+55); 5 beams 339 (+161); 8 beams 453 (+266); 16 beams 792 (+578); 5 beams with 3 keywords
// 309.

use whistle::Options;

use crate::clips::{beams, contract_clips, transcribe, whistle};
use crate::support::timed_runs_ms;

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
    let mut greedy = 0.0;
    for n in [1, 2, 3, 5, 8, 16] {
        let mut tokens = 0;
        let (times, ms) = timed_runs_ms(7, || {
            tokens = transcribe(clip, &beams(n)).tokens.len();
        });
        if n == 1 {
            greedy = ms;
        }
        println!(
            "beams {n:2}: median {ms:7.1} ms per call (+{:.1} over greedy; runs {times:.0?}; \
             {tokens} tokens); self-attention cache {} KiB per hypothesis at that length",
            ms - greedy,
            kv_per_position * (tokens + 2) / 1024
        );
    }
    let keyed = Options {
        beams: 5,
        keywords: vec!["Siobhan".into(), "PETG".into(), "MatterHackers".into()],
        ..Options::default()
    };
    let (times, ms) = timed_runs_ms(7, || {
        transcribe(clip, &keyed);
    });
    println!("beams 5 with 3 keywords: median {ms:.1} ms (runs {times:.0?})");
}
