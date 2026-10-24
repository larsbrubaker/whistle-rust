// The engine's own streaming output on the contract clips (test-data/engine-stream/, recorded
// by oracle/engine_stream_transcripts.py, 1 s chunks): the final committed text against ours,
// and the commit latency of both (seconds received when a word commits, minus its end time).
// Ours runs the live preset (1 beam), the engine its default 3 beams.

use serde_json::Value;

use crate::run::{feed, greedy, joined, latencies, median, CLIPS};
use crate::shared::clip;
use crate::support::{read_json, repo_root};
use crate::wer::wer;

fn engine(name: &str) -> Value {
    read_json(&repo_root().join(format!("test-data/engine-stream/{name}.json")))
}

fn engine_text(updates: &[Value]) -> String {
    let parts: Vec<&str> = updates
        .iter()
        .filter_map(|u| u["text"].as_str())
        .filter(|t| !t.is_empty())
        .collect();
    parts.join(" ")
}

fn engine_latencies(updates: &[Value]) -> Vec<f64> {
    let mut out = Vec::new();
    for u in updates {
        let received = u["received"].as_f64().expect("received");
        for w in u["words"].as_array().expect("words") {
            out.push(received - w["end"].as_f64().expect("end"));
        }
    }
    out
}

#[test]
fn final_streamed_text_is_close_to_the_engine_stream() {
    let mut total = 0.0;
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    for name in CLIPS {
        let recorded = engine(name);
        let updates = recorded["updates"].as_array().expect("updates");
        let fed = feed(&clip(name), 1.0, &greedy());
        let rate = wer(&engine_text(updates), &joined(&fed));
        println!(
            "{name:12} wer {rate:.3}  latency ours {:.2} s engine {:.2} s\n  engine {}\n  ours   {}",
            median(latencies(&fed)),
            median(engine_latencies(updates)),
            engine_text(updates),
            joined(&fed)
        );
        total += rate;
        ours.extend(latencies(&fed));
        theirs.extend(engine_latencies(updates));
    }
    let mean = total / CLIPS.len() as f64;
    println!(
        "mean wer {mean:.3}; median commit latency ours {:.2} s, engine {:.2} s",
        median(ours),
        median(theirs)
    );
    // Measured: 0.047, every clip identical but pl_zosia (0.375: the engine, at its default
    // 3 beams, reads "10 milimetrów lewo."; our greedy decode reads it otherwise, offline as
    // streamed). 0.031 before the deterministic math moved pl_zosia's greedy decode. Median
    // commit latency 1.26 s ours (1.31 s before the math), 1.15 s the engine's.
    assert!(
        mean <= 0.047,
        "mean word error rate {mean:.3} against the engine stream"
    );
}
