// The token contract, teacher-forced (CLAUDE.md "Testing framework", net 2): each clip's WAV goes
// through the port's own pipeline (the VAD window `plan_windows` picks, features, encoder) and
// the cached decoder is fed the manifest's reference tokens. At every position the port's
// restricted argmax (step 0 over the language tags, later steps over the text ids) must be the
// reference's next token, except the near-ties listed in test-data/near_ties.json. The
// free-running snapshot of the port's own output (test-data/snapshots/) is pinned here too.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use whistle::audio::{features, read_wav, SAMPLE_RATE};
use whistle::model::{DecoderState, Encoder};
use whistle::{plan_windows, Options, VadMode, Whistle};

use super::{read_json, repo_root, Fixture};

/// A listed near-tie never hides a real bug: the port's margin there must stay this small.
pub const NEAR_TIE_MARGIN: f32 = 0.05;

/// One teacher-forced position: the reference's token, the port's restricted argmax, and the
/// reference token's score minus the best other allowed score (negative where they differ).
pub struct Position {
    pub index: usize,
    pub reference: u32,
    pub port: u32,
    pub margin: f32,
}

/// Teacher-forces `tokens` (BOS, language tag, text, EOS) on `wav` as `vad` windows it.
pub fn teacher_forced(
    whistle: &Whistle,
    wav: &[u8],
    vad: VadMode,
    tokens: &[u32],
) -> Vec<Position> {
    let audio = read_wav(wav).expect("clip decodes");
    // Production resamples other rates first; the contract clips are all 16 kHz, so this path
    // feeds the same samples the pipeline would.
    assert_eq!(audio.sample_rate, SAMPLE_RATE, "contract clips are 16 kHz");
    let samples = audio.samples;
    let windows = plan_windows(&samples, vad);
    assert_eq!(windows.len(), 1, "a contract clip is one window");
    let w = whistle.weights();
    let t = whistle.tokenizer();
    let enc = Encoder::new(w)
        .forward(&features(&samples[windows[0].decode.clone()]))
        .unwrap();
    let mem = w
        .decoder_memory(&enc.data, enc.frames, enc.valid_frames)
        .unwrap();
    let mut state = DecoderState::new(w, &mem, w.geometry.max_seq_len as usize).unwrap();
    let tags: Vec<u32> = t.language_tags().iter().map(|&(_, id)| id).collect();
    let first_tag = *tags.iter().min().unwrap() as usize;
    let mut out = Vec::new();
    for index in 1..tokens.len() {
        let scores = state.step(tokens[index - 1]).unwrap();
        let allowed: Vec<u32> = if index == 1 {
            tags.clone()
        } else {
            (0..first_tag as u32).collect()
        };
        let reference = tokens[index];
        let best = |skip: Option<u32>| {
            allowed
                .iter()
                .copied()
                .filter(|&id| Some(id) != skip)
                .max_by(|&a, &b| {
                    scores[a as usize]
                        .total_cmp(&scores[b as usize])
                        .then(b.cmp(&a))
                })
                .unwrap()
        };
        let port = best(None);
        let other = best(Some(reference));
        out.push(Position {
            index,
            reference,
            port,
            margin: scores[reference as usize] - scores[other as usize],
        });
    }
    out
}

pub fn vad_name(vad: VadMode) -> &'static str {
    match vad {
        VadMode::Off => "off",
        VadMode::GateAndTrim => "gate_and_trim",
        VadMode::Gate => "gate",
    }
}

/// The VAD modes the contract tests check; a near-tie listed under any other mode would be
/// checked by nothing, so it fails the load.
pub const CHECKED_MODES: [&str; 2] = ["off", "gate_and_trim"];

/// test-data/near_ties.json: `(clip, vad mode, manifest position) -> (reference, port)`.
pub fn near_ties() -> BTreeMap<(String, String, usize), (u32, u32)> {
    let v = read_json(&repo_root().join("test-data/near_ties.json"));
    v["near_ties"]
        .as_array()
        .expect("near_ties is a list")
        .iter()
        .flat_map(|e| {
            let modes: Vec<String> = e["vad"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m.as_str().unwrap().to_string())
                .collect();
            for m in &modes {
                assert!(
                    CHECKED_MODES.contains(&m.as_str()),
                    "near_ties.json: no test checks vad mode {m:?}"
                );
            }
            modes.into_iter().map(move |m| {
                let key = (
                    e["clip"].as_str().unwrap().to_string(),
                    m,
                    e["position"].as_u64().unwrap() as usize,
                );
                (
                    key,
                    (
                        e["reference"].as_u64().unwrap() as u32,
                        e["port"].as_u64().unwrap() as u32,
                    ),
                )
            })
        })
        .collect()
}

/// Checks the teacher-forced contract on every clip in `clips` under `vad`; prints every
/// position's margin and the smallest per clip, and fails on any unlisted divergence or a
/// listed near-tie that is no longer near.
pub fn check_contract(whistle: &Whistle, clips: &[(String, Vec<u8>)], vad: VadMode) {
    let ties = near_ties();
    let mode = vad_name(vad);
    let mut failures = Vec::new();
    let mut used = 0;
    for (clip, wav) in clips {
        let tokens = Fixture::load(clip).tokens();
        let positions = teacher_forced(whistle, wav, vad, &tokens);
        let margins: Vec<String> = positions
            .iter()
            .map(|p| format!("{:+.3}", p.margin))
            .collect();
        let min = positions
            .iter()
            .map(|p| p.margin)
            .fold(f32::INFINITY, f32::min);
        println!("{mode} {clip}: min margin {min:+.4}; {}", margins.join(" "));
        for p in &positions {
            match ties.get(&(clip.clone(), mode.to_string(), p.index)) {
                Some(&(reference, port)) => {
                    used += 1;
                    if p.reference != reference || p.port != port || p.margin.abs() >= NEAR_TIE_MARGIN {
                        failures.push(format!(
                            "{clip} position {}: listed near-tie {reference}->{port}, now \
                             {}->{} margin {:+.4}",
                            p.index, p.reference, p.port, p.margin
                        ));
                    }
                }
                None if p.port != p.reference => failures.push(format!(
                    "{clip} position {}: reference {}, port {}, margin {:+.4} (not a listed near-tie)",
                    p.index, p.reference, p.port, p.margin
                )),
                None => {}
            }
        }
    }
    let listed = ties.keys().filter(|k| k.1 == mode).count();
    assert_eq!(
        used, listed,
        "{mode}: a listed near-tie names no clip position"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The committed snapshot of the port's own free-running output, `name` in test-data/snapshots/.
/// It is the port's output, not the reference's; regenerate it only when a change deliberately
/// moves the output (`regenerate_snapshots`, ignored, in the decoding area).
pub fn snapshot(name: &str) -> Value {
    read_json(&repo_root().join(format!("test-data/snapshots/{name}.json")))
}

/// The port's free-running output on `clips` with `options`, as the snapshot stores it.
pub fn free_running(whistle: &Whistle, clips: &[(String, Vec<u8>)], options: &Options) -> Value {
    let mut map = serde_json::Map::new();
    for (clip, wav) in clips {
        let got = whistle.transcribe_wav(wav, options).unwrap();
        map.insert(
            clip.clone(),
            json!({"language": got.language, "text": got.text, "tokens": got.tokens}),
        );
    }
    Value::Object(map)
}
