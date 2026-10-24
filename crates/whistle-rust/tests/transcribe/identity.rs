// Cross-platform bit identity (CLAUDE.md "Testing framework", net 2): FNV-1a hashes of the f32
// bits of each stage (features, stem, encoder output, and the decoder's scores for the first
// steps teacher-forced on the manifest tokens) on two clips, committed as constants. Every
// target computes the same bits (deterministic math, src/math), so these must hold unchanged on
// every target: pinned on aarch64 macOS on 2026-10-24 and equal on wasm32 under wasmtime, with
// and without simd128 (`scripts/wasm-test.sh`); Linux and Windows x86_64 CI will run them too
// (not yet run on x86_64). A mismatch names the first stage where a target parts ways.
// `examples/stage_hashes.rs` prints the features, stem and encoder hashes for any WAV.

use whistle::audio::{features, read_wav};
use whistle::model::{DecoderState, Encoder};

use crate::shared::{wav, weights};
use crate::support::Fixture;

/// Decoder steps hashed: BOS, the language tag and the first text tokens.
const STEPS: usize = 4;

fn fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
    })
}

fn stage_hashes(clip: &str) -> Vec<(String, u64)> {
    let w = weights();
    let samples = read_wav(&wav(clip)).unwrap().samples;
    let feats = features(&samples);
    let stem = w.stem.forward(&feats).unwrap();
    let enc = Encoder::new(w).forward(&feats).unwrap();
    let mut out = vec![
        ("features".to_string(), fnv(&feats.data)),
        ("stem".to_string(), fnv(&stem.data)),
        ("encoder".to_string(), fnv(&enc.data)),
    ];
    let mem = w
        .decoder_memory(&enc.data, enc.frames, enc.valid_frames)
        .unwrap();
    let mut state = DecoderState::new(w, &mem, w.geometry.max_seq_len as usize).unwrap();
    for (i, &token) in Fixture::load(clip).tokens().iter().take(STEPS).enumerate() {
        out.push((format!("step {i}"), fnv(&state.step(token).unwrap())));
    }
    out
}

fn check(clip: &str, want: [u64; 3 + STEPS]) {
    let got = stage_hashes(clip);
    for (stage, hash) in &got {
        println!("{clip} {stage}: {hash:#018x}");
    }
    for ((stage, hash), want) in got.iter().zip(want) {
        assert_eq!(
            *hash, want,
            "{clip}: {stage} hash {hash:#018x}, pinned {want:#018x}"
        );
    }
}

#[test]
fn pl_zosia_stages_hash_to_the_pinned_bits_on_every_target() {
    check(
        "pl_zosia",
        [
            0xc2bd5141d5e5bea0,
            0xa6a1416203dc17f7,
            0x4f7335e5d5006c02,
            0x1b6fff94c3fe2afa,
            0xca9e7b7e19a68c11,
            0xdec0ba4984cc67e7,
            0xcb7dbf72135689a9,
        ],
    );
}

#[test]
fn en_samantha_stages_hash_to_the_pinned_bits_on_every_target() {
    check(
        "en_samantha",
        [
            0x67dab822afd3f4b1,
            0x0da51a55ee2e08e7,
            0x287a3639e83829c1,
            0xc659113e965d9702,
            0x448292c7542c9e94,
            0x018680e6e5f73f66,
            0x94207ef79e19747f,
        ],
    );
}
