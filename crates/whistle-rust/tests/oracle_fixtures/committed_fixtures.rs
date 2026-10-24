// The committed fixture tier is complete: every clip in index.json has its WAV and manifest, and
// every committed array exists with the shape and dtype its manifest records.

use crate::support::{self, Fixture};

#[test]
fn every_clip_has_audio_and_a_manifest() {
    let clips = support::clip_names();
    assert!(clips.len() >= 9, "index.json lists {} clips", clips.len());
    for clip in &clips {
        let wav = support::repo_root()
            .join("test-data/clips")
            .join(format!("{clip}.wav"));
        assert!(wav.is_file(), "{clip}: missing {}", wav.display());
        let fixture = Fixture::load(clip);
        assert_eq!(fixture.manifest["clip"], clip.as_str());
    }
}

#[test]
fn every_committed_array_matches_its_manifest() {
    for clip in support::clip_names() {
        let fixture = Fixture::load(&clip);
        for name in fixture.names() {
            if fixture.tier(&name) != "core" {
                continue;
            }
            let array = fixture.array(&name);
            let entry = &fixture.manifest["arrays"][&name];
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|d| d.as_u64().expect("dim") as usize)
                .collect();
            assert_eq!(array.shape(), shape.as_slice(), "{clip}: {name} shape");
            assert_eq!(
                array.dtype(),
                entry["dtype"].as_str().unwrap(),
                "{clip}: {name} dtype"
            );
        }
    }
}

#[test]
fn every_clip_commits_its_stage_boundaries() {
    for clip in support::clip_names() {
        let fixture = Fixture::load(&clip);
        for name in [
            "samples",
            "features",
            "stem/out",
            "encoder/out",
            "tokens",
            "decoder/out",
        ] {
            assert_eq!(
                fixture.tier(name),
                "core",
                "{clip}: {name} must be committed"
            );
        }
    }
}

#[test]
fn token_arrays_match_the_manifest_token_streams() {
    for clip in support::clip_names() {
        let fixture = Fixture::load(&clip);
        let tokens: Vec<u32> = match fixture.array("tokens") {
            support::Npy::I32 { data, .. } => data.iter().map(|&t| t as u32).collect(),
            other => panic!("{clip}: tokens are {}", other.dtype()),
        };
        assert_eq!(tokens, fixture.tokens(), "{clip}");
        // BOS, a language tag, then text up to EOS (upstream's greedy prompt).
        assert_eq!(tokens[0], 2, "{clip}: starts with BOS");
        assert!(tokens[1] >= 8192, "{clip}: second token is a language tag");
        assert_eq!(*tokens.last().unwrap(), 1, "{clip}: ends with EOS");
    }
}

#[test]
fn relative_l2_measures_the_documented_quantity() {
    assert_eq!(support::relative_l2(&[3.0, 4.0], &[3.0, 4.0]), 0.0);
    let err = support::relative_l2(&[3.0, 4.5], &[3.0, 4.0]);
    assert!((err - 0.1).abs() < 1e-12, "{err}");
    assert_eq!(support::relative_l2(&[0.5], &[0.0]), 0.5);
}

#[test]
fn npy_reader_rejects_malformed_files() {
    assert!(support::npy::parse(b"not numpy").is_err());
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
    let padded = format!("{header:<54}\n");
    bytes.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    bytes.extend_from_slice(padded.as_bytes());
    bytes.extend_from_slice(&1.5f32.to_le_bytes());
    // One element short of the declared shape.
    assert!(support::npy::parse(&bytes).is_err());
    bytes.extend_from_slice(&2.5f32.to_le_bytes());
    let array = support::npy::parse(&bytes).expect("well-formed");
    assert_eq!(
        array,
        support::Npy::F32 {
            shape: vec![2],
            data: vec![1.5, 2.5]
        }
    );
}
