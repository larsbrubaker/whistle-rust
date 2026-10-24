// Mel features (`whistle::audio::features`, `mel_matrix`) against the oracle: upstream's mel
// filter bank (test-data/fixtures/mel.npy, oracle/dump_mel.py) and every clip's `features`
// fixture (an exact stage, CLAUDE.md "Tolerance policy").

use whistle::audio::{features, mel_matrix, FFT_BINS, MELS};

use crate::support::{self, npy, relative_l2, Fixture, Npy};

/// Measured when this stage landed: the port is 1.65e-6..3.70e-6 (relative L2) from the speech
/// clips' fixtures, nl_xander the largest. The reference computes in f32 (XLA's FFT, log and
/// reductions); an independent numpy f64 evaluation of the same formula lands 1.654e-6..3.704e-6
/// from the fixtures, the same as the port to about 4 digits. So the gap is upstream's own f32
/// rounding, not a difference in the computation, and 1e-6 is out of reach without reproducing
/// XLA's operation order.
const FEATURES_BOUND: f64 = 5e-6;

#[test]
fn mel_matrix_matches_upstream() {
    let Npy::F32 { shape, data } =
        npy::read(&support::repo_root().join("test-data/fixtures/mel.npy")).unwrap()
    else {
        panic!("mel.npy is not float32");
    };
    assert_eq!(shape, [FFT_BINS, MELS]);
    let ours = mel_matrix();
    let differing = ours.iter().zip(&data).filter(|(a, b)| a != b).count();
    let error = relative_l2(&ours, &data);
    eprintln!(
        "mel_matrix: {differing} of {} entries differ, relative L2 {error:e}",
        data.len()
    );
    assert!(error <= 1e-7, "relative L2 {error:e}");
}

#[test]
fn speech_clip_features_match_the_oracle() {
    for clip in support::clip_names().into_iter().filter(|c| c != "silence") {
        let fixture = Fixture::load(&clip);
        let (_, samples) = fixture.f32("samples");
        let (shape, expected) = fixture.f32("features");
        let ours = features(&samples);
        assert_eq!([ours.frames, MELS], [shape[0], shape[1]], "{clip}");
        assert_eq!(
            ours.valid_frames as u64,
            fixture.manifest["feature_frames"].as_u64().unwrap(),
            "{clip}"
        );
        let error = relative_l2(&ours.data, &expected);
        eprintln!("{clip}: features relative L2 {error:e}");
        assert!(error <= FEATURES_BOUND, "{clip}: relative L2 {error:e}");
    }
}

#[test]
fn short_and_empty_clips_frame_without_panicking() {
    for (n, frames, valid) in [
        (0usize, 1usize, 0usize),
        (1, 1, 0),
        (159, 1, 0),
        (160, 2, 1),
        (399, 3, 2),
        (400, 3, 2),
    ] {
        let samples: Vec<f32> = (0..n)
            .map(|i| ((i * 37) % 17) as f32 / 17.0 - 0.5)
            .collect();
        let f = features(&samples);
        assert_eq!(
            (f.frames, f.valid_frames, f.data.len()),
            (frames, valid, frames * MELS),
            "{n} samples"
        );
        assert!(f.data.iter().all(|v| v.is_finite()), "{n} samples");
        assert!(
            f.data[valid * MELS..].iter().all(|&v| v == 0.0),
            "{n} samples: masked frames"
        );
    }
}

/// The silence clip's reference features are a rounding artifact: every valid log-mel value is
/// the same constant, XLA's f32 sum puts the mean one ulp away from it, and `/ (std + 1e-5)`
/// blows that ulp up to -0.16005 in every cell. The exact result for constant input is zero, and
/// the port computes exactly that; this pins that and the fixture's artifact side by side so a
/// change on either side is noticed.
#[test]
fn silence_features_are_exactly_zero_where_the_reference_has_a_rounding_artifact() {
    let fixture = Fixture::load("silence");
    let (_, samples) = fixture.f32("samples");
    let (_, expected) = fixture.f32("features");
    let ours = features(&samples);
    assert!(ours.valid_frames > 0);
    assert!(ours.data.iter().all(|&v| v == 0.0));
    let valid = ours.valid_frames * MELS;
    assert!(expected[..valid]
        .iter()
        .all(|&v| (v + 0.16005333).abs() < 1e-6));
    assert!(expected[valid..].iter().all(|&v| v == 0.0));
}
