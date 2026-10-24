// The shape of the detector's output on any input: deterministic, sorted, non-overlapping,
// non-empty ranges inside the input, at every awkward length.

use whistle::vad::{has_speech, speech_segments};

use crate::signals::{non_speech_cases, speech_clips, white};

fn check_shape(name: &str, samples: &[f32]) {
    let segments = speech_segments(samples);
    assert_eq!(
        segments,
        speech_segments(samples),
        "{name}: not deterministic"
    );
    assert_eq!(has_speech(samples), !segments.is_empty(), "{name}");
    for s in &segments {
        assert!(
            s.start < s.end && s.end <= samples.len(),
            "{name}: {s:?} of {}",
            samples.len()
        );
    }
    for pair in segments.windows(2) {
        assert!(
            pair[0].end < pair[1].start,
            "{name}: {segments:?} overlap or touch"
        );
    }
}

#[test]
fn output_is_deterministic_sorted_disjoint_and_in_bounds() {
    for clip in speech_clips() {
        check_shape(&clip.name, &clip.samples);
        // Two clips back to back with a long pause: two segments, still ordered.
        let mut joined = clip.samples.clone();
        joined.extend(vec![0.0f32; 32000]);
        joined.extend(&clip.samples);
        check_shape(&format!("{} twice", clip.name), &joined);
        assert!(speech_segments(&joined).len() >= 2, "{}", clip.name);
    }
    for (name, samples) in non_speech_cases() {
        check_shape(&name, &samples);
    }
}

#[test]
fn short_and_odd_lengths_are_handled() {
    let noise = white(0.1, -20.0, 3);
    for len in [0, 1, 159, 160, 399, 400, 401, 559, 560, 1599] {
        check_shape(&format!("{len} samples"), &noise[..len]);
    }
    let clip = &speech_clips()[0].samples;
    for cut in [399, 400, 401, 1000] {
        check_shape(&format!("clip minus {cut}"), &clip[..clip.len() - cut]);
    }
}

#[test]
fn non_finite_and_full_scale_samples_do_not_panic() {
    let mut samples = white(1.0, -20.0, 5);
    samples[100] = f32::NAN;
    samples[2000] = f32::INFINITY;
    let _ = speech_segments(&samples);
    let square: Vec<f32> = (0..16000)
        .map(|i| if i % 80 < 40 { 1.0 } else { -1.0 })
        .collect();
    check_shape("full-scale square", &square);
}
