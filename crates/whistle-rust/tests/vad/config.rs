// Extreme `VadConfig` values (±∞, NaN, 1e30, negative, 0) in every field, on a real clip and on
// empty input: no panic, no allocation sized by the configuration, and well-formed output.

use whistle::vad::{speech_segments_with, VadConfig};

use crate::signals::load_clip;

type Setter = fn(&mut VadConfig, f32);

fn fields() -> Vec<(&'static str, Setter)> {
    vec![
        ("absolute_floor_db", |c, v| c.absolute_floor_db = v),
        ("floor_window_s", |c, v| c.floor_window_s = v),
        ("floor_smoothing_s", |c, v| c.floor_smoothing_s = v),
        ("margin_on_db", |c, v| c.margin_on_db = v),
        ("margin_off_db", |c, v| c.margin_off_db = v),
        ("modulation_window_s", |c, v| c.modulation_window_s = v),
        ("min_modulation_span_s", |c, v| c.min_modulation_span_s = v),
        ("min_modulation_db", |c, v| c.min_modulation_db = v),
        ("low_cut_hz", |c, v| c.low_cut_hz = v),
        ("min_speech_s", |c, v| c.min_speech_s = v),
        ("merge_gap_s", |c, v| c.merge_gap_s = v),
        ("hangover_before_s", |c, v| c.hangover_before_s = v),
        ("hangover_after_s", |c, v| c.hangover_after_s = v),
    ]
}

const EXTREMES: [f32; 7] = [
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    1e30,
    -1e30,
    -1.0,
    0.0,
];

#[test]
fn extreme_settings_never_panic_and_give_well_formed_segments() {
    let clip = load_clip("en_daniel");
    for (name, set) in fields() {
        for value in EXTREMES {
            let mut config = VadConfig::default();
            set(&mut config, value);
            assert_eq!(
                speech_segments_with(&config, &[]),
                vec![],
                "{name} = {value}"
            );
            let segments = speech_segments_with(&config, &clip);
            for s in &segments {
                assert!(
                    s.start < s.end && s.end <= clip.len(),
                    "{name} = {value}: {s:?}"
                );
            }
            for p in segments.windows(2) {
                assert!(p[0].end < p[1].start, "{name} = {value}: {segments:?}");
            }
        }
    }
}

#[test]
fn every_field_extreme_at_once_never_panics() {
    let clip = load_clip("en_daniel");
    for value in EXTREMES {
        let mut config = VadConfig::default();
        for (_, set) in fields() {
            set(&mut config, value);
        }
        let segments = speech_segments_with(&config, &clip);
        assert!(
            segments
                .iter()
                .all(|s| s.start < s.end && s.end <= clip.len()),
            "{value}"
        );
    }
}

#[test]
fn a_nan_setting_behaves_as_its_default() {
    let clip = load_clip("en_daniel");
    let expected = speech_segments_with(&VadConfig::default(), &clip);
    for (name, set) in fields() {
        let mut config = VadConfig::default();
        set(&mut config, f32::NAN);
        assert_eq!(
            speech_segments_with(&config, &clip),
            expected,
            "{name} = NaN"
        );
    }
}
