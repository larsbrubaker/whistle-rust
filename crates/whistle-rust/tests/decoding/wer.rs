// Word error rate for the decoding measurements, shared by the decoding test area and the
// crate's ranking measurement (`src/transcribe/tests.rs` includes this file by path): a
// normalizer (lower case, no punctuation, spoken numbers and units as the model writes them),
// word edit distance, and the normalized and raw rates. Pure functions, no crate imports.

/// Spoken numbers and units as the model writes them: the clips say "twenty millimeters" and
/// the model writes "20 mm", which is the same transcript to a reader.
const SPOKEN: [(&str, &str); 16] = [
    ("twenty", "20"),
    ("zwei", "2"),
    ("cinq", "5"),
    ("noventa", "90"),
    ("drie", "3"),
    ("dziesięć", "10"),
    ("millimeters", "mm"),
    ("millimeter", "mm"),
    ("millimètres", "mm"),
    ("milimetrów", "mm"),
    ("zentimeter", "cm"),
    ("centimeters", "cm"),
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("five", "5"),
];

/// Lower-cased words with punctuation removed and spoken numbers and units mapped to the
/// model's digits and abbreviations.
pub fn normalize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let spaced: String = lower
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    spaced
        .split_whitespace()
        .map(|w| {
            SPOKEN
                .iter()
                .find(|(spoken, _)| *spoken == w)
                .map_or(w.to_string(), |(_, written)| written.to_string())
        })
        .collect()
}

/// Word edits (substitutions, insertions, deletions) from `reference` to `hypothesis`.
pub fn edits(reference: &[String], hypothesis: &[String]) -> usize {
    let mut row: Vec<usize> = (0..=hypothesis.len()).collect();
    for (i, r) in reference.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let best = (row[j] + 1)
                .min(row[j + 1] + 1)
                .min(diagonal + usize::from(r != h));
            diagonal = row[j + 1];
            row[j + 1] = best;
        }
    }
    row[hypothesis.len()]
}

/// The normalized word error rate of `hypothesis` against `reference`.
pub fn wer(reference: &str, hypothesis: &str) -> f64 {
    let r = normalize(reference);
    edits(&r, &normalize(hypothesis)) as f64 / r.len().max(1) as f64
}

/// The raw word error rate: whitespace words, case and punctuation kept.
pub fn raw_wer(reference: &str, hypothesis: &str) -> f64 {
    let split = |t: &str| t.split_whitespace().map(str::to_string).collect::<Vec<_>>();
    let r = split(reference);
    edits(&r, &split(hypothesis)) as f64 / r.len().max(1) as f64
}
