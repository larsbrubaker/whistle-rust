# whistle-rust

A pure Rust port of [Whistle](https://huggingface.co/Cactus-Compute/whistle), Cactus Compute's
16.9 MB on-device speech-to-text model: 16 kHz mono audio in; text, language and word timings
out; English, German, French, Spanish, Italian, Dutch and Polish. One library for macOS,
Windows, Linux and the browser (wasm32), with no native code and no runtime dependencies.

**Status: in progress.** Transcription of up to 10 minutes of audio works, in one pass up to
30 s and in windows beyond that (`docs/design/windowing.md`): cut inside the detector's
pauses, with a hard cut and a 1 s overlap only in speech with no pause, and at fixed places
with the detector off. Greedy (`beams: 1`) reproduces the reference decode on every test
clip, the default 3-beam search reproduces the engine's transcripts, keyword biasing favours
the names and jargon you list, and word timestamps give each word's start, end and
probability. Streaming is still to come
(`PORTING_PLAN.md`).

## Usage

```rust,no_run
use whistle::{Options, Whistle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The library never fetches anything: the caller reads, bundles or downloads the archive.
    let model = std::fs::read("models/whistle.cact")?;
    let whistle = Whistle::from_cact(&model)?;
    let wav = std::fs::read("speech.wav")?;
    let transcript = whistle.transcribe_wav(&wav, &Options::default())?;
    println!("{:?}: {}", transcript.language, transcript.text);
    Ok(())
}
```

`Whistle::transcribe` takes 16 kHz mono `f32` samples instead. `Options` forces a language
(`language: Some("de".into())`), sets how the voice-activity detector gates the audio (`vad`:
`GateAndTrim` by default, so silence gives an empty transcript), takes a `Cancel` handle
that stops a transcription from another thread with `Error::Cancelled`, runs a beam search
(`beams`: 3 by default, the smallest count that reproduced every engine transcript on the test
clips, for about +195 ms on a 3.3 s clip; `beams: 1` is the greedy decode; up to 16), and favours names, places and jargon you list
(`keywords: vec!["PETG".into()]`, strength `keyword_bonus`). See `docs/design/decoding.md`.
`word_timestamps: true` fills `transcript.words` with each word's `start` and `end` in seconds
from the start of the input and its `probability`, aligned from the decoder's cross-attention
(within one 80 ms frame of the engine's on the test clips, for about +55 ms on a 3.4 s clip;
`docs/design/timestamps.md`). `Transcript` is `PartialEq` but not `Eq` (the times are f32).
Audio over 30 s is split into windows of at most 30 s (`whistle::plan_windows` shows the cuts):
the language detected on the first window with at least 3 words is forced on the rest (set
`language` to choose it), the texts join with spaces, and word times count from the start of
the input; over 10 minutes is `Error::Input`.

## How it is built

The port follows upstream's Apache-2.0 JAX model code
([cactus-compute/needle](https://github.com/cactus-compute/needle)) and is tested stage by
stage against that code running on the same weights (`docs/design/oracle.md`). The model is
described in `docs/design/model.md`.

```bash
scripts/fetch-weights.sh        # models/whistle.cact (Apache-2.0, 16.9 MB)
scripts/gates.sh                # fmt, clippy, tests, wasm32 build
```

Regenerating the oracle fixtures needs the reference code, the fp32 checkpoint and Python 3.12:

```bash
scripts/fetch-reference.sh && scripts/fetch-weights.sh --checkpoint && oracle/setup.sh
oracle/.venv/bin/python -I oracle/dump.py
```

## License

Apache-2.0 (`LICENSE`). The weights and the reference model code are Apache-2.0 by Cactus
Compute, Inc. (`THIRD_PARTY_NOTICES.md`).
