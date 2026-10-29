# whistle-rust

A pure Rust port of [Whistle](https://huggingface.co/Cactus-Compute/whistle), Cactus Compute's
16.9 MB on-device speech-to-text model: 16 kHz mono audio in; text, language and word timings
out; English, German, French, Spanish, Italian, Dutch and Polish. One library for macOS,
Windows, Linux and the browser (wasm32), with no native code and no runtime dependencies.

Transcription of up to 10 minutes of audio works, in one pass up to 30 s and in windows beyond
that, cut inside the voice-activity detector's pauses. Greedy (`beams: 1`) reproduces the
reference decode on every test clip, the default 3-beam search reproduces the engine's
transcripts, keyword biasing favours the names and jargon you list, and word timestamps give
each word's start, end and probability. Live audio streams through `Whistle::stream` with the
live preset `Options::streaming()`, committing each word once two passes agree on it.

## Install

```bash
cargo add whistle-rust
```

The package is `whistle-rust`; the library is imported as `whistle`:

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
`GateAndTrim` by default, so silence gives an empty transcript), takes a `Cancel` handle that
stops a transcription from another thread with `Error::Cancelled`, runs a beam search (`beams`:
3 by default, the smallest count that reproduced every engine transcript on the test clips, for
about +195 ms on a 3.3 s clip; `beams: 1` is the greedy decode; up to 16), and favours names,
places and jargon you list (`keywords: vec!["PETG".into()]`, strength `keyword_bonus`; see
[search.rs](https://github.com/larsbrubaker/whistle-rust/blob/main/crates/whistle-rust/src/transcribe/search.rs) and
[keywords.rs](https://github.com/larsbrubaker/whistle-rust/blob/main/crates/whistle-rust/src/transcribe/keywords.rs)).
`word_timestamps: true` fills `transcript.words` with each word's `start` and `end` in seconds
from the start of the input and its `probability`, aligned from the decoder's cross-attention
(within one 80 ms frame of the engine's on the test clips, for about +55 ms on a 3.4 s clip;
[align.rs](https://github.com/larsbrubaker/whistle-rust/blob/main/crates/whistle-rust/src/transcribe/align.rs)).
`Transcript` is `PartialEq` but not `Eq` (the times are f32).
`transcript.repeated` is true when the decoder ran into a repetition loop (a token or phrase
repeated over and over, which garbled audio, noise or a language the model was not trained on
can cause) and was rolled back past it: the text holds no loop, but it is a reading of audio
the model could barely hear. A caller with another reading of the same audio (a live stream's
committed text) may prefer that one ([repeat.rs](https://github.com/larsbrubaker/whistle-rust/blob/main/crates/whistle-rust/src/transcribe/repeat.rs)).
Audio over 30 s is split into windows of at most 30 s (`whistle::plan_windows` shows the cuts;
[window.rs](https://github.com/larsbrubaker/whistle-rust/blob/main/crates/whistle-rust/src/transcribe/window.rs)):
the language detected on the first window with at least 3 words is forced on the rest (set
`language` to choose it), the texts join with spaces, and word times count from the start of
the input; over 10 minutes is `Error::Input`.

Live audio goes through `Whistle::stream`, best with the live preset `Options::streaming()`
(one beam: a one-thread wasm update stays under the 1 s step, where 3 beams can fall behind):

```rust,no_run
use whistle::{Options, Whistle};

fn live(whistle: &Whistle, chunks: impl Iterator<Item = Vec<f32>>) -> Result<(), whistle::Error> {
    let mut stream = whistle.stream(&Options::streaming())?;
    for chunk in chunks {
        // 16 kHz mono, any size; small chunks (a step or less) keep each push short.
        let update = stream.push(&chunk)?;
        println!("{} [{}]", update.text, update.pending);
    }
    println!("{}", stream.finish()?.text);
    Ok(())
}
```

Each `StreamUpdate` holds the words newly committed (never retracted, times from the start of
the stream), their text (`continues` when it begins with the rest of the word before: append
it with no space), the unconfirmed `pending` tail (`pending_continues` when it begins with the
rest of the last committed word, "0 mm" after a committed "2": show it with no space before
it), the language and any audio `dropped` at the 30 s cap; `transcript()` returns everything
committed so far (its `repeated` set when a pass it committed words from ran into a loop).
`StreamUpdate` is `#[non_exhaustive]`: later versions may add fields. A
word commits when two passes one `stream_step` (1 s) apart agree on it, about 1.3 s after it
ends, or when a pause ends the utterance ([stream.rs](https://github.com/larsbrubaker/whistle-rust/blob/main/crates/whistle-rust/src/transcribe/stream.rs)). The stream borrows the `Whistle`, so an app
keeps the model alive beside it.

A microphone at its own rate (44.1 or 48 kHz) goes through `audio::StreamResampler` on the way
in. It keeps its filter's history and phase across pushes, so the pushes come out exactly as
`audio::resample` converts the whole recording; resampling each push on its own would put a
seam at every push boundary, enough to change a word. `finish` gives the last few
milliseconds, which wait for input past them:

```rust,no_run
use whistle::audio::StreamResampler;
use whistle::{Options, Whistle};

fn live_at(
    whistle: &Whistle,
    rate: u32,
    chunks: impl Iterator<Item = Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut to_16k = StreamResampler::new(rate, 16_000)?;
    let mut stream = whistle.stream(&Options::streaming())?;
    for chunk in chunks {
        let update = stream.push(&to_16k.push(&chunk))?;
        println!("{} [{}]", update.text, update.pending);
    }
    stream.push(&to_16k.finish())?;
    println!("{}", stream.finish()?.text);
    Ok(())
}
```

## Getting the model

The weights are not in the crate. `whistle::model_file` names the exact file this crate version
was tested against:

- `MODEL_URL`: `whistle.cact` from Hugging Face `Cactus-Compute/whistle` at a pinned revision
- `MODEL_SHA256`: its SHA-256, as lowercase hex
- `MODEL_BYTES`: its size, 16,919,407 bytes

`Whistle::from_cact` takes the archive as bytes and does no I/O, so how the file arrives is the
caller's choice. A native app bundles it with the app or downloads it once on first use into its
data directory; a web host downloads it (from that URL or its own copy) and caches it, for
example in the Cache API or IndexedDB. Check the length and SHA-256 after a download:

```rust,no_run
use whistle::model_file::{MODEL_BYTES, MODEL_SHA256};

fn check(model: &[u8], sha256_hex: &str) -> Result<(), String> {
    if model.len() as u64 != MODEL_BYTES || sha256_hex != MODEL_SHA256 {
        return Err("whistle.cact is not the file this crate was tested against".into());
    }
    Ok(())
}
```

**License note.** `whistle.cact` is Copyright Cactus Compute, Inc., under the Apache License
2.0. An application that ships or downloads it must include the Apache-2.0 license text and the
notice in `THIRD_PARTY_NOTICES.md` (packaged with this crate).

## Tests

The crate's tests compare every stage with fixtures computed by the reference model and run on
the real model file, so they need the repository (`test-data/` and `models/`) and run there,
not from the downloaded package: clone
[github.com/larsbrubaker/whistle-rust](https://github.com/larsbrubaker/whistle-rust), run
`scripts/fetch-weights.sh`, then `scripts/gates.sh`.

## License

Apache-2.0 (`LICENSE`). whistle-rust is a Rust translation of Cactus Compute's Apache-2.0 model
code ([cactus-compute/needle](https://github.com/cactus-compute/needle)); the weights are
Apache-2.0 by Cactus Compute, Inc. (`THIRD_PARTY_NOTICES.md`). Source, tests and the design
docs: [github.com/larsbrubaker/whistle-rust](https://github.com/larsbrubaker/whistle-rust).
