# whistle-rust

A pure Rust port of [Whistle](https://huggingface.co/Cactus-Compute/whistle), Cactus Compute's
16.9 MB on-device speech-to-text model: 16 kHz mono audio in; text, language and word timings
out; English, German, French, Spanish, Italian, Dutch and Polish. One library for macOS,
Windows, Linux and the browser (wasm32), with no native code and no runtime dependencies.

**Status: in progress.** Greedy transcription of up to 30 s of audio works and reproduces the
reference decode on every test clip; beam search, timestamps, streaming, windowing past 30 s
and performance work are still to come (`PORTING_PLAN.md`).

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
`GateAndTrim` by default, so silence gives an empty transcript), and takes a `Cancel` handle
that stops a transcription from another thread with `Error::Cancelled`.

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
