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

The crate's own README, [`crates/whistle-rust/README.md`](crates/whistle-rust/README.md), is
its crates.io page and its docs.rs front page (the crate docs include it, so its examples
compile as doctests). The full usage guide and options are there; this file covers the
repository.

## Install

```bash
cargo add whistle-rust
```

```rust,ignore
use whistle::{Options, Whistle};
let whistle = Whistle::from_cact(&std::fs::read("models/whistle.cact")?)?;
let transcript = whistle.transcribe_wav(&std::fs::read("speech.wav")?, &Options::default())?;
```

The package is `whistle-rust`; the library is imported as `whistle`. See the crate README for
the options (language, voice-activity gating, beams, keywords, word timestamps, cancelling) and how
audio over 30 s is windowed. Live audio streams through `Whistle::stream` with the
`Options::streaming()` preset (the crate README and `docs/design/streaming.md`).

## Getting the model

The weights are not in the crate and the library never fetches anything: `Whistle::from_cact`
takes `whistle.cact` as bytes. `whistle::model_file` pins the file this crate version was tested
against: `MODEL_URL` (Hugging Face `Cactus-Compute/whistle` at a fixed revision), `MODEL_SHA256`
and `MODEL_BYTES`. Native apps bundle it or download it once on first use; web hosts download
and cache it. Check the size and SHA-256 after a download. `whistle.cact` is Apache-2.0 by
Cactus Compute, Inc.: an application that ships it includes the Apache-2.0 license and
`THIRD_PARTY_NOTICES.md`. In this repository, `scripts/fetch-weights.sh` downloads it into
`models/` and checks it.

## How it is built

The port follows upstream's Apache-2.0 JAX model code
([cactus-compute/needle](https://github.com/cactus-compute/needle)) and is tested stage by
stage against that code running on the same weights (`docs/design/oracle.md`). The model is
described in `docs/design/model.md`.

```bash
scripts/fetch-weights.sh        # models/whistle.cact (Apache-2.0, 16.9 MB)
scripts/gates.sh                # fmt, clippy, tests, wasm32 build, package check
scripts/check-package.sh        # the packaged crate builds and documents on its own
```

CI (`.github/workflows/ci.yml`) runs the same checks on Linux, macOS and Windows on every push
and pull request, plus the wasm32 builds.

Regenerating the oracle fixtures needs the reference code, the fp32 checkpoint and Python 3.12:

```bash
scripts/fetch-reference.sh && scripts/fetch-weights.sh --checkpoint && oracle/setup.sh
oracle/.venv/bin/python -I oracle/dump.py
```

## Releasing

`.github/workflows/publish.yml` publishes to crates.io; it runs only when started by hand
(Actions > publish > Run workflow). It fetches the model, runs the tests in release, runs
`scripts/check-package.sh`, then `cargo publish -p whistle-rust`.

1. Bump `version` in the root `Cargo.toml` (`[workspace.package]`) and push to `main`.
2. Make sure the `CARGO_REGISTRY_TOKEN` repository secret is current: a 90-day crates.io token,
   rotated with `gh secret set CARGO_REGISTRY_TOKEN --repo larsbrubaker/whistle-rust`. The first
   publish needs the publish-new scope; later ones need publish-update for `whistle-rust`.
3. Run the workflow.

The token lives only in the repository secret, never on a development machine.

## License

Apache-2.0 (`LICENSE`; the crate packages copies of it and of `THIRD_PARTY_NOTICES.md`, kept
identical by the `package_files` test). The weights and the reference model code are Apache-2.0 by Cactus
Compute, Inc. (`THIRD_PARTY_NOTICES.md`).
