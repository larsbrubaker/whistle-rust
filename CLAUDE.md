# whistle-rust — agent guidelines

A **pure Rust** port of [Whistle](https://huggingface.co/Cactus-Compute/whistle), Cactus
Compute's 16.9 MB on-device speech-to-text model: 16 kHz mono audio in; text, language and word
timings out; English, German, French, Spanish, Italian, Dutch and Polish. One library for
macOS, Windows, Linux and the browser (wasm32). First consumer: MatterCAD Rust's voice input.
Open work is in `PORTING_PLAN.md`; how the model and the oracle work is in `docs/design/`.

## Philosophy

**Quality through iterations.** Start with correct implementations, then improve.

**Be a collaborator, not a stenographer.** Apply judgment, push back when something looks wrong,
and propose what you believe is the best approach, even if it differs from what was asked.
Explain the trade-offs, then defer once a decision is made.

**Correct from first principles.** The reference says what the model computes; what a caller
needs (an empty transcript for silence, an error they can show) is decided for the caller.

## The contract (non-negotiable)

1. **Pure Rust.** No C/C++, no `-sys` crates, no native build scripts, no linking to the Needle
   engine. The library builds for `wasm32-unknown-unknown` and has no runtime dependencies.
2. **Clean room.** Port only from Apache-2.0 sources: upstream's model code in
   `needle-reference/` (github.com/cactus-compute/needle, pinned in `REFERENCE`) and the
   Apache-2.0 weights. **Never fetch, read or follow github.com/cactus-compute/cactus** (the
   engine source, under a non-commercial license) and never disassemble or trace the Needle
   engine binaries: they are a black-box transcript oracle only (`oracle/engine_transcripts.py`).
   What only the closed engine does (beam search, keyword biasing, timestamps, streaming, the
   silence gate) is designed here from its documented behaviour. Record third-party material in
   `THIRD_PARTY_NOTICES.md` in the same change.
3. **No stubs.** No `todo!()`, `unimplemented!()`, `panic!("not implemented")`, placeholders or
   partial implementations. If a dependency is missing, build it first. Every method that lands
   is complete.
4. **Never weaken a test to make it pass.** Every failure is a real bug, found by
   instrumentation and root-cause analysis. A tolerance or expected value changes only when the
   behaviour it pins is deliberately changed, and the commit says why with the measurement.
5. **Fixtures are the contract.** `test-data/fixtures/` is written only by the oracle scripts
   (`oracle/dump*.py`, listed in `docs/design/oracle.md`). Existing fixtures are regenerated only
   when `REFERENCE` changes, all scripts together, in their own commit; a new fixture lands with
   the script that writes it.

## Testing framework

Three nets. A change is not done until all that apply are green.

1. **Stage tests against the oracle** — `crates/whistle-rust/tests/<area>/<file>.rs`. Each area
   is one test binary, `crates/whistle-rust/tests/<area>.rs`, which pulls in the shared
   `support/` and its files with `#[path = "<area>/<file>.rs"] mod <file>;`. Each ported stage
   loads its clip fixtures (`support::Fixture`) and compares with the reference. Tests MUST
   exercise production code, never copies. Unit tests that need private access live in
   `src/**/tests.rs`. Name a test for the behaviour it pins.
2. **End to end** — the Rust greedy decode reproduces every clip's reference token stream
   exactly (`manifest.json` `tokens`), and the designed decode features hold their quality bar
   against the engine transcripts (`test-data/engine/`).
3. **wasm32** — the library builds for `wasm32-unknown-unknown` (`scripts/gates.sh`).

Plus the gates that run with net 1:
- `file_compliance` (`crates/workspace-gates/tests/file_compliance.rs`): every `.rs`, `.wgsl`,
  `.py`, `.sh`, `.ts` and `.js` file has at most **800 non-empty lines**, starts with a header
  comment, and contains no merge-conflict markers. No exemptions, ever.
- `cargo clippy --workspace --all-targets -- -D warnings`.

### Tolerance policy

The model runs in deployment numerics: weights dequantized from the archive, activations
fake-quantized to A8, the KV path at the archive's `kv_bits`. A8 rounding is discontinuous, so
float noise flips a few activations to the neighbouring int8 step and the flips spread:
upstream's own encoder output moves 1.5–2.8% (relative L2) between equivalent JAX execution
modes and under 3e-7 input noise (`docs/design/oracle.md`). So:

- Compare by **relative L2**, `support::relative_l2`, never by max-abs.
- **Exact stages** (archive parsing and dequantization, tokenizer, mel features, anything
  before the first A8 step): match to f32 rounding, relative L2 ≤ 1e-6, token ids identical.
- **Quantized stages:** test one layer at a time, starting from the previous layer's fixture
  (`layerNN/lanes`), never from the port's own earlier output. Each stage's bound is set from
  the measured error when the stage first lands and written beside its test with that
  measurement; it is never raised afterwards without the orchestrator's sign-off.
- **The contract is the token stream:** greedy tokens identical to the fixture on every clip.

### Test-first bug fixing

1. Write a failing test that reproduces the bug.
2. Fix the root cause with the minimal change.
3. Watch the test pass. Never commit a fix without its test.

## Numbers

- **f32** throughout inference, as the reference runs it; accumulate dot products in f32 unless
  a measured stage needs more, and keep accumulation order stable once a bound is set.
- **Rounding:** `jnp.round` is round-half-to-even; the A8 and CQ paths use
  `f32::round_ties_even`.
- **Hashes** (Engram n-gram indices) are u32 with wrapping multiply, exactly as the reference.
- `as` saturates; comment where a cast can overflow. `Result` everywhere in the library; a
  malformed archive or unsupported audio is an error with a message a caller can show, never a
  panic.

## Architecture

```
crates/
  whistle-rust/        the library (`whistle`): .cact reader, CQ dequant, tokenizer, mel front
                       end, stem, encoder, decoder, decoding; tests/ holds the stage tests
  workspace-gates/     file_compliance (tests only)
oracle/                Python (needle-reference + JAX): fixtures, clip synthesis, engine
                       transcripts; never used by the Rust build
scripts/               fetch-reference.sh, fetch-weights.sh, gates.sh
test-data/             clips/ (WAV), fixtures/ (committed oracle tier), engine/ (engine JSON)
docs/design/           how the model, the archive and the oracle work
```

The library takes the archive as bytes and never does I/O of its own beyond what a caller
passes in, so native callers memory-map or bundle `whistle.cact` and browser callers download
and cache it themselves.

## Coding standards

- **800-line limit per file**, enforced by `file_compliance`. No exemptions. Split by
  responsibility (the `file-size-refactoring` skill); never trim comments or blank lines to fit.
- **Every file starts with a header**: what the file is and how it relates to its neighbours.
- **Comments explain why**, in plain terms, as this code's own reasoning. Naming the reference
  function a port follows is useful ("follows `whistle.features`"); citing another
  implementation as the *reason* for a choice is not.
- `Result`/`Option` over `unwrap` in library code.
- No `unsafe` unless there is no alternative (SIMD intrinsics may need it); document every
  `unsafe` block.
- **Performance: measure, never guess.**
- **Copyright:** Lars Brubaker, Apache-2.0 (`LICENSE`), a derivative of Apache-2.0 upstream
  code (`THIRD_PARTY_NOTICES.md`).

## Planning docs

`PORTING_PLAN.md` is the planning doc: **open work only**, each item saying what a caller would
notice and what is needed. History lives in git. Update it in the same change that makes it
stale; delete an item when it is done, rewrite it to what is left when part of it is done.
Design docs in `docs/design/` describe how the code and the model are now.

## Commands

```bash
scripts/gates.sh                              # the full gate: fmt, clippy, nextest, doctests, wasm32
cargo nextest run --workspace --test <area>   # one area's tests
cargo test --workspace --test file_compliance # file compliance
cargo build -p whistle-rust --target wasm32-unknown-unknown
cargo clippy --workspace --all-targets -- -D warnings
scripts/fetch-weights.sh [--checkpoint] [--engine]   # models/whistle.cact (+ fp32 checkpoint, engine CLI)
scripts/fetch-reference.sh                    # needle-reference/ at the pinned commit
oracle/setup.sh                               # oracle/.venv (Python 3.12, JAX)
oracle/.venv/bin/python -I oracle/cact_params.py      # check the archive rebuild against the checkpoint
oracle/.venv/bin/python -I oracle/dump.py             # stage fixtures (regenerate only when REFERENCE changes)
oracle/.venv/bin/python -I oracle/dump_mel.py         # the mel matrix fixture (test-data/fixtures/mel.npy)
oracle/.venv/bin/python -I oracle/dump_archive.py     # the archive and tokenizer fixture (test-data/fixtures/archive/)
oracle/.venv/bin/python -I oracle/dump_engram.py      # the Engram hash fixture (test-data/fixtures/engram_indices.json)
oracle/.venv/bin/python -I oracle/engine_transcripts.py   # regenerate the engine transcripts
```

The toolchain (with clippy, rustfmt and the wasm32 target) is pinned in `rust-toolchain.toml`.

## Git

`origin` is github.com/larsbrubaker/whistle-rust (public). Only `main` is pushed; no PRs. Each
implementer branch lands on `main` as **one squash commit** whose message says what a caller can
now do; the orchestrator then deletes the branch and its worktree. Push after every landed step.

## Orchestration pattern

The main session acts as **planner and orchestrator only**; it does not write or edit library
code. All implementation is delegated to the `implementer` subagent
(`.claude/agents/implementer.md`), one scoped step at a time. Review is delegated to the
`reviewer` subagent. When tests fail, use the `fix-test-failures` agent, which treats every
failure as a real bug.

Brief each implementer with one deliverable and a 25-minute budget; a run over 30 minutes is an
error. Leave about 40% of the machine for other work: run at most round(0.6 × physical cores)
implementers at once and never more than one per 10 GB of RAM (2 on the 24 GB Mac). Give each
one `CARGO_BUILD_JOBS` = floor(0.6 × logical CPUs / implementers running) in its brief. Before
dispatching, check `sysctl vm.swapusage`: swap in use means wait.

Each implementer works on its own branch in its own worktree, created explicitly by the
orchestrator: `git worktree add .claude/worktrees/<step> -b <step> main` (git-ignored). Don't
rely on the Agent tool's `isolation: "worktree"`: this repo sits inside the rust-apps umbrella,
and that option branches whichever repo the session's working directory is in. A worktree has no
`models/`, `needle-reference/` or `test-data/fixtures-local/`; link them from the main checkout
(`ln -s ../../../models .claude/worktrees/<step>/models`, likewise the others) before the brief.
The orchestrator lands a finished branch with `git merge --squash <step>`, commits, runs
`scripts/gates.sh`, pushes, then `git worktree remove` and `git branch -D`.

Working conventions that every brief repeats:
- Implementers commit on their own branch, never push, and never edit `PORTING_PLAN.md`; they
  list what their change makes stale, and the orchestrator prunes the plan when it merges.
- Every stage that lands has its oracle test, with the measured error in the report.
