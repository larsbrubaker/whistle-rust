# The oracle: reference fixtures and how to trust them

The Rust port is tested against upstream's own JAX model code, run on exactly the weights the
port loads. Nothing in the Rust build calls Python; the oracle only writes fixtures.

## Pieces

| File | What it does |
|---|---|
| `REFERENCE` | Pins `needle` (the Apache-2.0 model code), `whistle-weights` (Hugging Face revision of `whistle.cact` and the fp32 checkpoint) and `needle3-engine` (the black-box engine). |
| `scripts/fetch-reference.sh` | Clones `needle-reference/` at the pinned commit. |
| `scripts/fetch-weights.sh` | Downloads `models/whistle.cact`, optionally `models/whistle.safetensors` (`--checkpoint`) and the engine CLI (`--engine`); checks `scripts/weights.sha256`. |
| `oracle/setup.sh` | `oracle/.venv` (Python 3.12) from `requirements.lock` (jax 0.12.0.1, flax 0.12.10). |
| `oracle/cact_params.py` | Rebuilds the Flax parameter tree from `whistle.cact` and checks it against the checkpoint. |
| `oracle/dump.py` | Writes the per-clip stage fixtures. |
| `oracle/dump_archive.py` | Writes `test-data/fixtures/archive/`: the geometry, codebooks, and per tensor (export order and name) its dtype, shape, width and checksums of upstream's `read_export` values with 64 sampled elements; full arrays for a few tensors; tokenizer encode and decode cases. |
| `oracle/dump_mel.py` | Writes `test-data/fixtures/mel.npy`, upstream's `mel_matrix` for the default config. |
| `oracle/make_clips.py` | Synthesizes `test-data/clips/*.wav` with macOS `say` (16 kHz mono PCM16) from `oracle/clips.json`, plus one second of silence. |
| `oracle/engine_transcripts.py` | Runs the engine CLI over the clips into `test-data/engine/*.json`. |

## Weights: the archive, not the checkpoint

`cact_params.load` takes upstream's export order and names from `export._tensors` (with its
packers stubbed), dequantizes every archive tensor with upstream's `read_export`, and writes each
back into the checkpoint-shaped tree through the inverse of its export transform (transposes for
`[out, in]` matrices, the mHC `phi` reshape, the Engram table reshape, stacked layers). The
checkpoint supplies only the tree's shape and the config.

`cact_params.verify` (run `oracle/cact_params.py`) checks that rebuild leaf by leaf:
- every fp16 tensor equals the checkpoint value rounded to fp16, exactly;
- every CQ tensor equals upstream's `cq_quantize` of the checkpoint value at the width the
  archive records, except for nearest-codeword ties that float noise flips: at most 0.02% of
  128-wide groups differ (a wrong transpose makes nearly all differ);
- the Hadamard permutations and the mel matrix stored in the archive equal upstream's.

The archive mixes widths: 145 CQ2 matrices, 7 CQ4 (the embedding and the mHC `phi`
projections), 524 fp16 vectors, 4 fp32 (permutations, mel), 1 raw tokenizer. Upstream's public
exporter writes only CQ4, so it did not produce this file; the format and `read_export` handle
every width.

## Fixtures

`oracle/dump.py` runs the model in deployment numerics (`quantize.configure_deploy(act_bits=8,
kv_bits=<header>, kv_group=64)`, `quant=True`) and decodes with upstream's greedy reference
(`whistle_finetune._score_quantised`): BOS, a language tag (step 0 may only pick a tag, later
steps never do), text until EOS.

Per clip it records the input samples (`int16 / 32768`), the mel features, the stem output, the
encoder input lanes and every encoder layer's lane state, the encoder output, the greedy tokens,
the decoder's embedded input, memory (encoder output plus gated position encoding), Engram keys
and values, every decoder layer's lane state, the decoder output and the logits.

Flax does not capture inside the `nn.scan` that runs the layer stacks, so `dump.py` records each
stack's real inputs at its call, then replays it layer by layer through upstream's `_LaneBody` on
each layer's parameter slice, capturing everything inside the block. The replay's final output
must match the real stack's (checked, relative L2 ≤ 5e-2). Without quantization the two agree to
2e-6; with A8 they differ by about 2% for the reason below. Each dumped layer state is exactly
`_LaneBody` applied to the previous dumped state, which is why a Rust layer test starts from the
previous layer's fixture.

Two tiers:
- **Committed** (`test-data/fixtures/<clip>/`, ~16 MB): every clip's stage boundaries and tokens;
  for `en_samantha` and `de_anna` also every layer's lanes and the logits.
- **Local** (`test-data/fixtures-local/<clip>/`, git-ignored, ~78 MB): block internals for the
  `--full` clips (every attention, projection, norm and MLP output per layer), and the per-layer
  lanes and logits of the other clips. Regenerate with `oracle/dump.py`.

Each `manifest.json` lists every array with its tier, shape and dtype, plus the tokens, decoded
text, language and the replay errors. `index.json` lists the clips and the pinned reference.

## Why tolerances are relative L2

A8 fake quantization rounds activations to int8 steps of a row's absmax, which is discontinuous.
Float noise from a different operation order flips a few activations to the neighbouring step,
and those flips spread through later layers. Measured on upstream alone (2026-10-24, jax
0.12.0.1): the encoder output differs by 1.5–2.3% (relative L2) between eager and `jax.jit`,
and between `scan_unroll` 1 and 8. Decoding from the jitted encoder output (either unroll)
gives greedy tokens identical to the fixtures on all nine clips. So the stage
tests compare one layer at a time from the previous fixture, and the token stream is the
end-to-end contract (CLAUDE.md "Tolerance policy").

## The silence clip's features

Every valid cell of the silence clip's `features` fixture is -0.16005333, where the exact answer
is 0: every log-mel value is the same constant, XLA's f32 mean lands about 2e-6 away from it, and
`/(std + 1e-5)` magnifies that. The port computes each bin's mean and variance relative to its
first valid frame, so a constant bin gives exactly 0. The silence clip's later stage tests
therefore start from the fixture's own features, never from the port's, and its end-to-end
transcript is the silence gate's job (an empty transcript), not the reference's greedy tokens.

## What the references say about the clips

The JAX greedy transcripts match the engine's word for word on seven of nine clips. Polish
differs at the end (`... do 10 ml.` against the engine's `... do 10 milimetrów lewo.`), which is
consistent with the engine searching with beams. On a second of silence the JAX greedy decode
says "Thank you." while the engine returns an empty transcript: the engine gates silence before
or during decoding.
