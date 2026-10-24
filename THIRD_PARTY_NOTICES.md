# Third-party notices

## Whistle model code (needle)

whistle-rust is a Rust translation of the Whistle model definition, quantization and archive
format in github.com/cactus-compute/needle (`needle/model/whistle.py`, `architecture.py`,
`quantize.py`, `export.py`), pinned in `REFERENCE`.

Copyright Cactus Compute, Inc. Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes: re-expressed in Rust as an inference-only
library; training, fine-tuning and export code are not included.

The oracle scripts in `oracle/` import that code (fetched into `needle-reference/`, not
redistributed here) to generate test fixtures.

## Whistle weights

`whistle.cact` (huggingface.co/Cactus-Compute/whistle, revision pinned in `REFERENCE`) is
Copyright Cactus Compute, Inc., Apache License 2.0. It is not stored in this repository;
`scripts/fetch-weights.sh` downloads it. Applications that ship it must include the Apache-2.0
license text and this notice.

Citation: "Whistle: Speech Recognition for Tiny Devices" (2026), Cactus Compute, Inc.

## Test fixtures

`test-data/fixtures/` holds tensors computed by the needle model code from the Whistle weights
(Apache-2.0, as above). `test-data/clips/` is synthesized speech generated for this project.
`test-data/engine/` holds transcripts produced by the Needle 3 engine binary
(huggingface.co/Cactus-Compute/needle3, Apache-2.0 model card), used only as a black-box
reference.
