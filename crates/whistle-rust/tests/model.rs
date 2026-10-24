// Test area: the model (`whistle::model`): binding the archive's tensors by role against the
// archive oracle fixture, and the ported stages (stem, encoder input, ZC-RMSNorm, Hadamard MLP)
// and the encoder (block sub-modules, each layer, end to end)
// against the clips' stage fixtures (test-data/fixtures/, oracle/dump.py).
// The decoder's inputs (embedding, memory, Engram hash, keys, values and add) are tested in
// `decoder_input` and `engram`; its block sub-modules in `decoder_block`, each layer, the output
// and the logits in `decoder_stack`.

#[path = "support/mod.rs"]
mod support;

#[path = "model/binding.rs"]
mod binding;
#[path = "model/decoder_block.rs"]
mod decoder_block;
#[path = "model/decoder_input.rs"]
mod decoder_input;
#[path = "model/decoder_stack.rs"]
mod decoder_stack;
#[path = "model/encoder_block.rs"]
mod encoder_block;
#[path = "model/encoder_stack.rs"]
mod encoder_stack;
#[path = "model/engram.rs"]
mod engram;
#[path = "model/hadamard.rs"]
mod hadamard;
#[path = "model/shared.rs"]
mod shared;
#[path = "model/stem.rs"]
mod stem;
