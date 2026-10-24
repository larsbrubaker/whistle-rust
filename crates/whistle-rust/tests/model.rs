// Test area: the model (`whistle::model`): binding the archive's tensors by role against the
// archive oracle fixture, and the ported stages (stem, encoder input, ZC-RMSNorm, Hadamard MLP)
// and the encoder (block sub-modules, each layer, end to end)
// against the clips' stage fixtures (test-data/fixtures/, oracle/dump.py).

#[path = "support/mod.rs"]
mod support;

#[path = "model/binding.rs"]
mod binding;
#[path = "model/encoder_block.rs"]
mod encoder_block;
#[path = "model/encoder_stack.rs"]
mod encoder_stack;
#[path = "model/hadamard.rs"]
mod hadamard;
#[path = "model/shared.rs"]
mod shared;
#[path = "model/stem.rs"]
mod stem;
