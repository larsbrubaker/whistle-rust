// Test area: the model (`whistle::model`): binding the archive's tensors by role against the
// archive oracle fixture, and the ported stages (stem, encoder input, ZC-RMSNorm, Hadamard MLP)
// against the clips' stage fixtures (test-data/fixtures/, oracle/dump.py).

#[path = "support/mod.rs"]
mod support;

#[path = "model/binding.rs"]
mod binding;
#[path = "model/hadamard.rs"]
mod hadamard;
#[path = "model/shared.rs"]
mod shared;
#[path = "model/stem.rs"]
mod stem;
