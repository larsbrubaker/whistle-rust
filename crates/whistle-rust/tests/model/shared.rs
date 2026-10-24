// Shared state for the model test area: the model's weights, bound once from
// models/whistle.cact and reused by every test in the binary.

use std::sync::OnceLock;

use whistle::cact::Archive;
use whistle::model::Weights;

use crate::support::archive::model_bytes;

pub fn weights() -> &'static Weights {
    static WEIGHTS: OnceLock<Weights> = OnceLock::new();
    WEIGHTS.get_or_init(|| {
        let bytes = model_bytes();
        let archive = Archive::parse(&bytes).expect("whistle.cact parses");
        Weights::from_archive(&archive).expect("whistle.cact binds")
    })
}
