// Diagnostic: FNV-1a hashes of the f32 bits of each encoder-side stage (features, stem, encoder
// output) for one WAV, so a native build and a wasm32 build can be compared stage by stage
// (scripts/wasm-test.sh runs the suite; this finds where two targets first part ways).
//
//   cargo run --release -p whistle-rust --example stage_hashes -- models/whistle.cact clip.wav
use whistle::audio::{features, read_wav};
use whistle::cact::Archive;
use whistle::model::{Encoder, Weights};

fn hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let w = Weights::from_archive(&Archive::parse(&bytes).unwrap()).unwrap();
    let samples = read_wav(&std::fs::read(&args[2]).unwrap()).unwrap().samples;
    let feats = features(&samples);
    println!("features {:016x}", hash(&feats.data));
    let stem = w.stem.forward(&feats).unwrap();
    println!("stem {:016x}", hash(&stem.data));
    let enc = Encoder::new(&w).forward(&feats).unwrap();
    println!("encoder {:016x}", hash(&enc.data));
}
