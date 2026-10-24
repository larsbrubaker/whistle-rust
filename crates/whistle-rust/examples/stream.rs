// Streams a WAV file through `Whistle::stream` in fixed chunks at one thread and prints each
// update (committed text, pending tail, compute time) and the median and max update cost. Builds
// natively and for wasm32-wasip1, so scripts/wasm-bench.mjs can time it under Node.
//
//   cargo run --release --example stream -- <model.cact> <clip.wav> [chunk seconds] [beams]

use std::time::Instant;

use whistle::audio::read_wav;
use whistle::{Options, Whistle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, model, wav, rest @ ..] = args.as_slice() else {
        return Err("usage: stream <model.cact> <clip.wav> [chunk seconds] [beams]".into());
    };
    let chunk: f32 = rest.first().map_or(Ok(1.0), |s| s.parse())?;
    let beams: usize = rest.get(1).map_or(Ok(1), |s| s.parse())?;
    whistle::set_threads(1);
    let whistle = Whistle::from_cact(&std::fs::read(model)?)?;
    let audio = read_wav(&std::fs::read(wav)?)?;
    let options = Options {
        beams,
        ..Options::default()
    };
    let mut stream = whistle.stream(&options)?;
    let size = ((chunk * 16_000.0) as usize).max(1);
    let mut ms = Vec::new();
    for piece in audio.samples.chunks(size) {
        let started = Instant::now();
        let update = stream.push(piece)?;
        ms.push(started.elapsed().as_secs_f64() * 1000.0);
        println!(
            "{:7.1} ms  {:?}  pending {:?}",
            ms[ms.len() - 1],
            update.text,
            update.pending
        );
    }
    let started = Instant::now();
    let update = stream.finish()?;
    ms.push(started.elapsed().as_secs_f64() * 1000.0);
    println!("{:7.1} ms  {:?}  (finish)", ms[ms.len() - 1], update.text);
    ms.sort_by(f64::total_cmp);
    println!(
        "updates {}  median {:.1} ms  max {:.1} ms",
        ms.len(),
        ms[ms.len() / 2],
        ms[ms.len() - 1]
    );
    Ok(())
}
