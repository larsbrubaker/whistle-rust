//! What the desktop demo's tests share (`engine_tests.rs`, `bridge_tests.rs`, `model_tests.rs`,
//! `shell_host.rs`): the test clips from the checkout's `test-data/clips/`, what the one they
//! stream says, and scratch directories that delete themselves.

use std::path::{Path, PathBuf};

/// What every reading of `en_samantha` says (the library's snapshot and the engine's transcript).
pub const SAMANTHA: &str = "Make the cube 20 mm wide and round the top edges.";

/// The bytes of `test-data/clips/<name>.wav`.
pub fn clip(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/clips")
        .join(format!("{name}.wav"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A fresh, empty directory for one test, deleted with everything in it when dropped (the
/// model tests put a 17 MB model in theirs).
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("whistle-demo-native-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        Scratch(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
