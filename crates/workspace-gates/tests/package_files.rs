// Package-files gate: the whistle-rust crate packages its own copies of the root LICENSE and
// THIRD_PARTY_NOTICES.md (cargo only packages files inside the crate directory, and Apache-2.0
// requires the license and notices to travel with the code). This keeps each copy identical to
// the root file, so editing one without the other fails the tests.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/workspace-gates sits two levels below the workspace root")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    // Line endings are normalized so a CRLF checkout compares equal.
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n")
}

#[test]
fn crate_license_and_notices_match_the_root_files() {
    let root = workspace_root();
    for name in ["LICENSE", "THIRD_PARTY_NOTICES.md"] {
        let original = read(&root.join(name));
        let copy = read(&root.join("crates/whistle-rust").join(name));
        assert!(
            original == copy,
            "crates/whistle-rust/{name} differs from the root {name}: copy the root file over it"
        );
    }
}
