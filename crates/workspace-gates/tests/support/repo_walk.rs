// The repository walk the file gates share: `file_compliance.rs` and `markdown_placement.rs` both
// pull this file in with `#[path]`, so the two can never disagree about what the repository is.
//
// The walk starts at the workspace root and reads the file system, not git's index, so a file an
// agent has just written counts before it is committed. Skipped trees are build output, VCS data
// and third-party material that is not ours: every Cargo target directory (any directory holding
// the `CACHEDIR.TAG` Cargo writes into each one, wherever CARGO_TARGET_DIR pointed), target/ next
// to a Cargo.toml or anywhere under .claude/, .git/, node_modules/, demo/web/pkg/,
// demo/web/dist/, .claude/worktrees/, and the downloaded or generated trees: needle-reference/
// (upstream's code), models/ (weights), oracle/.venv/ and oracle/.engine/ (the oracle's Python
// and the engine binary) and test-data/fixtures-local/ (regenerated tensors). A `target`
// directory elsewhere is source and is walked, as are real sources under .claude/. Symbolic
// links are neither followed nor listed (a worktree's `models` is one).

use std::fs;
use std::path::{Path, PathBuf};

/// The workspace root: this crate sits at `crates/workspace-gates`.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/workspace-gates sits two levels below the workspace root")
        .to_path_buf()
}

/// True for a directory (relative to the root, `/`-separated) that the walk skips by name.
/// `beside_cargo_toml` says whether its parent directory holds a `Cargo.toml`: there, and
/// anywhere under `.claude/` (agent tooling, where a stray `CARGO_TARGET_DIR` build lands and
/// no source is called `target`), a `target` directory is Cargo's build output rather than a
/// source directory that happens to be called `target`.
pub fn is_skipped_dir(relative: &str, beside_cargo_toml: bool) -> bool {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    (name == "target" && (beside_cargo_toml || relative.starts_with(".claude/")))
        || matches!(name, ".git" | "node_modules")
        || matches!(
            relative,
            "demo/web/pkg"
                | "demo/web/dist"
                | ".claude/worktrees"
                | "needle-reference"
                | "models"
                | "oracle/.venv"
                | "oracle/.engine"
                | "test-data/fixtures-local"
        )
}

/// True for a Cargo target directory, whatever it is called and wherever it sits: Cargo writes
/// a `CACHEDIR.TAG` into every target directory it builds into.
fn holds_cachedir_tag(dir: &Path) -> bool {
    dir.join("CACHEDIR.TAG").is_file()
}

/// Every file under `dir` that the walk keeps, appended to `out` in directory order (callers
/// sort). `root` is what [`is_skipped_dir`]'s relative paths are measured from.
pub fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        let file_type = fs::symlink_metadata(&path).expect("metadata").file_type();
        if file_type.is_dir() {
            let beside_cargo_toml = dir.join("Cargo.toml").is_file();
            if !is_skipped_dir(&relative(root, &path), beside_cargo_toml)
                && !holds_cachedir_tag(&path)
            {
                collect_files(root, &path, out);
            }
        } else if file_type.is_file() {
            out.push(path);
        }
    }
}

/// `path` relative to `root`, `/`-separated on every platform.
pub fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("under the root")
        .to_string_lossy()
        .replace('\\', "/")
}

/// Every file the walk keeps under `root`, relative and `/`-separated, sorted.
pub fn repo_files(root: &Path) -> Vec<String> {
    let mut paths = Vec::new();
    collect_files(root, root, &mut paths);
    let mut files: Vec<String> = paths.iter().map(|p| relative(root, p)).collect();
    files.sort();
    files
}

/// A scratch directory tree under the system temp folder, removed when dropped.
pub struct TempTree(pub PathBuf);

impl TempTree {
    pub fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "workspace-gates-{label}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create the temp tree");
        TempTree(dir)
    }

    pub fn write(&self, relative: &str, text: &str) {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().expect("a parent")).expect("create the parent");
        fs::write(&path, text).expect("write the file");
    }
}

impl Drop for TempTree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
