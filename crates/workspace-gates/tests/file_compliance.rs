// Workspace-wide file compliance gate (CLAUDE.md, "Testing framework"). Walks the whole
// repository from the workspace root and checks every source file:
//
//   (a) at most 800 non-empty (non-whitespace) lines — `.rs .wgsl .py .sh .ts .js .yml .yaml`;
//   (b) starts with a header comment — `//` or `/*` for rs/wgsl/ts/js (covers `//!`), `#` for
//       py/sh/yml/yaml, where a `#!` shebang line must be followed by a `#` comment line;
//   (c) no merge-conflict markers — in every UTF-8 text file, Markdown included.
//
// No exemption list, ever: a file over the limit is split by responsibility (the
// `file-size-refactoring` skill), never squeezed. Skipped trees are build output, VCS data
// and third-party material that is not ours: every Cargo target directory (any directory
// holding the `CACHEDIR.TAG` Cargo writes into each one, wherever CARGO_TARGET_DIR pointed),
// target/ next to a Cargo.toml or anywhere under .claude/, .git/, node_modules/, demo/web/pkg/,
// demo/web/dist/, .claude/worktrees/, and the downloaded or generated trees: needle-reference/
// (upstream's code), models/ (weights), oracle/.venv/ and oracle/.engine/ (the oracle's Python
// and the engine binary) and test-data/fixtures-local/ (regenerated tensors). A `target`
// directory elsewhere is source and is checked, as are real sources under .claude/.
//
// A file with a checked source extension that cannot be read or is not UTF-8 fails the scan
// rather than being skipped; other non-UTF-8 files (images, fonts) are binary and ignored.

use std::fs;
use std::path::{Path, PathBuf};

const MAX_NON_EMPTY_LINES: usize = 800;
const SOURCE_EXTENSIONS: [&str; 8] = ["rs", "wgsl", "py", "sh", "ts", "js", "yml", "yaml"];
const REFACTOR_HINT: &str =
    "Split each file by responsibility; see the `file-size-refactoring` skill. No exemptions.";

fn workspace_root() -> PathBuf {
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
fn is_skipped_dir(relative: &str, beside_cargo_toml: bool) -> bool {
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

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
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

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("under the root")
        .to_string_lossy()
        .replace('\\', "/")
}

fn extension(path: &Path) -> &str {
    path.extension().and_then(|e| e.to_str()).unwrap_or("")
}

fn has_header_comment(ext: &str, text: &str) -> bool {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    match ext {
        "py" | "sh" | "yml" | "yaml" => {
            if first.starts_with("#!") {
                lines.next().is_some_and(|l| l.starts_with('#'))
            } else {
                first.starts_with('#')
            }
        }
        _ => first.starts_with("//") || first.starts_with("/*"),
    }
}

fn is_conflict_marker(line: &str) -> bool {
    let line = line.trim_end_matches('\r');
    line.starts_with("<<<<<<< ") || line == "=======" || line.starts_with(">>>>>>> ")
}

struct Scan {
    files: Vec<(String, String)>,
}

/// Decides what the scan does with one file's contents: `Ok(Some(text))` to check it,
/// `Ok(None)` to ignore it (a binary file without a checked source extension), or `Err` when a
/// file with a checked source extension cannot be read or is not UTF-8 — those must fail the
/// gate, never slip past it. Other unreadable or binary files are ignored.
fn load_text(ext: &str, bytes: std::io::Result<Vec<u8>>) -> Result<Option<String>, String> {
    let is_source = SOURCE_EXTENSIONS.contains(&ext);
    match bytes {
        Err(e) if is_source => Err(format!("unreadable: {e}")),
        // Not a checked source file: nothing to hold it to beyond conflict markers.
        Err(_) => Ok(None),
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => Ok(Some(text)),
            Err(_) if is_source => Err("not valid UTF-8".to_string()),
            // Binary files (images, fonts) have no lines to check.
            Err(_) => Ok(None),
        },
    }
}

fn scan() -> Scan {
    let root = workspace_root();
    let mut paths = Vec::new();
    collect_files(&root, &root, &mut paths);
    paths.sort();
    let mut files = Vec::new();
    let mut failures = Vec::new();
    for p in paths {
        let path = relative(&root, &p);
        match load_text(extension(&p), fs::read(&p)) {
            Ok(Some(text)) => files.push((path, text)),
            Ok(None) => {}
            Err(why) => failures.push(format!("  {path}: {why}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} file(s) could not be checked:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Scan { files }
}

fn source_files(scan: &Scan) -> impl Iterator<Item = &(String, String)> {
    scan.files
        .iter()
        .filter(|(path, _)| SOURCE_EXTENSIONS.contains(&extension(Path::new(path))))
}

#[test]
fn source_files_are_at_most_800_non_empty_lines() {
    let scan = scan();
    let offenders: Vec<String> = source_files(&scan)
        .filter_map(|(path, text)| {
            let count = text.lines().filter(|l| !l.trim().is_empty()).count();
            (count > MAX_NON_EMPTY_LINES).then(|| format!("  {path}: {count} non-empty lines"))
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "{} file(s) exceed {MAX_NON_EMPTY_LINES} non-empty lines:\n{}\n{REFACTOR_HINT}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn source_files_start_with_a_header_comment() {
    let scan = scan();
    let offenders: Vec<String> = source_files(&scan)
        .filter(|(path, text)| !has_header_comment(extension(Path::new(path)), text))
        .map(|(path, _)| format!("  {path}"))
        .collect();
    assert!(
        offenders.is_empty(),
        "{} file(s) do not start with a header comment (what the file is and \
         how it relates to its neighbors):\n{}\n{REFACTOR_HINT}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn no_merge_conflict_markers() {
    let scan = scan();
    let offenders: Vec<String> = scan
        .files
        .iter()
        .filter_map(|(path, text)| {
            let count = text.lines().filter(|l| is_conflict_marker(l)).count();
            (count > 0).then(|| format!("  {path}: {count} conflict marker line(s)"))
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "{} file(s) contain merge-conflict markers:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn compliance_scan_sees_the_workspace() {
    // Guards against a walk that silently checks nothing (wrong root, over-broad skips).
    let scan = scan();
    let paths: Vec<&str> = scan.files.iter().map(|(p, _)| p.as_str()).collect();
    for expected in [
        "crates/workspace-gates/src/lib.rs",
        "crates/workspace-gates/tests/file_compliance.rs",
        "crates/whistle-rust/src/lib.rs",
        "oracle/dump.py",
        "scripts/gates.sh",
        "CLAUDE.md",
    ] {
        assert!(paths.contains(&expected), "scan missed {expected}");
    }
    assert!(!paths.iter().any(|p| p.starts_with("needle-reference/")));
    assert!(!paths.iter().any(|p| p.starts_with("oracle/.venv/")));
    assert!(!paths.iter().any(|p| p.contains("target/")));
}

#[test]
fn compliance_rules_classify_examples() {
    assert!(has_header_comment("rs", "//! crate docs\nfn main() {}"));
    assert!(has_header_comment("wgsl", "/* shader */"));
    assert!(!has_header_comment("rs", "\n// late header"));
    assert!(!has_header_comment("js", "const x = 1;"));
    assert!(has_header_comment(
        "sh",
        "#!/usr/bin/env bash\n# what it does\n"
    ));
    assert!(!has_header_comment("sh", "#!/usr/bin/env bash\nset -e\n"));
    assert!(has_header_comment("py", "# generator\nimport os\n"));
    assert!(has_header_comment("yml", "# CI\nname: ci\n"));
    assert!(!has_header_comment("yaml", "name: ci\n"));
    assert!(is_conflict_marker("<<<<<<< HEAD"));
    assert!(is_conflict_marker("======="));
    assert!(is_conflict_marker(">>>>>>> branch\r"));
    assert!(!is_conflict_marker("========"));
    assert!(!is_conflict_marker("a <<<<<<< b"));
    assert!(is_skipped_dir("crates/whistle-rust/target", true));
    assert!(is_skipped_dir("needle-reference", false));
    assert!(is_skipped_dir("oracle/.venv", false));
    assert!(is_skipped_dir("test-data/fixtures-local", false));
    assert!(!is_skipped_dir("test-data/fixtures", false));
    assert!(is_skipped_dir("target", true));
    assert!(is_skipped_dir("web/node_modules", false));
    assert!(is_skipped_dir("demo/web/pkg", false));
    assert!(is_skipped_dir("demo/web/dist", false));
    // The demo is its own Cargo workspace; its build output is skipped, its sources checked.
    assert!(is_skipped_dir("demo/target", true));
    assert!(!is_skipped_dir("demo/web/src", false));
    assert!(!is_skipped_dir("demo/whistle-demo/src", true));
    assert!(is_skipped_dir(".claude/worktrees", false));
    assert!(is_skipped_dir(".claude/target", false));
    assert!(is_skipped_dir(".claude/x/target", false));
    assert!(!is_skipped_dir(".claude/agents", false));
    assert!(!is_skipped_dir("crates/whistle-rust/src", true));
}

#[test]
fn compliance_only_skips_target_beside_cargo_toml() {
    // A `target` directory that is not Cargo's build output is source and must be checked.
    assert!(!is_skipped_dir("crates/whistle-rust/src/target", false));
    assert!(!is_skipped_dir("web/tests/target", false));
}

/// A scratch directory tree under the system temp folder, removed when dropped.
struct TempTree(PathBuf);

impl TempTree {
    fn new(label: &str) -> Self {
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

    fn write(&self, relative: &str, text: &str) {
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

#[test]
fn compliance_walk_skips_cargo_build_output() {
    // A coverage run once built into `.claude/target`, and the scan then failed on the generated
    // `.rs` files there. Cargo output is skipped wherever it sits; real sources are still walked.
    let tree = TempTree::new("walk");
    let checked = [
        "Cargo.toml",
        "src/lib.rs",
        ".claude/hooks/check.sh",
        // Not Cargo output: no CACHEDIR.TAG, not beside a Cargo.toml, not under .claude/.
        "web/tests/target/fixture.js",
    ];
    for path in checked {
        tree.write(path, "// header\n");
    }
    for path in [
        "target/debug/build/out.rs",
        ".claude/x/target/debug/build/generated.rs",
        ".claude/x/target/CACHEDIR.TAG",
        ".claude/cov/CACHEDIR.TAG",
        ".claude/cov/debug/build/generated.rs",
        "build-out/CACHEDIR.TAG",
        "build-out/debug/generated.rs",
    ] {
        tree.write(path, "no header\n");
    }
    // A `target` under .claude/ without a CACHEDIR.TAG (a partial or foreign build) is skipped
    // too: nothing named `target` there is ours.
    tree.write(".claude/y/target/release/generated.rs", "no header\n");

    let mut paths = Vec::new();
    collect_files(&tree.0, &tree.0, &mut paths);
    let mut found: Vec<String> = paths.iter().map(|p| relative(&tree.0, p)).collect();
    found.sort();
    let mut expected: Vec<String> = checked.iter().map(|p| p.to_string()).collect();
    expected.sort();
    assert_eq!(found, expected);
}

#[test]
fn compliance_fails_on_unreadable_or_non_utf8_sources() {
    let invalid = vec![0x2f, 0x2f, 0xff, 0xfe];
    let denied = || Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
    assert!(load_text("rs", Ok(invalid.clone())).is_err());
    assert!(load_text("py", denied()).is_err());
    assert_eq!(load_text("png", denied()), Ok(None));
    // Binary files without a checked extension are ignored, text files are checked.
    assert_eq!(load_text("png", Ok(invalid)), Ok(None));
    assert_eq!(
        load_text("rs", Ok(b"// header\n".to_vec())),
        Ok(Some("// header\n".to_string()))
    );
}
