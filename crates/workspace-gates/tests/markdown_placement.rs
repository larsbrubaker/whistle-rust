// Markdown placement gate (CLAUDE.md, "Documentation"). Documentation lives in the code, in the
// header comment of the file, module or crate it describes, where whoever works on that code
// reads it. A Markdown file is allowed only where it is not documentation of code:
//
//   - `README.md`, `CLAUDE.md`, `THIRD_PARTY_NOTICES.md` and `PORTING_PLAN.md` (the plan for
//     open work) at the repository root;
//   - `crates/whistle-rust/README.md` and `crates/whistle-rust/THIRD_PARTY_NOTICES.md`, which
//     ship in the published crate (its crates.io and docs.rs page, and its notices);
//   - anything under `.claude/` (agent instructions and skills).
//
// Every other `.md` or `.markdown` file (any case) fails the gate, named with what to do about
// it. The files are those `support/repo_walk.rs` walks, the same as `file_compliance.rs`'s, and
// untracked files count, so a new doc fails before it is committed.

#[path = "support/repo_walk.rs"]
mod repo_walk;

use repo_walk::{repo_files, workspace_root, TempTree};

/// The Markdown files allowed by exact path.
const ALLOWED_FILES: [&str; 6] = [
    "README.md",
    "CLAUDE.md",
    "THIRD_PARTY_NOTICES.md",
    "PORTING_PLAN.md",
    "crates/whistle-rust/README.md",
    "crates/whistle-rust/THIRD_PARTY_NOTICES.md",
];

/// The trees whose Markdown files are all allowed.
const ALLOWED_TREES: [&str; 1] = [".claude/"];

/// What the gate makes of one repository path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// Not a Markdown file: the gate has nothing to say about it.
    NotMarkdown,
    /// A Markdown file in one of the allowed places.
    Allowed,
    /// A Markdown file anywhere else.
    Refused,
}

/// True for a Markdown file by its extension, whatever its case.
fn is_markdown(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => {
            ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown")
        }
        _ => false,
    }
}

/// Where `path` (relative to the root, `/`-separated) stands.
fn placement(path: &str) -> Placement {
    if !is_markdown(path) {
        return Placement::NotMarkdown;
    }
    let under = |tree: &str| path.strip_prefix(tree).is_some_and(|rest| !rest.is_empty());
    if ALLOWED_FILES.contains(&path) || ALLOWED_TREES.iter().any(|tree| under(tree)) {
        Placement::Allowed
    } else {
        Placement::Refused
    }
}

/// The gate's failure message for the refused paths, empty when there are none.
fn failure_message(refused: &[&str]) -> String {
    if refused.is_empty() {
        return String::new();
    }
    let mut message = format!(
        "{} Markdown file(s) outside the places Markdown may live:\n",
        refused.len()
    );
    for path in refused {
        message.push_str(&format!("  {path}\n"));
    }
    message.push_str(
        "Documentation lives in the code. Move what it explains into the header or doc comments \
         of the code it is about (a rule spanning several files into the owning crate's root; \
         measurements onto the test or bench that makes them), then delete it. Open work goes \
         in PORTING_PLAN.md (CLAUDE.md, \"Documentation\").\n",
    );
    message
}

/// The refused paths among the files the walk keeps under `root`.
fn misplaced(root: &std::path::Path) -> Vec<String> {
    repo_files(root)
        .into_iter()
        .filter(|path| placement(path) == Placement::Refused)
        .collect()
}

#[test]
fn markdown_lives_only_in_the_allowed_places() {
    let refused = misplaced(&workspace_root());
    let refused: Vec<&str> = refused.iter().map(String::as_str).collect();
    let message = failure_message(&refused);
    assert!(message.is_empty(), "{message}");
}

#[test]
fn markdown_gate_sees_the_allowed_files() {
    // Guards against a walk that silently checks nothing (wrong root, over-broad skips).
    let files = repo_files(&workspace_root());
    for expected in [
        "CLAUDE.md",
        "README.md",
        "PORTING_PLAN.md",
        "crates/whistle-rust/README.md",
    ] {
        assert!(
            files.iter().any(|p| p == expected),
            "the walk missed {expected}"
        );
    }
    assert!(files.iter().any(|p| p.starts_with(".claude/agents/")));
}

#[test]
fn markdown_placement_allows_the_listed_places() {
    for path in [
        "README.md",
        "CLAUDE.md",
        "THIRD_PARTY_NOTICES.md",
        "PORTING_PLAN.md",
        "crates/whistle-rust/README.md",
        "crates/whistle-rust/THIRD_PARTY_NOTICES.md",
        ".claude/agents/implementer.md",
        ".claude/skills/file-size-refactoring/SKILL.md",
    ] {
        assert_eq!(placement(path), Placement::Allowed, "{path}");
    }
}

#[test]
fn markdown_placement_refuses_docs_anywhere_else() {
    for path in [
        "docs/design/streaming.md",
        "docs/notes.md",
        "demo/README.md",
        "demo/whistle-demo/README.md",
        "crates/workspace-gates/README.md",
        "crates/whistle-rust/src/NOTES.md",
        "crates/whistle-rust/docs/README.md",
        "crates/README.md",
        "oracle/README.md",
        "scripts/README.md",
        "test-data/README.md",
        "CONTRIBUTING.md",
        // An allowed name in the wrong place is still refused.
        "demo/PORTING_PLAN.md",
        "crates/whistle-rust/CLAUDE.md",
        // `.claude` itself is a name, not the tree.
        ".claude.md",
    ] {
        assert_eq!(placement(path), Placement::Refused, "{path}");
    }
    // Not Markdown, so not this gate's concern.
    for path in [
        "crates/x/src/lib.rs",
        "docs/demo.png",
        "README",
        ".md",
        "docs/md",
    ] {
        assert_eq!(placement(path), Placement::NotMarkdown, "{path}");
    }
}

#[test]
fn markdown_placement_matches_extensions_in_any_case() {
    for path in [
        "notes.MD",
        "notes.Md",
        "notes.markdown",
        "demo/guide.MARKDOWN",
    ] {
        assert_eq!(placement(path), Placement::Refused, "{path}");
    }
    // The allowed names are exact: another case is another file.
    assert_eq!(placement("readme.md"), Placement::Refused);
    assert_eq!(placement(".claude/agents/X.MD"), Placement::Allowed);
}

#[test]
fn markdown_gate_reads_untracked_files_and_skips_build_trees() {
    let tree = TempTree::new("markdown");
    tree.write("Cargo.toml", "[workspace]\n");
    tree.write("README.md", "readme\n");
    tree.write(".claude/agents/implementer.md", "agent\n");
    tree.write("docs/design/old.md", "a design doc\n");
    tree.write("crates/x/NOTES.md", "a stray doc\n");
    // Build output, downloaded and third-party trees are never walked.
    for skipped in [
        "target/doc/README.md",
        "node_modules/pkg/README.md",
        "demo/web/node_modules/pkg/README.md",
        "demo/web/pkg/README.md",
        "demo/web/dist/README.md",
        "needle-reference/README.md",
        "models/README.md",
        "oracle/.venv/lib/README.md",
        "test-data/fixtures-local/README.md",
        ".claude/worktrees/step/notes.md",
        ".git/info/notes.md",
        "build-out/CACHEDIR.TAG",
        "build-out/README.md",
    ] {
        tree.write(skipped, "skipped\n");
    }
    assert_eq!(
        misplaced(&tree.0),
        ["crates/x/NOTES.md", "docs/design/old.md"]
    );
}

#[test]
fn markdown_gate_failure_message_says_what_to_do() {
    let message = failure_message(&["docs/x.md"]);
    assert!(message.contains("  docs/x.md\n"));
    assert!(message.contains("into the header or doc comments of the code it is about"));
    assert!(message.contains("PORTING_PLAN.md"));
    assert!(failure_message(&[]).is_empty());
}
