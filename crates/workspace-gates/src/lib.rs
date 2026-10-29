//! workspace-gates: a test-only crate. Its gates live in `tests/`: `file_compliance.rs` (the
//! 800-non-empty-line limit, header comments and merge-conflict markers across the whole
//! repository, CLAUDE.md "Testing framework") and `markdown_placement.rs` (Markdown only where
//! it is not documentation of code, CLAUDE.md "Documentation"), both over the repository walk
//! in `tests/support/repo_walk.rs`. There is deliberately no library code here.
