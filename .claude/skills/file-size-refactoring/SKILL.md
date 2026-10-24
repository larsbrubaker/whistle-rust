---
name: file-size-refactoring
description: Guidance for fixing a file-size violation reported by the `file_compliance` test. Use when a Rust source file (or script) exceeds the 800-line limit. Explains how to reduce file size by decomposition while keeping code quality.
---

# File Size Refactoring

The `file_compliance` test (`crates/workspace-gates/tests/file_compliance.rs`) fails when any source file in
the workspace has more than **800 non-empty lines**. Comments, code, braces and `use` lines all
count; blank and whitespace-only lines do not. There are no exemptions and none will be added.

## Why the limit exists

A smaller file is easier for a human to understand, navigate and maintain. A file past ~800
meaningful lines has almost always accumulated too many responsibilities. Hitting the limit is a
signal that the file deserves structural attention, and the answer is always to **decompose it
into smaller, cohesive pieces**, never to compress the code to squeeze under the number.

## Valid approaches

1. **Remove dead code** — unused functions, commented-out blocks, unreachable branches.
2. **Extract by responsibility** — free functions that don't need `self` move to a sibling
   module; data transforms, validation, and I/O each get their own module.
3. **Extract by feature** — group functions that implement one thing (e.g. `sift/orientation.rs`,
   `sift/descriptor.rs`) under a directory module.
4. **Composition** — when methods lean on shared state, extract a collaborator struct the
   original owns and delegates to. Each extracted type is a nameable concept with its own tests.
5. **Split `impl` blocks along responsibility boundaries** — Rust allows several `impl Foo` blocks
   in different modules of the same crate. This is acceptable when extraction would force private fields `pub`, but prefer a real type. Keep
   fields `pub(crate)` or narrower and say why in the new file's header.
6. **Move tests out** — a large `#[cfg(test)] mod tests` becomes `foo/tests.rs`
   (`#[cfg(test)] mod tests;`) or an integration test under `tests/`.

## How to evaluate a split

- Can the new module have a clear, purposeful name? If not, the split is artificial.
- Does it represent one cohesive concept, not "the second half of the file"?
- Why did the file grow? Understanding that leads to a better decomposition.

## What NOT to do

- Don't remove blank lines, comments or whitespace, and don't join statements or compress
  formatting. That trades readability for a number, which defeats the purpose.
- Don't add an exemption — there is no exemption list.
- Don't treat small overages differently: 801 lines needs the same structural thinking as 1200.

## After refactoring

1. `cargo test -p workspace-gates --test file_compliance`
2. Every new file starts with its header comment (what it is and how it
   relates to its neighbors), and the old file's header is updated to match.
