#!/usr/bin/env bash
# Checks that the whistle-rust crate as published builds on its own (CLAUDE.md "Commands";
# run by scripts/gates.sh, .github/workflows/ci.yml and publish.yml before `cargo publish`).
#
# Packages the crate with `cargo package` (only what Cargo.toml's `include` lists), unpacks the
# .crate into a temp dir outside the workspace, and there, with its own target dir: builds it,
# documents it (a broken or private intra-doc link is an error) and compiles its doctests, the README's usage examples.
# Fails if any step fails or the .crate is over 1 MB. The pinned toolchain from
# rust-toolchain.toml is used throughout, since the unpacked crate does not carry that file.
set -euo pipefail
cd "$(dirname "$0")/.."
max_bytes=1048576
channel=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
if [ -z "$channel" ]; then
  echo "no channel = \"...\" line in rust-toolchain.toml" >&2
  exit 1
fi
export RUSTUP_TOOLCHAIN=$channel
tmp=$(mktemp -d)
# Leave $tmp before removing it (the build runs inside it), and never let cleanup fail the run.
trap 'cd / && rm -rf "$tmp" || true' EXIT

# --allow-dirty: the gates run before a change is committed. --no-verify: the build below is
# the verification, done outside the workspace so nothing from it leaks in.
cargo package -p whistle-rust --allow-dirty --no-verify --target-dir "$tmp/package-target"
crates=("$tmp"/package-target/package/whistle-rust-*.crate)
crate=${crates[0]}
size=$(wc -c <"$crate" | tr -d ' ')
echo "package: $(basename "$crate"), $size bytes (limit $max_bytes)"
if [ "$size" -gt "$max_bytes" ]; then
  echo "package is over 1 MB: check Cargo.toml's include list" >&2
  exit 1
fi

mkdir "$tmp/unpacked"
tar -xzf "$crate" -C "$tmp/unpacked"
dir=$(find "$tmp/unpacked" -mindepth 1 -maxdepth 1 -type d)
cd "$dir"
# --target-dir as an argument, not CARGO_TARGET_DIR: Windows git-bash rewrites POSIX paths in
# arguments for native programs such as cargo.
target="$tmp/build-target"
cargo build --locked --target-dir "$target"
RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links -D rustdoc::private_intra_doc_links" cargo doc --locked --no-deps --target-dir "$target"
cargo test --locked --doc --target-dir "$target"
echo "package ok"
