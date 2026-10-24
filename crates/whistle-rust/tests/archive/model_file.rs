// The pinned model file constants (`whistle::model_file`) against the archive the tests run on
// (models/whistle.cact, fetched by scripts/fetch-weights.sh) and against the repository's own
// pins (REFERENCE "whistle-weights", scripts/weights.sha256), so the URL, digest and size a
// caller ships with are the ones every other test in this repo was run against.

use whistle::model_file::{MODEL_BYTES, MODEL_SHA256, MODEL_URL};

use crate::support::{model_path, repo_root, sha256::sha256_hex};

fn read_repo_file(relative: &str) -> String {
    let path = repo_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn sha256_matches_the_fips_180_examples() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // Two blocks: the padding spills into a second block.
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn sha256_handles_the_padding_boundaries() {
    // 55 bytes: the largest message whose padding and length still fit in one block. 64 bytes:
    // exactly one block of data, so the padding is a block of its own. Expected values from
    // `shasum -a 256`.
    assert_eq!(
        sha256_hex(&[b'a'; 55]),
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
    );
    assert_eq!(
        sha256_hex(&[b'a'; 64]),
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
    );
}

#[test]
fn downloaded_archive_has_the_pinned_size_and_sha256() {
    let path = model_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/fetch-weights.sh first)",
            path.display()
        )
    });
    assert_eq!(bytes.len() as u64, MODEL_BYTES);
    assert_eq!(sha256_hex(&bytes), MODEL_SHA256);
}

#[test]
fn constants_match_the_repository_pins() {
    let reference = read_repo_file("REFERENCE");
    let revision = reference
        .lines()
        .find_map(|line| line.strip_prefix("whistle-weights "))
        .expect("REFERENCE pins whistle-weights")
        .trim();
    assert_eq!(
        MODEL_URL,
        format!("https://huggingface.co/Cactus-Compute/whistle/resolve/{revision}/whistle.cact")
    );
    let sums = read_repo_file("scripts/weights.sha256");
    let pinned = sums
        .lines()
        .find_map(|line| line.trim_end().strip_suffix("  models/whistle.cact"))
        .expect("scripts/weights.sha256 lists models/whistle.cact");
    assert_eq!(MODEL_SHA256, pinned);
}
