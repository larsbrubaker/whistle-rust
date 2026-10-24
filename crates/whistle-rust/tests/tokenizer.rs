// Test area: the tokenizer (`whistle::tokenizer`) against upstream's `RefTokenizer`, through the
// archive fixture's encode and decode cases (test-data/fixtures/archive/, oracle/dump_archive.py),
// plus synthetic blobs for the merge, marker and byte-fallback rules.

#[path = "support/mod.rs"]
mod support;

#[path = "tokenizer/malformed.rs"]
mod malformed;
#[path = "tokenizer/reference_cases.rs"]
mod reference_cases;
#[path = "tokenizer/synthetic.rs"]
mod synthetic;
