// Test area: the `.cact` archive reader and dequantization (`whistle::cact`, `whistle::quant`)
// against the archive oracle fixture (test-data/fixtures/archive/, oracle/dump_archive.py), plus
// synthetic archives for the CQ widths whistle.cact does not use and for malformed input.

#[path = "support/mod.rs"]
mod support;

#[path = "archive/dequantize.rs"]
mod dequantize;
#[path = "archive/directory.rs"]
mod directory;
#[path = "archive/malformed.rs"]
mod malformed;
#[path = "archive/widths.rs"]
mod widths;
