// Test area: the committed oracle fixtures themselves (test-data/fixtures/). Every later stage
// test reads them, so this area checks they are complete and readable before any of those run.

#[path = "support/mod.rs"]
mod support;

#[path = "oracle_fixtures/committed_fixtures.rs"]
mod committed_fixtures;
