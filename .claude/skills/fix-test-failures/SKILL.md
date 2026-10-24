---
name: fix-test-failures
description: "This skill should be used after running tests when failures occur. It provides a systematic approach to diagnosing test failures through instrumentation and root cause analysis."
---

# Fix Test Failures

This skill provides a systematic approach to diagnosing and fixing test failures.

## The Goal

When a test fails, the goal is to **understand what went wrong before changing anything.** A test failure is valuable information -- it's telling you something about the system that you didn't expect. The worst thing you can do is silence that signal without understanding it.

Most of the time, a failing test reveals a real bug in the production code. Occasionally, the test itself has a problem -- an incorrect assumption, a setup issue, or a requirement that genuinely changed. Either way, the path forward is the same: investigate until you understand, then make the right fix.

## When to Use This Skill

- Tests fail and the cause isn't immediately obvious
- A test is flaky or intermittently failing
- You need to understand why a test is failing before fixing it
- You've made changes and tests are now failing

## The Process

### Step 1: Run Tests and Capture Failures

Run the failing test(s) to see the current error:

```bash
# Run all tests
cargo test --workspace

# Run with detailed output
cargo test --workspace -- --nocapture

# Run one test area
cargo test --workspace --test <area>

# Run a specific test method
cargo test -p <crate> <test_name> -- --exact --nocapture
```

> Name filters are substring matches; add `-- --exact` to run exactly one test.

Record the exact error message and stack trace. This is your starting point.

### Step 2: Understand What the Test Expects

Before adding instrumentation, read the test carefully:

1. What behavior is this test validating?
2. What assertion is failing?
3. What values were expected vs. received?
4. Form a hypothesis about what might be wrong

### Step 3: Add Strategic Instrumentation

Add `eprintln!` / `dbg!` statements (run with `--nocapture`) to expose state at key points. The goal is to see what's actually happening inside the code, not just what the test reports.

**For state-related failures:**
```rust
eprintln!("State before operation: {:?}");
// ... operation ...
eprintln!("State after operation: {:?}");
```

**For object inspection:**
```rust
eprintln!("rows: {:?}");
eprintln!("relative L2 vs fixture: {:?}");
```

**For execution flow:**
```rust
eprintln!("Entering {:?} with: {:?}");
// ... method body ...
eprintln!("Returning: {:?}");
```

### Step 4: Run Instrumented Tests

Run the test again with verbose output:

```bash
cargo test -p <crate> <test_name> -- --exact --nocapture
```

Look at the output to understand:
- What values are actually present
- Where execution diverges from expectations
- What state is incorrect and when it became incorrect

### Step 5: Identify the Root Cause

Based on instrumentation output, determine what's actually wrong:

- **Bug in production code** (most common) -- the code doesn't do what it should
- **Test assumption is incorrect** (rare) -- the test expected something that was never the right behavior
- **Requirements changed** -- the code intentionally changed and the test needs to reflect the new expected behavior
- **Numerical drift** -- A8 rounding flips (expected, bounded) versus a formula error (a bug)
- **Environment issue** -- file paths, missing resources, platform differences

### Step 6: Make the Right Fix

What you fix depends on what you found:

- **Production bug**: Fix the code so it produces the correct behavior. This is the most common case.
- **Incorrect test**: If the test itself was wrong (wrong expected value, flawed setup), fix the test. But be confident -- if you're not sure whether the test or the code is wrong, assume the code is wrong and investigate further.
- **Changed requirements**: Update the test to reflect the new correct behavior. This is different from weakening a test -- you're updating it because the definition of "correct" changed.

Common production code fixes:
- **Logic errors**: Fix the algorithm or condition
- **State issues**: Ensure proper initialization or cleanup
- **Numerics**: Check the formula against the reference line (axis, epsilon, rounding mode)
- **File path issues**: Resolve fixtures from `CARGO_MANIFEST_DIR`, never the working directory

### Step 7: Verify and Clean Up

1. Run the test again to confirm it passes
2. Run the full test suite to ensure no regressions: `cargo test --workspace`
3. **Remove all instrumentation** -- the debug output was for diagnosis only
4. Commit the fix

## Common Pitfalls

These approaches might feel like they solve the problem, but they hide it instead:

- **Weakening an assertion** to make it pass means the test no longer validates what it was designed to check. The bug is still there, just undetected.
- **Swallowing errors** (`.ok()`, `let _ =`) means errors happen silently. Users will hit them even if tests don't.
- **Mocking away the behavior being tested** turns the test into a tautology -- it only proves the mock works, not the real code.
- **Using `#[ignore]` permanently** means the test exists but protects nothing. If it's not worth fixing, it's not worth keeping.

These aren't forbidden -- they're just counterproductive. If you find yourself reaching for one, it usually means you haven't found the root cause yet.

## Iterative Debugging

If the first round of instrumentation doesn't reveal the issue:

1. Add more instrumentation at earlier points in execution
2. Log intermediate values, not just final state
3. Check for side effects from other code
4. Verify the test loads the right clip and stage
5. Check if the issue is environment-specific (file paths, platform)

Keep iterating until the root cause is clear. The goal is understanding, then fixing.

## Project Test Structure

```
crates/<crate>/tests/<area>.rs     # one test binary per area, pulling in tests/<area>/<file>.rs
crates/<crate>/tests/support/      # shared test support (fixture loader, .npy reader, tolerances)
crates/<crate>/src/**/tests.rs     # unit tests that need private access
test-data/fixtures/<clip>/         # committed oracle fixtures (manifest.json + .npy)
test-data/fixtures-local/<clip>/   # git-ignored per-layer internals, from oracle/dump.py
```

**Key patterns:**
- Stage tests load a clip's fixture and compare with the reference by relative L2 under
  CLAUDE.md's tolerance policy; a layer test starts from the previous layer's fixture.
- When a stage drifts, compare against the finer-grained internals in
  `test-data/fixtures-local/` (regenerate them with `oracle/dump.py` if absent) to find the
  first sub-step that diverges. A drift is a bug until the numbers prove it is A8 rounding.
- Assertions: plain `assert!`/`assert_eq!` with a message naming the clip, stage and error.
