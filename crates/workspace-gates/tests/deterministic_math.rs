// Deterministic-math gate (CLAUDE.md "Testing framework", net 2): the library computes every
// transcendental through `whistle-rust/src/math` (pure-Rust ports that round identically on
// every target), never through std's `exp`, `ln`, `sin`, ..., which call the platform libm and
// differ in the last bit between macOS, glibc, MSVC and wasm. Scans every `.rs` file under
// crates/whistle-rust/src/ except `math` itself and the unit-test files (`tests.rs`,
// `*_tests.rs`), which may use std as a reference. A use is a method call `.name(` or `::name` followed
// by a non-identifier character, so calls and function paths (`map(f32::sin)`) both count. `sqrt`, `floor`, `abs` and `powi` (exact in
// IEEE 754, or plain multiplies) are allowed.

use std::fs;
use std::path::{Path, PathBuf};

/// Method and path forms of std's libm-backed functions.
const BANNED: [&str; 23] = [
    "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10", "powf", "sin", "cos", "tan",
    "sin_cos", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "cbrt", "hypot", "erf",
];

fn library_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("whistle-rust/src")
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n != "math")
                || path.parent() != Some(&library_src())
            {
                collect(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn is_exempt(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name == "tests.rs" || name.ends_with("_tests.rs") || path == library_src().join("math.rs")
}

/// `line` with its `//` comment removed and string literals blanked, so neither a comment nor a
/// string ("http://...", "use .exp()") hides code or counts as code. A `'"'` char literal is
/// skipped whole; block comments are not used in the library.
fn code_of(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let (mut out, mut i, mut in_str) = (String::new(), 0, false);
    while i < chars.len() {
        let c = chars[i];
        if in_str {
            if c == '\\' {
                i += 1;
            } else if c == '"' {
                in_str = false;
                out.push('"');
            }
        } else if c == '"' {
            in_str = true;
            out.push('"');
        } else if c == '\'' && chars.get(i + 1) == Some(&'"') && chars.get(i + 2) == Some(&'\'') {
            i += 2;
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            break;
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

/// The banned names used on one line: a method call `.name(`, or `::name` followed by anything
/// but an identifier character (a call, or a function path such as `map(f32::exp)`), except on
/// the `math::` path (the deterministic functions share std's names).
fn banned_calls(line: &str) -> Vec<&'static str> {
    let code = code_of(line);
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    BANNED
        .iter()
        .copied()
        .filter(|name| {
            [".", "::"].iter().any(|sep| {
                let pat = format!("{sep}{name}");
                code.match_indices(&pat).any(|(i, _)| {
                    let next = code[i + pat.len()..].chars().next();
                    let before = &code[..i];
                    if *sep == "." {
                        // A method call; `.cos` without `(` is a field (the RoPE table).
                        next == Some('(')
                    } else {
                        !next.is_some_and(ident) && !before.ends_with("math")
                    }
                })
            })
        })
        .collect()
}

#[test]
fn the_library_uses_no_std_transcendental_outside_math() {
    let mut files = Vec::new();
    collect(&library_src(), &mut files);
    assert!(files.len() > 20, "the scan found the library sources");
    let mut found = Vec::new();
    for path in files.iter().filter(|p| !is_exempt(p)) {
        let text = fs::read_to_string(path).unwrap();
        for (i, line) in text.lines().enumerate() {
            for name in banned_calls(line) {
                found.push(format!(
                    "{}:{}: `{name}`: use crate::math",
                    path.display(),
                    i + 1
                ));
            }
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn the_scan_catches_method_and_path_forms() {
    assert_eq!(banned_calls("let y = (x - max).exp();"), ["exp"]);
    assert_eq!(banned_calls("let y = f64::ln(x);"), ["ln"]);
    assert_eq!(banned_calls("out.extend(angles.map(f32::exp));"), ["exp"]);
    assert_eq!(
        banned_calls("let s = \"http://x\"; let y = v.sin();"),
        ["sin"]
    );
    assert_eq!(
        banned_calls("let y = crate::math::exp(x) + math::ln(x);"),
        Vec::<&str>::new()
    );
    assert_eq!(
        banned_calls("let n = rope.cos.len() + x.sine();"),
        Vec::<&str>::new()
    );
    assert_eq!(
        banned_calls("let s = \"use .exp() here\";"),
        Vec::<&str>::new()
    );
    assert_eq!(
        banned_calls("let y = x.sqrt().powi(2); // .exp( in a comment"),
        Vec::<&str>::new()
    );
}
