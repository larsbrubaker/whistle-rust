//! Tests for where the model comes from (`model.rs`): a command-line path first and alone, a
//! download whose bytes are not the pinned model rejected and never cached, a good download
//! cached and then read from the cache, and a damaged cached copy deleted. Downloads come from a
//! one-shot HTTP server on localhost; the good model is the checkout's `models/whistle.cact`
//! (`scripts/fetch-weights.sh`).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Mutex;

use super::*;
use crate::test_support::Scratch;

/// The pinned model from the checkout.
fn model_bytes() -> Vec<u8> {
    std::fs::read(checkout_model()).expect("models/whistle.cact (run scripts/fetch-weights.sh)")
}

/// Serve `body` once at `http://127.0.0.1:<port>/whistle.cact`.
fn serve_once(body: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        if let Ok((mut socket, _)) = listener.accept() {
            // Read the request head (it ends with a blank line).
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") && socket.read(&mut byte).unwrap_or(0) == 1 {
                head.push(byte[0]);
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes());
            let _ = socket.write_all(&body);
        }
    });
    format!("http://127.0.0.1:{port}/whistle.cact")
}

/// A URL nothing listens on (the port was free a moment ago).
fn dead_url() -> String {
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("free port")
        .port();
    format!("http://127.0.0.1:{port}/whistle.cact")
}

fn source(cache: &std::path::Path, url: String) -> ModelSource {
    ModelSource {
        path: None,
        checkout: None,
        cache_dir: Some(cache.to_path_buf()),
        url,
    }
}

/// A progress report nobody reads.
fn quiet() -> impl Fn(u64, u64, &str) {
    |_, _, _| {}
}

#[test]
fn a_command_line_path_is_loaded_before_anything_else() {
    let cache = Scratch::new("path-first");
    let mut s = source(cache.path(), dead_url());
    s.path = Some(checkout_model());
    let (bytes, from) = s.load(&quiet()).expect("the path loads");
    assert_eq!(bytes.len() as u64, MODEL_BYTES);
    assert_eq!(from, checkout_model().display().to_string());
    assert!(
        !s.cache_file().expect("cache file").exists(),
        "nothing downloaded"
    );
}

#[test]
fn a_bad_command_line_path_is_an_error_not_a_download() {
    let cache = Scratch::new("path-bad");
    let wrong = cache.path().join("not-the-model.cact");
    std::fs::write(&wrong, b"not a model").expect("write");
    let reports = Mutex::new(0);
    let mut s = source(cache.path(), serve_once(model_bytes()));
    s.path = Some(wrong.clone());
    let err = s
        .load(&|_, _, _| *reports.lock().unwrap() += 1)
        .expect_err("a wrong file is refused");
    assert!(err.contains(&wrong.display().to_string()), "{err}");
    assert!(err.contains("not the expected whistle.cact"), "{err}");
    assert_eq!(*reports.lock().unwrap(), 0, "no download was tried");
}

#[test]
fn a_download_with_the_wrong_checksum_is_rejected_and_not_cached() {
    let cache = Scratch::new("bad-checksum");
    // The right size, so only the SHA-256 tells it from the model.
    let mut corrupted = model_bytes();
    corrupted[1000] ^= 0xff;
    let s = source(cache.path(), serve_once(corrupted));
    let err = s.load(&quiet()).expect_err("a corrupted model is refused");
    assert!(err.contains("checksum mismatch"), "{err}");
    assert!(!s.cache_file().expect("cache file").exists());
}

#[test]
fn a_download_larger_than_the_model_is_refused() {
    let cache = Scratch::new("too-large");
    let s = source(
        cache.path(),
        serve_once(vec![0u8; MODEL_BYTES as usize + 10]),
    );
    let err = s.load(&quiet()).expect_err("too large");
    assert!(err.contains("larger than whistle.cact"), "{err}");
}

#[test]
fn a_good_download_is_cached_and_the_next_start_reads_the_cache() {
    let cache = Scratch::new("cached");
    let url = serve_once(model_bytes());
    let last = Mutex::new((0, 0));
    let s = source(cache.path(), url.clone());
    let (bytes, from) = s
        .load(&|loaded, total, _| *last.lock().unwrap() = (loaded, total))
        .expect("downloaded");
    assert_eq!(from, url);
    assert_eq!(bytes.len() as u64, MODEL_BYTES);
    assert_eq!(*last.lock().unwrap(), (MODEL_BYTES, MODEL_BYTES));
    let file = s.cache_file().expect("cache file");
    assert!(file.exists());
    let dir = std::fs::read_dir(file.parent().expect("dir")).expect("listed");
    assert_eq!(dir.count(), 1, "the cached model alone, no part file left");

    // The server is gone; the cache answers.
    let again = source(cache.path(), dead_url());
    let (_, from) = again.load(&quiet()).expect("from the cache");
    assert_eq!(from, "cache");
}

#[test]
fn a_damaged_cached_copy_is_deleted() {
    let cache = Scratch::new("damaged");
    let s = source(cache.path(), dead_url());
    let file = s.cache_file().expect("cache file");
    std::fs::create_dir_all(file.parent().expect("dir")).expect("mkdir");
    std::fs::write(&file, b"half a model").expect("write");
    let err = s.load(&quiet()).expect_err("no good copy anywhere");
    assert!(err.contains("127.0.0.1"), "the download was tried: {err}");
    assert!(!file.exists(), "the damaged copy is gone");
}

#[test]
fn the_checkout_copy_is_used_before_the_cache_and_the_download() {
    let cache = Scratch::new("checkout");
    let mut s = source(cache.path(), dead_url());
    s.checkout = Some(checkout_model());
    let (_, from) = s.load(&quiet()).expect("the checkout copy");
    assert_eq!(from, checkout_model().display().to_string());
}
