//! Where the desktop demo's engine thread (`engine.rs`) gets `whistle.cact`, checked against the
//! size and SHA-256 the library pins (`whistle::model_file`) before use:
//!
//! 1. a path given on the command line (`--model`): only that file, so a wrong path is an
//!    error rather than a silent download;
//! 2. otherwise the checkout's `models/whistle.cact` (what `scripts/fetch-weights.sh` writes),
//!    then the cache, `<cache dir>/whistle-rust/<sha256>/whistle.cact` (another model version
//!    never reads this one; a bad cached copy is deleted), then a download from
//!    `whistle::model_file::MODEL_URL` into the cache.
//!
//! The download goes through ureq over rustls with the RustCrypto provider (pure Rust). A
//! server silent for [`STALL`] fails the download; a body larger than the model is refused.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};
use whistle::model_file::{MODEL_BYTES, MODEL_SHA256, MODEL_URL};

/// A download silent this long (no connection, no answer, no bytes) has stalled.
const STALL: Duration = Duration::from_secs(20);

/// The places to look, in order.
#[derive(Debug, Clone)]
pub struct ModelSource {
    /// The command line's file: when set, the only place.
    pub path: Option<PathBuf>,
    /// The checkout's copy.
    pub checkout: Option<PathBuf>,
    /// The download cache's root (the model sits under its SHA-256).
    pub cache_dir: Option<PathBuf>,
    /// Where to download from when nothing local is good.
    pub url: String,
}

/// Download progress: bytes so far, the total, and where from.
pub type Progress<'a> = &'a dyn Fn(u64, u64, &str);

impl ModelSource {
    /// The desktop demo's places: `path` if given, else the checkout, the cache and the URL.
    pub fn new(path: Option<PathBuf>) -> ModelSource {
        ModelSource {
            path,
            checkout: Some(checkout_model()),
            cache_dir: platform_cache_dir(),
            url: MODEL_URL.to_string(),
        }
    }

    fn cache_file(&self) -> Option<PathBuf> {
        self.cache_dir
            .as_ref()
            .map(|d| d.join(MODEL_SHA256).join("whistle.cact"))
    }

    /// The checked model and where it came from, or why there is none (every place tried).
    pub fn load(&self, progress: Progress) -> Result<(Vec<u8>, String), String> {
        if let Some(path) = &self.path {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            check(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            return Ok((bytes, path.display().to_string()));
        }
        let mut failures = Vec::new();
        // No checkout copy is the usual case away from a development checkout: not reported.
        if let Some(path) = &self.checkout {
            if let Ok(bytes) = std::fs::read(path) {
                match check(&bytes) {
                    Ok(()) => return Ok((bytes, path.display().to_string())),
                    Err(e) => failures.push(format!("{}: {e}", path.display())),
                }
            }
        }
        let cached = self.cache_file();
        if let Some(file) = &cached {
            if let Ok(bytes) = std::fs::read(file) {
                if check(&bytes).is_ok() {
                    return Ok((bytes, "cache".into()));
                }
                // A partial or damaged copy: drop it so the download below replaces it.
                let _ = std::fs::remove_file(file);
            }
        }
        match download(&self.url, progress).and_then(|bytes| {
            check(&bytes).map_err(|e| format!("{}: {e}", self.url))?;
            Ok(bytes)
        }) {
            Ok(bytes) => {
                if let Some(file) = &cached {
                    save(file, &bytes);
                }
                Ok((bytes, self.url.clone()))
            }
            Err(e) => {
                failures.push(e);
                Err(failures.join("; "))
            }
        }
    }
}

/// `models/whistle.cact` in the whistle-rust checkout this binary was built from.
pub fn checkout_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/whistle.cact")
}

/// The platform's cache directory for the demo: `~/Library/Caches`, `%LOCALAPPDATA%`, or
/// `$XDG_CACHE_HOME` (else `~/.cache`), under `whistle-rust/`.
fn platform_cache_dir() -> Option<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches"))
    } else if cfg!(windows) {
        env("LOCALAPPDATA")
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))
    };
    base.map(|b| b.join("whistle-rust"))
}

/// `bytes` is the pinned model: the right size and SHA-256.
pub fn check(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() as u64 != MODEL_BYTES {
        return Err(format!(
            "{} bytes, not the expected whistle.cact ({MODEL_BYTES} bytes)",
            bytes.len()
        ));
    }
    let digest = Sha256::digest(bytes);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    if hex != MODEL_SHA256 {
        return Err(format!(
            "checksum mismatch: SHA-256 {hex}, not the expected whistle.cact's {MODEL_SHA256}"
        ));
    }
    Ok(())
}

/// Write `bytes` to `file` through a temporary name, so a crash never leaves half a model where
/// the cache is read; the name carries the process id, so two copies of the demo downloading
/// at once never write the same file. A failure only costs the next start a download.
fn save(file: &Path, bytes: &[u8]) {
    let Some(dir) = file.parent() else {
        return;
    };
    let partial = file.with_extension(format!("{}.part", std::process::id()));
    let saved = std::fs::create_dir_all(dir)
        .and_then(|_| std::fs::write(&partial, bytes))
        .and_then(|_| std::fs::rename(&partial, file));
    if let Err(e) = saved {
        eprintln!("whistle-demo-native: could not cache the model: {e}");
        let _ = std::fs::remove_file(&partial);
    }
}

/// An agent that gives up on a name lookup or a server silent for [`STALL`] (ureq 3 has no
/// timeouts of its own).
/// The body has no total limit: a slow link that keeps delivering may take as long as it needs.
fn agent() -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder()
        .provider(ureq::tls::TlsProvider::Rustls)
        .unversioned_rustls_crypto_provider(Arc::new(rustls_rustcrypto::provider()))
        .root_certs(ureq::tls::RootCerts::WebPki)
        .build();
    ureq::Agent::config_builder()
        .tls_config(tls)
        .timeout_resolve(Some(STALL))
        .timeout_connect(Some(STALL))
        .timeout_recv_response(Some(STALL))
        .timeout_per_read(Some(STALL))
        .build()
        .into()
}

/// GET `url`, reporting progress against the model's pinned size (not `Content-Length`, which
/// counts encoded bytes when the body is compressed); refuses more bytes than the model has.
fn download(url: &str, progress: Progress) -> Result<Vec<u8>, String> {
    let mut response = agent().get(url).call().map_err(|e| format!("{url}: {e}"))?;
    // usize is at least 32 bits and the model is 16.9 MB.
    let expected = MODEL_BYTES as usize;
    let mut reader = response.body_mut().with_config().limit(u64::MAX).reader();
    let mut bytes = Vec::with_capacity(expected);
    let mut buf = vec![0u8; 64 * 1024];
    let mut reported = 0;
    progress(0, MODEL_BYTES, url);
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            break;
        }
        if bytes.len() + n > expected {
            return Err(format!("{url} is larger than whistle.cact"));
        }
        bytes.extend_from_slice(&buf[..n]);
        if bytes.len() - reported >= 256 * 1024 {
            reported = bytes.len();
            progress(bytes.len() as u64, MODEL_BYTES, url);
        }
    }
    progress(bytes.len() as u64, MODEL_BYTES, url);
    Ok(bytes)
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
