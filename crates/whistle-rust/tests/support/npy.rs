// The .npy reader the oracle fixtures need (test support only): NumPy format versions 1.0-3.0,
// little-endian, C order, dtypes float32, int32 and uint8 (what oracle/dump.py writes).

use std::fs;
use std::path::Path;

/// One array: its shape and its values widened to the element type it was stored as.
#[derive(Debug, Clone, PartialEq)]
pub enum Npy {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I32 { shape: Vec<usize>, data: Vec<i32> },
    U8 { shape: Vec<usize>, data: Vec<u8> },
}

impl Npy {
    pub fn shape(&self) -> &[usize] {
        match self {
            Npy::F32 { shape, .. } | Npy::I32 { shape, .. } | Npy::U8 { shape, .. } => shape,
        }
    }

    /// The dtype name NumPy prints (`float32`, `int32`, `uint8`), as the manifests record it.
    pub fn dtype(&self) -> &'static str {
        match self {
            Npy::F32 { .. } => "float32",
            Npy::I32 { .. } => "int32",
            Npy::U8 { .. } => "uint8",
        }
    }
}

pub fn read(path: &Path) -> Result<Npy, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse(bytes: &[u8]) -> Result<Npy, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => {
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (len as usize, 12)
        }
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = std::str::from_utf8(bytes.get(start..start + header_len).ok_or("short header")?)
        .map_err(|_| "header is not UTF-8")?;
    if field(header, "fortran_order")? != "False" {
        return Err("Fortran-order arrays are not supported".into());
    }
    let descr = field(header, "descr")?.trim_matches('\'').to_string();
    let shape = parse_shape(field(header, "shape")?)?;
    let count: usize = shape.iter().product();
    let body = &bytes[start + header_len..];
    let need = |size: usize| {
        if body.len() == count * size {
            Ok(())
        } else {
            Err(format!(
                "{} data bytes for {count} elements of {size}",
                body.len()
            ))
        }
    };
    match descr.as_str() {
        "<f4" => {
            need(4)?;
            let data = body
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            Ok(Npy::F32 {
                shape,
                data: data.collect(),
            })
        }
        "<i4" => {
            need(4)?;
            let data = body
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            Ok(Npy::I32 {
                shape,
                data: data.collect(),
            })
        }
        "|u1" => {
            need(1)?;
            Ok(Npy::U8 {
                shape,
                data: body.to_vec(),
            })
        }
        other => Err(format!("unsupported dtype {other}")),
    }
}

/// The text of one `'key': value` entry of the header dict, up to the next top-level comma.
fn field<'a>(header: &'a str, key: &str) -> Result<&'a str, String> {
    let at = header
        .find(&format!("'{key}':"))
        .ok_or(format!("header has no {key}"))?;
    let rest = header[at + key.len() + 3..].trim_start();
    let mut depth = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' | '}' if depth == 0 => return Ok(rest[..i].trim()),
            _ => {}
        }
    }
    Err(format!("unterminated {key}"))
}

fn parse_shape(text: &str) -> Result<Vec<usize>, String> {
    let inner = text
        .trim()
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'));
    let inner = inner.ok_or(format!("bad shape {text}"))?;
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| format!("bad shape {text}")))
        .collect()
}
