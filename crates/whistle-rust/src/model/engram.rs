//! The decoder's Engram sites (`architecture.Engram`, driven by `Whistle.decode`): hashed n-gram
//! indices over the token sequence ([`engram_indices`]), each site's keys and values from its
//! packed tables ([`Engram::keys_values`]), and the gated add at the top of a decoder block
//! ([`engram_add`], `DecoderBlock`). The tables and projections are the [`Engram`] bound in
//! `weights`; the arithmetic is `ops`. The decoder stack calls these once per sequence (indices,
//! keys, values) and once per Engram layer (the add).

use crate::cact::Geometry;
use crate::error::Error;

use super::ops::{a8, rms_unit, sigmoid};
use super::weights::{Engram, EngramHash, Matrix};

/// The longest n-gram an archive may hash: Whistle uses orders 2 and 3, and the ceiling keeps
/// the per-position hash loop short whatever the header says.
pub const MAX_NGRAM_ORDER: u32 = 8;

/// `_ENGRAM_SEED`, the golden-ratio constant the per-table seeds are multiples of.
const SEED: u32 = 0x9E37_79B9;
/// `_ENGRAM_PRIME`, the 32-bit FNV prime.
const PRIME: u32 = 0x0100_0193;

/// `engram_indices`: for each position `t` and each table (orders major, then hash heads), the
/// slot `hash(tokens[t - order + 1 ..= t]) % slots`, returned `[T, orders · heads]` row-major.
///
/// Table `oi · heads + h` starts from `seed = SEED · (oi · stride + h + 1)` (`stride` is
/// `seed_heads`, or `heads` when 0), then for `j` in `0..order` folds in the token `j` places
/// back (0 before the sequence start) as `acc = (acc ^ token) · PRIME`, then `acc ^= acc >> 15`.
/// All arithmetic is u32 with wrapping multiply, as the reference's `uint32` arrays compute it.
/// Positions whose n-gram reaches before the start still get an index; the gather zeroes them
/// ([`ngram_in_sequence`]). A table count or output size that overflows `usize` is
/// [`Error::Input`].
pub fn engram_indices(
    tokens: &[u32],
    orders: &[u32],
    heads: usize,
    slots: u32,
    seed_heads: usize,
) -> Result<Vec<u32>, Error> {
    let stride = if seed_heads == 0 { heads } else { seed_heads };
    let tables = orders
        .len()
        .checked_mul(heads)
        .filter(|&t| tokens.len().checked_mul(t).is_some())
        .ok_or_else(|| {
            Error::Input(format!(
                "{} tokens × {} orders × {heads} heads of Engram indices overflow",
                tokens.len(),
                orders.len()
            ))
        })?;
    let mut out = vec![0u32; tokens.len() * tables];
    // A zero slot count has no valid index; every lookup then fails in `keys_values`.
    let slots = slots.max(1);
    for (oi, &order) in orders.iter().enumerate() {
        for h in 0..heads {
            // `(SEED * (oi·stride + h + 1)) & 0xFFFFFFFF` on Python ints is the product's low 32
            // bits, which depend only on the low 32 bits of each factor: the wraparound is
            // intended, so every step wraps (`as u32` keeps the low bits too).
            let k = (oi as u32)
                .wrapping_mul(stride as u32)
                .wrapping_add(h as u32)
                .wrapping_add(1);
            let seed = SEED.wrapping_mul(k);
            let table = oi * heads + h;
            for t in 0..tokens.len() {
                let mut acc = seed;
                for j in 0..order as usize {
                    let token = t.checked_sub(j).map_or(0, |p| tokens[p]);
                    acc = (acc ^ token).wrapping_mul(PRIME);
                }
                acc ^= acc >> 15;
                out[t * tables + table] = acc % slots;
            }
        }
    }
    Ok(out)
}

/// The hashing geometry from the archive header, checked against what `Whistle.decode` uses:
/// it builds every Engram with `conv_dilation = max(orders)` and hashes with `seed_heads = 0`,
/// which `engram_indices` reads as `seed_heads or heads`, so a header recording 0 or `heads`
/// hashes identically and any other value would not be Whistle. Each n-gram order must lie in
/// `1..=min(MAX_NGRAM_ORDER, max_seq_len)`. Called by `bind`; a header that does not fit is
/// [`Error::Archive`].
pub(crate) fn hash_geometry(g: &Geometry) -> Result<EngramHash, Error> {
    let orders = g.engram_orders.clone();
    let tables = g.num_engram_tables as usize;
    let Some(&max_order) = orders.iter().max() else {
        return Err(Error::Archive("Engram sites without n-gram orders".into()));
    };
    if tables == 0 || !tables.is_multiple_of(orders.len()) {
        return Err(Error::Archive(format!(
            "{tables} Engram tables do not split over orders {orders:?}"
        )));
    }
    // The hash folds `order` tokens per position and table, so an order is bounded by the
    // longest decoder input and by an absolute ceiling: a hostile header cannot make the loop
    // arbitrarily slow.
    let ceiling = MAX_NGRAM_ORDER.min(g.max_seq_len);
    if let Some(&order) = orders.iter().find(|&&o| o == 0 || o > ceiling) {
        return Err(Error::Archive(format!(
            "Engram n-gram order {order} is outside 1 to {ceiling} (the ceiling \
             {MAX_NGRAM_ORDER}, and max_seq_len {})",
            g.max_seq_len
        )));
    }
    let heads = tables / orders.len();
    let seed_heads = g.engram_seed_heads as usize;
    if seed_heads != 0 && seed_heads != heads {
        return Err(Error::Archive(format!(
            "engram_seed_heads {seed_heads} is not supported (Whistle hashes with 0, the same \
             as {heads}, the hash heads per order)"
        )));
    }
    if g.engram_conv_dilation != max_order {
        return Err(Error::Archive(format!(
            "engram_conv_dilation {} is not supported (Whistle uses max(orders) = {max_order})",
            g.engram_conv_dilation
        )));
    }
    Ok(EngramHash {
        orders,
        heads,
        slots: g.engram_slots,
        dilation: max_order as usize,
    })
}

/// Whether position `t`'s n-gram of `order` lies inside an unpadded causal sequence
/// (`_mask_diag(causal_mask, order - 1)`): true once `t ≥ order - 1`.
pub fn ngram_in_sequence(t: usize, order: usize) -> bool {
    t + 1 >= order
}

/// Whether value-conv tap `j` at `dilation` reads inside an unpadded causal sequence at
/// position `t` (`_mask_diag(causal_mask, j · dilation)`): true once `t ≥ j · dilation`.
/// A shift `j · dilation` past `usize` reads before any sequence, so it is false.
pub fn tap_in_sequence(t: usize, j: usize, dilation: usize) -> bool {
    j.checked_mul(dilation).is_some_and(|shift| t >= shift)
}

impl Engram {
    /// This site's keys and values for the decoder input `tokens` (`Engram.__call__` with the
    /// causal masks of `Whistle.decode`), each `[T, d_model]`. The site's four table rows per
    /// position are gathered (zeroed where the n-gram reaches before the sequence), joined into
    /// one row, fake-quantized to A8 and projected by `key_proj` and `value_proj`; the values
    /// then pass through the causal depthwise conv `Σ_j taps[j] · v[t − j · dilation]`.
    ///
    /// The hashing geometry is the site's own (`hash`, checked when bound). Tables, projections
    /// and taps whose shapes do not fit together (or that geometry) are [`Error::Archive`].
    pub fn keys_values(&self, tokens: &[u32]) -> Result<(Matrix, Matrix), Error> {
        let (k, v) = self.project_rows(tokens, 0)?;
        let conv = self.value_conv(&v, 0)?;
        let n = tokens.len();
        let keys = Matrix::dense(n, self.key_proj.rows, k);
        let values = Matrix::dense(n, self.value_proj.rows, conv);
        Ok((keys, values))
    }

    /// The site's shapes checked against its hashing geometry; returns `(tables, sub_dim)`.
    fn check_shapes(&self) -> Result<(usize, usize), Error> {
        let EngramHash {
            orders,
            heads,
            slots,
            ..
        } = &self.hash;
        let tables = orders.len().checked_mul(*heads).unwrap_or(0);
        let sub_dim = self.tables.cols();
        let width = tables.checked_mul(sub_dim);
        let d = self.value_proj.rows;
        if tables == 0 || tables.checked_mul(*slots as usize) != Some(self.tables.rows()) {
            return Err(Error::Archive(format!(
                "{} Engram tables of {slots} slots ({orders:?} × {heads}) do not fit {} table \
                 rows",
                orders.len().saturating_mul(*heads),
                self.tables.rows()
            )));
        }
        if width != Some(self.key_proj.cols)
            || width != Some(self.value_proj.cols)
            || self.key_proj.rows != d
            || self.taps.cols != d
        {
            return Err(Error::Archive(format!(
                "Engram shapes do not fit: {tables} tables of {sub_dim}, key_proj [{}, {}], \
                 value_proj [{d}, {}], taps [{}, {}]",
                self.key_proj.rows,
                self.key_proj.cols,
                self.value_proj.cols,
                self.taps.rows,
                self.taps.cols
            )));
        }
        Ok((tables, sub_dim))
    }

    /// The keys and the values before the value conv for positions `first..` of the decoder
    /// input `tokens` (the whole prefix, so each n-gram sees its earlier tokens), each
    /// `[T − first, d_model]`: the site's four table rows per position gathered (zeroed where
    /// the n-gram reaches before the sequence), joined, A8, then `key_proj` and `value_proj`.
    /// [`Engram::keys_values`] runs it from 0; a cached decoder step runs it for its one new
    /// position. Errors are [`Engram::keys_values`]'s.
    pub(crate) fn project_rows(
        &self,
        tokens: &[u32],
        first: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        Ok(self.project_gathered(self.gather_rows(tokens, first)?))
    }

    /// A8 then `key_proj` and `value_proj` on gathered rows ([`Engram::gather_rows`], any number
    /// of them, from one decode or several): each output row depends on its own row only, so a
    /// batched decoder step projects every hypothesis's new row in one pass.
    pub(crate) fn project_gathered(&self, mut e: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        a8(&mut e, self.key_proj.cols);
        (self.key_proj.project(&e), self.value_proj.project(&e))
    }

    /// The gathered table rows of positions `first..` of `tokens`, `[T − first, tables ·
    /// sub_dim]`, before A8 ([`Engram::project_rows`]'s first half). Errors are its.
    pub(crate) fn gather_rows(&self, tokens: &[u32], first: usize) -> Result<Vec<f32>, Error> {
        let (tables, sub_dim) = self.check_shapes()?;
        let EngramHash {
            orders,
            heads,
            slots,
            ..
        } = &self.hash;
        let (heads, slots_usize) = (*heads, *slots as usize);
        let width = tables * sub_dim;
        let indices = engram_indices(tokens, orders, heads, *slots, 0)?;
        let n = tokens.len();
        if first > n {
            return Err(Error::Input(format!(
                "Engram rows from {first} of {n} tokens"
            )));
        }
        let gathered = (n - first)
            .checked_mul(width)
            .ok_or_else(|| Error::Input(format!("{n} tokens of Engram rows overflow")))?;
        // Gathered rows, `[T − first, tables · sub_dim]`: table `t`'s slice is its row times
        // ngram_ok.
        let mut e = vec![0.0f32; gathered];
        for t in first..n {
            let r = t - first;
            for table in 0..tables {
                let order = orders[table / heads] as usize;
                if !ngram_in_sequence(t, order) {
                    continue;
                }
                let row = table * slots_usize + indices[t * tables + table] as usize;
                let out = &mut e[r * width + table * sub_dim..r * width + (table + 1) * sub_dim];
                self.tables.row_into(row, out)?;
            }
        }
        Ok(e)
    }

    /// The causal value conv `Σ_j taps[j] · v[t − j · dilation]` for positions `first..` of
    /// `v` (`[T, d_model]`, every position's projected values from 0), `[T − first, d_model]`.
    /// A `v` that is not whole rows, or `first` past it, is [`Error::Input`].
    pub(crate) fn value_conv(&self, v: &[f32], first: usize) -> Result<Vec<f32>, Error> {
        let d = self.value_proj.rows;
        if d == 0 || !v.len().is_multiple_of(d) || first > v.len() / d {
            return Err(Error::Input(format!(
                "Engram values of {} from position {first} are not rows of {d}",
                v.len()
            )));
        }
        let n = v.len() / d;
        let taps = &self.taps;
        let dilation = self.hash.dilation;
        // `sum(taps[j] * shift_right(v, j · dilation) * tap_ok[j])`: Python's `sum` starts at 0
        // and adds the taps in order, which this keeps so the f32 sums round the same way.
        let tap_values = taps.data();
        let mut conv = vec![0.0f32; (n - first) * d];
        for t in first..n {
            let out = &mut conv[(t - first) * d..(t - first + 1) * d];
            for j in 0..taps.rows {
                let tap = &tap_values[j * d..(j + 1) * d];
                // `tap_in_sequence` holds exactly when the shift neither overflows nor reaches
                // before the start.
                let src = j
                    .checked_mul(dilation)
                    .and_then(|shift| t.checked_sub(shift));
                for (c, o) in out.iter_mut().enumerate() {
                    let shifted = src.map_or(0.0, |s| v[s * d + c]);
                    *o += tap[c] * shifted;
                }
            }
        }
        Ok(conv)
    }
}

/// The Engram add at the top of `DecoderBlock`: with every site's keys and values
/// (`[sites, T, width]` each) and the layer's site flags (`[sites]`, 1 for the site at this
/// layer, else 0, as `LaneStack` builds them),
/// `x + Σ_s flag_s · σ(⟨rms(x), rms(k_s)⟩ / √width) · v_s` per position. Shapes that do not
/// agree are [`Error::Input`].
pub fn engram_add(
    x: &[f32],
    keys: &[f32],
    values: &[f32],
    site_flags: &[f32],
    width: usize,
) -> Result<Vec<f32>, Error> {
    let sites = site_flags.len();
    let per_site = x.len();
    if width == 0 || !x.len().is_multiple_of(width) {
        return Err(Error::Input(format!(
            "{} block values are not whole rows of {width}",
            x.len()
        )));
    }
    if keys.len() != sites * per_site || values.len() != sites * per_site {
        return Err(Error::Input(format!(
            "Engram keys ({}) and values ({}) are not {sites} sites of {per_site} values",
            keys.len(),
            values.len()
        )));
    }
    let n = x.len() / width;
    let scale = (width as f32).sqrt();
    let xu = rms_unit(x, width);
    // Σ over sites first, then added to x, as the reference's einsum then `x +` order it.
    let mut add = vec![0.0f32; x.len()];
    for (s, &flag) in site_flags.iter().enumerate() {
        if flag == 0.0 {
            continue;
        }
        let site = s * per_site..(s + 1) * per_site;
        let ku = rms_unit(&keys[site.clone()], width);
        let v = &values[site];
        for t in 0..n {
            let row = t * width..(t + 1) * width;
            let dot: f32 = xu[row.clone()]
                .iter()
                .zip(&ku[row.clone()])
                .map(|(a, b)| a * b)
                .sum();
            let alpha = sigmoid(dot / scale);
            for (a, &vv) in add[row.clone()].iter_mut().zip(&v[row]) {
                *a += flag * alpha * vv;
            }
        }
    }
    Ok(x.iter().zip(&add).map(|(a, b)| a + b).collect())
}
