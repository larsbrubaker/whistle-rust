//! Double-double (about 106-bit) references for `math`'s f64 tests: `ln`, `log10`, `exp` and
//! `pow` evaluated far more precisely than f64, so `tests.rs` can measure the port's error
//! against the true value instead of against the platform libm (which is itself up to about an
//! ulp off, sometimes on the other side). Built from exact error-free transformations (`two_sum`
//! and an FMA-based `two_prod`) and series whose truncation error is below 2⁻¹¹⁰; the only
//! constant is `ln 2`, which is computed here too.

/// `hi + lo` with `|lo| ≤ ½ ulp(hi)`: `hi` is the value rounded to f64.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dd {
    pub(super) hi: f64,
    lo: f64,
}

fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    Dd {
        hi: s,
        lo: (a - (s - bb)) + (b - bb),
    }
}

fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    Dd {
        hi: s,
        lo: b - (s - a),
    }
}

fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    Dd {
        hi: p,
        lo: a.mul_add(b, -p),
    }
}

impl Dd {
    pub(super) fn from(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    fn add(self, b: Dd) -> Dd {
        let s = two_sum(self.hi, b.hi);
        let t = two_sum(self.lo, b.lo);
        let u = quick_two_sum(s.hi, s.lo + t.hi);
        quick_two_sum(u.hi, u.lo + t.lo)
    }

    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    fn mul(self, b: Dd) -> Dd {
        let p = two_prod(self.hi, b.hi);
        quick_two_sum(p.hi, p.lo + (self.hi * b.lo + self.lo * b.hi))
    }

    fn div(self, b: Dd) -> Dd {
        // Long division: three f64 quotient digits.
        let q1 = self.hi / b.hi;
        let r = self.add(b.mul(Dd::from(q1)).neg());
        let q2 = r.hi / b.hi;
        let r = r.add(b.mul(Dd::from(q2)).neg());
        let q3 = r.hi / b.hi;
        quick_two_sum(q1, q2).add(Dd::from(q3))
    }

    /// `self · 2ⁿ`, exact while the result stays normal.
    fn scale(self, n: i32) -> Dd {
        // Two factors so 2ⁿ itself never leaves the normal range.
        let half = n / 2;
        let f = |e: i32| f64::from_bits(((1023 + e) as u64) << 52);
        Dd {
            hi: self.hi * f(half) * f(n - half),
            lo: self.lo * f(half) * f(n - half),
        }
    }
}

/// `2 · atanh(t) = 2 · Σ t^(2i+1) / (2i+1)`, for `|t| ≤ 1/3`.
fn two_atanh(t: Dd) -> Dd {
    let u = t.mul(t);
    let mut acc = Dd::from(0.0);
    for i in (0..40).rev() {
        acc = acc
            .mul(u)
            .add(Dd::from(1.0).div(Dd::from((2 * i + 1) as f64)));
    }
    Dd::from(2.0).mul(t).mul(acc)
}

fn ln2() -> Dd {
    two_atanh(Dd::from(1.0).div(Dd::from(3.0)))
}

/// `ln x` for finite `x > 0`: `x = m · 2ᵉ` with `m ∈ [√½, √2]`, `ln m = 2 atanh((m−1)/(m+1))`.
pub(super) fn ln(x: f64) -> Dd {
    let (mut x, mut e) = (x, 0i32);
    if x < f64::MIN_POSITIVE {
        x *= f64::from_bits((1023 + 54) << 52);
        e -= 54;
    }
    let bits = x.to_bits();
    e += ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    if m > std::f64::consts::SQRT_2 {
        m /= 2.0; // exact
        e += 1;
    }
    // m − 1 is exact (Sterbenz: m ∈ [½, 2]); m + 1 is kept as a double-double.
    let t = Dd::from(m - 1.0).div(two_sum(m, 1.0));
    ln2().mul(Dd::from(e as f64)).add(two_atanh(t))
}

pub(super) fn log10(x: f64) -> Dd {
    ln(x).div(ln(10.0))
}

/// `eᵃ` for a double-double `a` whose result is normal: `a = k ln2 + r`, `|r| ≤ ½ ln2`, then
/// Taylor's series for `eʳ`.
fn exp_dd(a: Dd) -> Dd {
    let l2 = ln2();
    let k = (a.hi / l2.hi).round();
    let r = a.add(l2.mul(Dd::from(k)).neg());
    let mut acc = Dd::from(1.0);
    for n in (1..=30).rev() {
        acc = Dd::from(1.0).add(r.mul(acc).div(Dd::from(n as f64)));
    }
    acc.scale(k as i32)
}

pub(super) fn exp(x: f64) -> Dd {
    exp_dd(Dd::from(x))
}

/// `xʸ = e^(y ln x)` for `x > 0`.
pub(super) fn pow(x: f64, y: f64) -> Dd {
    exp_dd(ln(x).mul(Dd::from(y)))
}
