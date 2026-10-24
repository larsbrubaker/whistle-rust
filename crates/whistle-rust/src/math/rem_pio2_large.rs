//! `rem_pio2_large`, ported from the Rust `libm` crate (a port of musl libm, MIT; the algorithm is
//! FreeBSD msun / Sun fdlibm, notice kept below). Part of `math`: plain f32/f64 arithmetic only,
//! so it rounds identically on every target (see `math.rs`).

/* origin: FreeBSD /usr/src/lib/msun/src/k_rem_pio2.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

use super::scalbn;

// initial value for jk
const INIT_JK: [usize; 4] = [3, 4, 4, 6];

// Table of constants for 2/pi, 396 Hex digits (476 decimal) of 2/pi
//
//              integer array, contains the (24*i)-th to (24*i+23)-th
//              bit of 2/pi after binary point. The corresponding
//              floating value is
//
//                      ipio2[i] * 2^(-24(i+1)).
//
// NB: This table must have at least (e0-3)/24 + jk terms. Only f32 and f64 inputs reach it
//     here: e0 <= 1000 and jk <= 6, so at most 47 terms; the 66 kept are the source's table for
//     32-bit targets (quad precision, e0 <= 16360, would need 686).
const IPIO2: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

const PIO2: [f64; 8] = [
    1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
    7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
    5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
    3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
    1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
    1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
    2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
    2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
];

// fn rem_pio2_large(x : &[f64], y : &mut [f64], e0 : i32, prec : usize) -> i32
//
// Input parameters:
//      x[]     The input value (must be positive) is broken into nx
//              pieces of 24-bit integers in double precision format.
//              x[i] will be the i-th 24 bit of x. The scaled exponent
//              of x[0] is given in input parameter e0 (i.e., x[0]*2^e0
//              match x's up to 24 bits.
//
//              Example of breaking a double positive z into x[0]+x[1]+x[2]:
//                      e0 = ilogb(z)-23
//                      z  = scalbn(z,-e0)
//              for i = 0,1,2
//                      x[i] = z.floor()
//                      z    = (z-x[i])*2**24
//
//      y[]     output result in an array of double precision numbers.
//              The dimension of y[] is:
//                      24-bit  precision       1
//                      53-bit  precision       2
//                      64-bit  precision       2
//                      113-bit precision       3
//              The actual value is the sum of them. Thus for 113-bit
//              precison, one may have to do something like:
//
//              long double t,w,r_head, r_tail;
//              t = (long double)y[2] + (long double)y[1];
//              w = (long double)y[0];
//              r_head = t+w;
//              r_tail = w - (r_head - t);
//
//      e0      The exponent of x[0]. Must be <= 16360 or you need to
//              expand the ipio2 table.
//
//      prec    an integer indicating the precision:
//                      0       24  bits (single)
//                      1       53  bits (double)
//                      2       64  bits (extended)
//                      3       113 bits (quad)
//
// Here is the description of some local variables:
//
//      jk      jk+1 is the initial number of terms of ipio2[] needed
//              in the computation. The minimum and recommended value
//              for jk is 3,4,4,6 for single, double, extended, and quad.
//              jk+1 must be 2 larger than you might expect so that our
//              recomputation test works. (Up to 24 bits in the integer
//              part (the 24 bits of it that we compute) and 23 bits in
//              the fraction part may be lost to cancelation before we
//              recompute.)
//
//      jz      local integer variable indicating the number of
//              terms of ipio2[] used.
//
//      jx      nx - 1
//
//      jv      index for pointing to the suitable ipio2[] for the
//              computation. In general, we want
//                      ( 2^e0*x[0] * ipio2[jv-1]*2^(-24jv) )/8
//              is an integer. Thus
//                      e0-3-24*jv >= 0 or (e0-3)/24 >= jv
//              Hence jv = max(0,(e0-3)/24).
//
//      jp      jp+1 is the number of terms in PIo2[] needed, jp = jk.
//
//      q[]     double array with integral value, representing the
//              24-bits chunk of the product of x and 2/pi.
//
//      q0      the corresponding exponent of q[0]. Note that the
//              exponent for q[i] would be q0-24*i.
//
//      PIo2[]  double precision array, obtained by cutting pi/2
//              into 24 bits chunks.
//
//      f[]     ipio2[] in floating point
//
//      iq[]    integer array by breaking up q[] in 24-bits chunk.
//
//      fq[]    final product of x*(2/pi) in fq[0],..,fq[jk]
//
//      ih      integer. If >0 it indicates q[] is >= 0.5, hence
//              it also indicates the *sign* of the result.

/// Return the last three digits of N with y = x - N*pi/2
/// so that |y| < pi/2.
///
/// The method is to compute the integer (mod 8) and fraction parts of
/// (2/pi)*x without doing the full multiplication. In general we
/// skip the part of the product that are known to be a huge integer (
/// more accurately, = 0 mod 8 ). Thus the number of operations are
/// independent of the exponent of the input.
pub(crate) fn rem_pio2_large(x: &[f64], y: &mut [f64], e0: i32, prec: usize) -> i32 {
    let x1p24 = f64::from_bits(0x4170000000000000); // 0x1p24 === 2 ^ 24
    let x1p_24 = f64::from_bits(0x3e70000000000000); // 0x1p_24 === 2 ^ (-24)

    let nx = x.len();

    let mut fw: f64;
    let mut n: i32;
    let mut ih: i32;
    let mut z: f64;
    let mut f: [f64; 20] = [0.; 20];
    let mut fq: [f64; 20] = [0.; 20];
    let mut q: [f64; 20] = [0.; 20];
    let mut iq: [i32; 20] = [0; 20];

    /* initialize jk*/
    let jk = INIT_JK[prec];
    let jp = jk;

    /* determine jx,jv,q0, note that 3>q0 */
    let jx = nx - 1;
    let mut jv = (e0 - 3) / (24);
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);
    let jv = jv as usize;

    /* set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk] */
    let mut j = (jv as i32) - (jx as i32);
    let m = jx + jk;
    for i in 0..=m {
        f[i] = if j < 0 { 0. } else { IPIO2[j as usize] as f64 };
        j += 1;
    }

    /* compute q[0],q[1],...q[jk] */
    for i in 0..=jk {
        fw = 0f64;
        for j in 0..=jx {
            fw += x[j] * f[jx + i - j];
        }
        q[i] = fw;
    }

    let mut jz = jk;

    'recompute: loop {
        /* distill q[] into iq[] reversingly */
        let mut i = 0i32;
        z = q[jz];
        for j in (1..=jz).rev() {
            fw = (x1p_24 * z) as i32 as f64;
            iq[i as usize] = (z - x1p24 * fw) as i32;
            z = q[j - 1] + fw;
            i += 1;
        }

        /* compute n */
        z = scalbn(z, q0); /* actual value of z */
        z -= 8.0 * (z * 0.125).floor(); /* trim off integer >= 8 */
        n = z as i32;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            /* need iq[jz-1] to determine n */
            i = iq[jz - 1] >> (24 - q0);
            n += i;
            iq[jz - 1] -= i << (24 - q0);
            ih = iq[jz - 1] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[jz - 1] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            /* q > 0.5 */
            n += 1;
            let mut carry = 0i32;
            for i in 0..jz {
                /* compute 1-q */
                let j = iq[i];
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        iq[i] = 0x1000000 - j;
                    }
                } else {
                    iq[i] = 0xffffff - j;
                }
            }
            if q0 > 0 {
                /* rare case: chance is 1 in 12 */
                match q0 {
                    1 => {
                        iq[jz - 1] &= 0x7fffff;
                    }
                    2 => {
                        iq[jz - 1] &= 0x3fffff;
                    }
                    _ => {}
                }
            }
            if ih == 2 {
                z = 1. - z;
                if carry != 0 {
                    z -= scalbn(1., q0);
                }
            }
        }

        /* check if recomputation is needed */
        if z == 0. {
            let mut j = 0;
            for i in (jk..=jz - 1).rev() {
                j |= iq[i];
            }
            if j == 0 {
                /* need recomputation */
                let mut k = 1;
                while iq[jk - k] == 0 {
                    k += 1; /* k = no. of terms needed */
                }

                for i in (jz + 1)..=(jz + k) {
                    /* add q[jz+1] to q[jz+k] */
                    f[jx + i] = IPIO2[jv + i] as f64;
                    fw = 0f64;
                    for j in 0..=jx {
                        fw += x[j] * f[jx + i - j];
                    }
                    q[i] = fw;
                }
                jz += k;
                continue 'recompute;
            }
        }

        break;
    }

    /* chop off zero terms */
    if z == 0. {
        jz -= 1;
        q0 -= 24;
        while iq[jz] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        /* break z into 24-bit if necessary */
        z = scalbn(z, -q0);
        if z >= x1p24 {
            fw = (x1p_24 * z) as i32 as f64;
            iq[jz] = (z - x1p24 * fw) as i32;
            jz += 1;
            q0 += 24;
            iq[jz] = fw as i32;
        } else {
            iq[jz] = z as i32;
        }
    }

    /* convert integer "bit" chunk to floating-point value */
    fw = scalbn(1., q0);
    for i in (0..=jz).rev() {
        q[i] = fw * (iq[i] as f64);
        fw *= x1p_24;
    }

    /* compute PIo2[0,...,jp]*q[jz,...,0] */
    for i in (0..=jz).rev() {
        fw = 0f64;
        let mut k = 0;
        while (k <= jp) && (k <= jz - i) {
            fw += PIO2[k] * q[i + k];
            k += 1;
        }
        fq[jz - i] = fw;
    }

    /* compress fq[] into y[] */
    match prec {
        0 => {
            fw = 0f64;
            for i in (0..=jz).rev() {
                fw += fq[i];
            }
            y[0] = if ih == 0 { fw } else { -fw };
        }
        1 | 2 => {
            fw = 0f64;
            for i in (0..=jz).rev() {
                fw += fq[i];
            }
            y[0] = if ih == 0 { fw } else { -fw };
            fw = fq[0] - fw;
            for i in 1..=jz {
                fw += fq[i];
            }
            y[1] = if ih == 0 { fw } else { -fw };
        }
        3 => {
            /* painful */
            for i in (1..=jz).rev() {
                fw = fq[i - 1] + fq[i];
                fq[i] += fq[i - 1] - fw;
                fq[i - 1] = fw;
            }
            for i in (2..=jz).rev() {
                fw = fq[i - 1] + fq[i];
                fq[i] += fq[i - 1] - fw;
                fq[i - 1] = fw;
            }
            fw = 0f64;
            for i in (2..=jz).rev() {
                fw += fq[i];
            }
            if ih == 0 {
                y[0] = fq[0];
                y[1] = fq[1];
                y[2] = fw;
            } else {
                y[0] = -fq[0];
                y[1] = -fq[1];
                y[2] = -fw;
            }
        }
        _ => {}
    }
    n & 7
}
