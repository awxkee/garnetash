/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::cabac::{CabacEncoder, Contexts};
use crate::fmt::ChromaFormat;
use crate::headers::Headers;
use crate::intra::{
    DC_IDX, HOR_IDX, NUM_LUMA_MODE, PLANAR_IDX, VER_IDX, build_mpm, chroma_422_mode,
    chroma_cand_modes, encode_bdpcm_mode, encode_chroma_mode, encode_luma_mode,
};
use crate::partition::{CTU_SIZE, code_partitions};
use crate::predict::{RefSamples, predict_into};
use crate::transform::{
    dequantize_ts_wh, dequantize_wh, fwd_transform_wh_into, inv_transform_wh,
    inv_transform_wh_into, quantize_ts_wh, quantize_wh, rdoq_wh, satd,
};
use crate::tu::{TreeType, TuCoeffs, encode_transform_tree};
use std::sync::atomic::{AtomicI32, AtomicU8, Ordering::Relaxed};

const RDO_CANDIDATES: usize = 4;

#[allow(dead_code)] // used by in-crate tests
fn gather_refs(
    recon: &[i32],
    avail: &[bool],
    cw: usize,
    ch: usize,
    x: usize,
    y: usize,
    n: usize,
    bd: u8,
) -> RefSamples {
    gather_refs_wh(recon, avail, cw, ch, x, y, n, n, bd)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_refs_wh(
    recon: &[i32],
    avail: &[bool],
    cw: usize,
    ch: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    bd: u8,
) -> RefSamples {
    let corner = if x > 0 && y > 0 && avail[(y - 1) * cw + (x - 1)] {
        Some(recon[(y - 1) * cw + (x - 1)])
    } else {
        None
    };
    let above: Vec<Option<i32>> = (0..2 * bw)
        .map(|i| {
            let xx = x + i;
            if y > 0 && xx < cw && avail[(y - 1) * cw + xx] {
                Some(recon[(y - 1) * cw + xx])
            } else {
                None
            }
        })
        .collect();
    let left: Vec<Option<i32>> = (0..2 * bh)
        .map(|i| {
            let yy = y + i;
            if x > 0 && yy < ch && avail[yy * cw + (x - 1)] {
                Some(recon[yy * cw + (x - 1)])
            } else {
                None
            }
        })
        .collect();
    RefSamples::build(bw, bh, corner, &above, &left, bd)
}

/// Rectangular generalization of [`analyze`] using the `(w,h)` transform/quant.
fn analyze_wh(
    src_blk: &[i32],
    pred: &[i32],
    bw: usize,
    bh: usize,
    qp: u8,
    bd: u8,
    lambda: f64,
    lfnst_idx: u8,
    lfnst_mode: usize,
    dep_quant: bool,
) -> Vec<i32> {
    thread_local! {
        // Reused residual + coefficient scratch: avoids a per-call residual Vec
        // and a 16 KiB forward-transform buffer in this hot RDO inner function.
        static FWD: std::cell::RefCell<(Vec<i32>, Vec<i32>)> =
            std::cell::RefCell::new((vec![0; 64 * 64], vec![0; 64 * 64]));
    }
    let nn = bw * bh;
    FWD.with(|c| {
        let (res, coeff) = &mut *c.borrow_mut();
        for (r, (&s, &p)) in res[..nn].iter_mut().zip(src_blk.iter().zip(pred)) {
            *r = s - p;
        }
        fwd_transform_wh_into(&mut coeff[..nn], &res[..nn], bw, bh, bd);
        if lfnst_idx > 0 {
            crate::lfnst::apply_fwd_lfnst(&mut coeff[..nn], bw, bh, lfnst_mode, lfnst_idx as usize);
        }
        let levels = if dep_quant {
            return crate::transform::dq_trellis_wh(&coeff[..nn], bw, bh, qp, bd, lambda);
        } else if lambda > 0.0 {
            rdoq_wh(&coeff[..nn], bw, bh, qp, bd, lambda)
        } else {
            quantize_wh(&coeff[..nn], bw, bh, qp, bd)
        };
        levels[..nn].iter().map(|&l| l as i32).collect()
    })
}

/// Per-component "levels" for coding a `bw × bh` block: the quantized transform
/// coefficients in lossy mode, or — for lossless — the raw signed residual
/// `src - pred`, which the transform-skip residual coder transmits verbatim.
fn residual_levels_wh(
    src_blk: &[i32],
    pred: &[i32],
    bw: usize,
    bh: usize,
    qp: u8,
    bd: u8,
    lossless: bool,
    lambda: f64,
    dep_quant: bool,
) -> Vec<i32> {
    if lossless {
        (0..bw * bh).map(|i| src_blk[i] - pred[i]).collect()
    } else {
        analyze_wh(src_blk, pred, bw, bh, qp, bd, lambda, 0, 0, dep_quant)
    }
}

/// Reconstruct one block from its quantized `levels` and prediction, writing the
/// result into `recon` and marking `avail`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // used by in-crate tests
fn reconstruct(
    recon: &mut [i32],
    avail: &mut [bool],
    cw: usize,
    x: usize,
    y: usize,
    n: usize,
    pred: &[i32],
    levels: &[i32],
    qp: u8,
    bd: u8,
    max_val: i32,
    lossless: bool,
    ts: bool,
    lfnst_idx: u8,
    lfnst_mode: usize,
    dep_quant: bool,
    mts_idx: u8,
) {
    reconstruct_wh(
        recon, avail, cw, x, y, n, n, pred, levels, qp, bd, max_val, lossless, ts, lfnst_idx,
        lfnst_mode, dep_quant, mts_idx,
    );
}

/// Rectangular generalization of [`reconstruct`]: a `bw × bh` block at chroma
/// resolution. Square blocks use the wrapper, so the luma path is unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_wh(
    recon: &mut [i32],
    avail: &mut [bool],
    cw: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    pred: &[i32],
    levels: &[i32],
    qp: u8,
    bd: u8,
    max_val: i32,
    lossless: bool,
    ts: bool,
    lfnst_idx: u8,
    lfnst_mode: usize,
    dep_quant: bool,
    mts_idx: u8,
) {
    let nn = bw * bh;
    // In lossless mode the "levels" are the raw residual, so the reconstruction
    // is pred + residual == source, exactly (no transform / dequant). In lossy
    // transform-skip mode the levels are the TS-quantized spatial residual, so
    // the reconstruction dequantizes them directly with no inverse transform.
    let add_into = |recon: &mut [i32], avail: &mut [bool], res: &[i32]| {
        for (yy, (pred_row, res_row)) in pred
            .chunks_exact(bw)
            .zip(res.chunks_exact(bw))
            .take(bh)
            .enumerate()
        {
            let row = (y + yy) * cw + x;
            for (((r, a), &p), &rs) in recon[row..row + bw]
                .iter_mut()
                .zip(avail[row..row + bw].iter_mut())
                .zip(pred_row)
                .zip(res_row)
            {
                *r = (p + rs).clamp(0, max_val);
                *a = true;
            }
        }
    };
    if lossless {
        add_into(recon, avail, &levels[..nn]);
    } else {
        let lv: Vec<i16> = levels[..nn].iter().map(|&l| l as i16).collect();
        if ts {
            let deq = dequantize_ts_wh(&lv, bw, bh, qp);
            add_into(recon, avail, &deq[..nn]);
        } else {
            let mut deq = if dep_quant {
                crate::transform::dequantize_dq_wh(
                    &levels[..nn],
                    crate::residual::scan_coords(bw, bh),
                    bw,
                    bh,
                    qp,
                    bd,
                )
            } else {
                dequantize_wh(&lv, bw, bh, qp, bd)
            };
            if lfnst_idx > 0 {
                crate::lfnst::apply_inv_lfnst(
                    &mut deq[..nn],
                    bw,
                    bh,
                    lfnst_mode,
                    lfnst_idx as usize,
                    15,
                );
            }
            let inv = if mts_idx > 0 {
                crate::transform::inv_transform_mts_wh(&deq[..nn], bw, bh, bd, mts_idx)
            } else {
                inv_transform_wh(&deq[..nn], bw, bh, bd)
            };
            add_into(recon, avail, &inv[..nn]);
        }
    }
}

fn dpcm_levels_wh(src: &[i32], refs: &RefSamples, bw: usize, bh: usize, dir: u8) -> Vec<i32> {
    let mut lv = vec![0i32; bw * bh];
    if dir == 1 {
        // Horizontal: difference each sample from the one to its left (or the
        // left reference at the edge).
        for (y, (lv_row, src_row)) in lv
            .chunks_exact_mut(bw)
            .zip(src.chunks_exact(bw))
            .enumerate()
        {
            let mut prev = refs.left[y + 1];
            for (d, &sv) in lv_row.iter_mut().zip(src_row) {
                *d = sv - prev;
                prev = sv;
            }
        }
    } else {
        // Vertical: difference each sample from the one above (or the top
        // reference for the first row). `prev_row` tracks the row above.
        let mut prev_row: &[i32] = &refs.top[1..1 + bw];
        for (lv_row, src_row) in lv.chunks_exact_mut(bw).zip(src.chunks_exact(bw)) {
            for ((d, &sv), &p) in lv_row.iter_mut().zip(src_row).zip(prev_row) {
                *d = sv - p;
            }
            prev_row = src_row;
        }
    }
    lv
}

/// SATD of a `bw × bh` block, tiled into square Hadamard blocks of side
/// `min(bw,bh)`. Used only to rank chroma intra modes (4:2:2 chroma is
/// rectangular); it influences compression, never bitstream conformance, since
/// the chosen mode is coded and reproduced identically by the decoder.
fn satd_wh(block: &[i32], bw: usize, bh: usize) -> i64 {
    if bw == bh {
        return satd(block, bw);
    }
    let s = bw.min(bh);
    let mut total = 0i64;
    let mut tile = vec![0i32; s * s];
    let mut ty = 0;
    while ty < bh {
        let mut tx = 0;
        while tx < bw {
            for yy in 0..s {
                for xx in 0..s {
                    tile[yy * s + xx] = block[(ty + yy) * bw + (tx + xx)];
                }
            }
            total += satd(&tile, s);
            tx += s;
        }
        ty += s;
    }
    total
}

#[allow(dead_code)] // used by in-crate tests
fn place_block(
    recon: &mut [i32],
    avail: &mut [bool],
    cw: usize,
    x: usize,
    y: usize,
    n: usize,
    src_blk: &[i32],
) {
    place_block_wh(recon, avail, cw, x, y, n, n, src_blk);
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // used by in-crate tests
fn place_block_wh(
    recon: &mut [i32],
    avail: &mut [bool],
    cw: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    src_blk: &[i32],
) {
    for (yy, blk_row) in src_blk.chunks_exact(bw).take(bh).enumerate() {
        let off = (y + yy) * cw + x;
        recon[off..off + bw].copy_from_slice(blk_row);
        avail[off..off + bw].fill(true);
    }
}

fn reconstruct_block_wh(
    pred: &[i32],
    levels: &[i32],
    bw: usize,
    bh: usize,
    qp: u8,
    bd: u8,
    max_val: i32,
    lossless: bool,
    ts: bool,
    lfnst_idx: u8,
    lfnst_mode: usize,
    dep_quant: bool,
    mts_idx: u8,
) -> Vec<i32> {
    let nn = bw * bh;
    let res: Vec<i32> = if lossless {
        levels[..nn].to_vec()
    } else {
        let lv: Vec<i16> = levels[..nn].iter().map(|&l| l as i16).collect();
        if ts {
            dequantize_ts_wh(&lv, bw, bh, qp)[..nn].to_vec()
        } else {
            let mut deq = if dep_quant {
                crate::transform::dequantize_dq_wh(
                    &levels[..nn],
                    crate::residual::scan_coords(bw, bh),
                    bw,
                    bh,
                    qp,
                    bd,
                )
            } else {
                dequantize_wh(&lv, bw, bh, qp, bd)
            };
            if lfnst_idx > 0 {
                crate::lfnst::apply_inv_lfnst(
                    &mut deq[..nn],
                    bw,
                    bh,
                    lfnst_mode,
                    lfnst_idx as usize,
                    15,
                );
            }
            if mts_idx > 0 {
                crate::transform::inv_transform_mts_wh(&deq[..nn], bw, bh, bd, mts_idx)[..nn]
                    .to_vec()
            } else {
                inv_transform_wh(&deq[..nn], bw, bh, bd)[..nn].to_vec()
            }
        }
    };
    (0..nn)
        .map(|i| (pred[i] + res[i]).clamp(0, max_val))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn gather_refs_atomic_wh(
    recon: &[AtomicI32],
    avail: &[AtomicU8],
    cw: usize,
    ch: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    bd: u8,
) -> RefSamples {
    let av = |i: usize| avail[i].load(Relaxed) != 0;
    let rc = |i: usize| recon[i].load(Relaxed);
    let corner = if x > 0 && y > 0 && av((y - 1) * cw + (x - 1)) {
        Some(rc((y - 1) * cw + (x - 1)))
    } else {
        None
    };
    let above: Vec<Option<i32>> = (0..2 * bw)
        .map(|i| {
            let xx = x + i;
            if y > 0 && xx < cw && av((y - 1) * cw + xx) {
                Some(rc((y - 1) * cw + xx))
            } else {
                None
            }
        })
        .collect();
    let left: Vec<Option<i32>> = (0..2 * bh)
        .map(|i| {
            let yy = y + i;
            if x > 0 && yy < ch && av(yy * cw + (x - 1)) {
                Some(rc(yy * cw + (x - 1)))
            } else {
                None
            }
        })
        .collect();
    RefSamples::build(bw, bh, corner, &above, &left, bd)
}

/// Store a `bw × bh` block into the atomic plane and mark it available.
#[allow(clippy::too_many_arguments)]
fn place_atomic_wh(
    recon: &[AtomicI32],
    avail: &[AtomicU8],
    cw: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    block: &[i32],
) {
    for (yy, blk_row) in block.chunks_exact(bw).take(bh).enumerate() {
        let row = (y + yy) * cw + x;
        for ((r, a), &v) in recon[row..row + bw]
            .iter()
            .zip(&avail[row..row + bw])
            .zip(blk_row)
        {
            r.store(v, Relaxed);
            a.store(1, Relaxed);
        }
    }
}

/// Atomic counterpart of [`reconstruct_wh`]: dequantise/inverse-transform and
/// store into the shared plane.
#[allow(clippy::too_many_arguments)]
fn reconstruct_atomic_wh(
    recon: &[AtomicI32],
    avail: &[AtomicU8],
    cw: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    pred: &[i32],
    levels: &[i32],
    qp: u8,
    bd: u8,
    max_val: i32,
    lossless: bool,
    ts: bool,
    lfnst_idx: u8,
    lfnst_mode: usize,
    dep_quant: bool,
    mts_idx: u8,
) {
    let blk = reconstruct_block_wh(
        pred, levels, bw, bh, qp, bd, max_val, lossless, ts, lfnst_idx, lfnst_mode, dep_quant,
        mts_idx,
    );
    place_atomic_wh(recon, avail, cw, x, y, bw, bh, &blk);
}

/// Copy a `bw × bh` block out of `src` into a caller-provided `dst` slice,
/// so a leaf reuses one scratch buffer instead of allocating per call.
fn extract_block_wh_into(
    dst: &mut [i32],
    src: &[i32],
    cw: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
) {
    for (yy, dst_row) in dst.chunks_exact_mut(bw).take(bh).enumerate() {
        let row = (y + yy) * cw + x;
        dst_row.copy_from_slice(&src[row..row + bw]);
    }
}

/// Sample variance of a luma block, used as a detail measure for the split
/// decision.
fn block_variance(src: &[i32], cw: usize, x: usize, y: usize, n: usize) -> i64 {
    let (mut sum, mut sumsq) = (0i64, 0i64);
    for yy in 0..n {
        let row = (y + yy) * cw + x;
        for &sv in &src[row..row + n] {
            let v = sv as i64;
            sum += v;
            sumsq += v * v;
        }
    }
    let count = (n * n) as i64;
    let mean = sum / count;
    sumsq / count - mean * mean
}

const AQ_STRENGTH: f64 = 1.0;
const AQ_RANGE: i32 = 6;
const AQ_QP_MIN: i32 = 7;
const AQ_QP_MAX: i32 = 51;
pub(crate) const AQ_QG: usize = 32;
pub(crate) const AQ_CU_QP_DELTA_SUBDIV: u32 = 4;

/// Variance of luma over the in-bounds part of a (possibly edge-clipped) CTU.
fn region_variance(src: &[i32], cw: usize, ch: usize, x: usize, y: usize, ctu: usize) -> f64 {
    let bw = ctu.min(cw - x);
    let bh = ctu.min(ch - y);
    let (mut sum, mut sumsq) = (0i64, 0i64);
    for yy in 0..bh {
        let row = (y + yy) * cw + x;
        for &sv in &src[row..row + bw] {
            let v = sv as i64;
            sum += v;
            sumsq += v * v;
        }
    }
    let count = (bw * bh).max(1) as i64;
    let mean = sum / count;
    (sumsq / count - mean * mean).max(0) as f64
}

fn aq_qg_targets(
    src_y: &[i32],
    cw: usize,
    ch: usize,
    qg: usize,
    qg_cols: usize,
    qg_rows: usize,
    base_qp: u8,
) -> Vec<i32> {
    let n = qg_cols * qg_rows;
    let mut energy = vec![0f64; n];
    for r in 0..qg_rows {
        for c in 0..qg_cols {
            let v = region_variance(src_y, cw, ch, c * qg, r * qg, qg);
            energy[r * qg_cols + c] = (1.0 + v).log2();
        }
    }
    let mean = energy.iter().sum::<f64>() / n as f64;
    let lo = AQ_QP_MIN.min(base_qp as i32);
    energy
        .iter()
        .map(|&e| {
            let off = (AQ_STRENGTH * (e - mean)).round() as i32;
            (base_qp as i32 + off.clamp(-AQ_RANGE, AQ_RANGE)).clamp(lo, AQ_QP_MAX)
        })
        .collect()
}

pub(crate) fn aq_predict_qp(
    qg_map: &[i32],
    qg_cols: usize,
    qg: usize,
    ctu: usize,
    qg_x: usize,
    qg_y: usize,
    prev_qp: i32,
) -> i32 {
    let cell = |px: usize, py: usize| qg_map[(py / qg) * qg_cols + px / qg];
    if qg_x == 0 && qg_y.is_multiple_of(ctu) && qg_y > 0 {
        cell(qg_x, qg_y - 1)
    } else {
        let a = if !qg_y.is_multiple_of(ctu) {
            cell(qg_x, qg_y - 1)
        } else {
            prev_qp
        };
        let b = if !qg_x.is_multiple_of(ctu) {
            cell(qg_x - 1, qg_y)
        } else {
            prev_qp
        };
        (a + b + 1) >> 1
    }
}

/// Write `qp` into every QG-map cell covered by the quantization group whose
/// first leaf is (`x`, `y`, `n`). A leaf at least as large as the QG size is its
/// own QG and spans several cells; a smaller leaf occupies the single cell
/// covering its position.
pub(crate) fn aq_fill_qg(
    qg_map: &mut [i32],
    qg_cols: usize,
    qg: usize,
    x: usize,
    y: usize,
    n: usize,
    qp: i32,
) {
    let span = if n >= qg { n / qg } else { 1 };
    let (ox, oy) = (x / qg, y / qg);
    for cy in 0..span {
        for cx in 0..span {
            let i = (oy + cy) * qg_cols + ox + cx;
            if ox + cx < qg_cols && i < qg_map.len() {
                qg_map[i] = qp;
            }
        }
    }
}

#[allow(dead_code)]
struct CoreOutput {
    stream: Vec<u8>,
    slice_data: Vec<u8>,
    rec_y: Vec<i32>,
    rec_cb: Vec<i32>,
    rec_cr: Vec<i32>,
    cw: usize,
    ch: usize,
    cwc: usize,
    chc: usize,
    qp: u8,
    bit_depth: u8,
    leaf_count: usize,
}

const FWD_PREC: i32 = 13;
const FWD_BIAS: i32 = (1 << (FWD_PREC - 1)) - 1; // 4095
const FWD_YR: i32 = 2449;
const FWD_YG: i32 = 4809;
const FWD_YB: i32 = 934;
const FWD_CB_R: i32 = -1382;
const FWD_CB_G: i32 = -2714;
const FWD_CB_B: i32 = 4096;
const FWD_CR_R: i32 = 4096;
const FWD_CR_G: i32 = -3430;
const FWD_CR_B: i32 = -666;

#[inline(always)]
fn rgb_to_y_q13(r: i32, g: i32, b: i32) -> i32 {
    (FWD_YR * r + FWD_YG * g + FWD_YB * b + FWD_BIAS) >> FWD_PREC
}

/// end-to-end decode tests.
fn encode_core<S: Copy + Into<i32>>(
    rgb: &[S],
    width: u32,
    height: u32,
    qp: u8,
    bit_depth: u8,
    stride_px: usize,
    lossless: bool,
    chroma: ChromaFormat,
    scale_shift: u32,
    threads: usize,
    rdoq: bool,
    aq: bool,
    mtt: bool,
    lfnst: bool,
    dep_quant: bool,
    mts: bool,
    dual_tree: bool,
    cclm: bool,
    deblock: bool,
) -> CoreOutput {
    // Dual tree falls back to single tree for lossless non-4:2:0; AQ's cu_qp_delta
    // is not implemented for the dual tree, so it is disabled there.
    let dual = dual_tree && !lossless;
    // LFNST is validated for 8-bit (single and dual tree, every chroma format and
    // picture size). In the dual tree luma and chroma carry separate LFNST
    // indices; both are signalled (see replay_luma / replay_chroma).
    let lfnst_eff = lfnst;
    let dep_quant_eff = dep_quant;
    let headers = Headers {
        width,
        height,
        chroma,
        bit_depth: crate::fmt::BitDepth::from_bits(bit_depth),
        qp,
        lossless,
        // AQ falls back to a flat QP for lossless: cu_qp_delta interacts with
        // the transquant-bypass QG handling in a way the reference decoder
        // does not accept (same rationale as the dual tree, where it is also
        // disabled).
        aq: aq && !lossless,
        // MTT (like the dual tree) falls back to the plain quadtree for
        // lossless: only the QT path forces CUs <= 32 so transform-skip
        // (which lossless requires) is always available.
        mtt: mtt && !lossless,
        lfnst: lfnst_eff,
        dep_quant: dep_quant_eff,
        mts,
        dual_tree: dual,
        cclm: cclm && !lossless,
        deblock: deblock && bit_depth == 8 && !aq && !dual && !lossless && !mtt,
    };
    let cw = headers.coded_width() as usize;
    let ch = headers.coded_height() as usize;
    let (w, h) = (width as usize, height as usize);
    let shift = scale_shift;
    let mid = 1i32 << (bit_depth - 1);
    let max_val = (1i32 << bit_depth) - 1;

    // RGB -> YCbCr (BT.601 full range), luma at coded size with edge replication.
    let mut src_y = vec![0i32; cw * ch];
    // Chroma plane dimensions follow the subsampling factors; monochrome has none.
    let (sub_w, sub_h) = (chroma.sub_w(), chroma.sub_h());
    let has_chroma = !chroma.is_monochrome();
    let cwc = if has_chroma { cw / sub_w } else { 0 };
    let chc = if has_chroma { ch / sub_h } else { 0 };
    let mut src_cb = vec![0i32; cwc * chc];
    let mut src_cr = vec![0i32; cwc * chc];
    let sample = |x: usize, y: usize| {
        let sx = x.min(w - 1);
        let sy = y.min(h - 1);
        let o = (sy * w + sx) * stride_px;
        (rgb[o].into(), rgb[o + 1].into(), rgb[o + 2].into())
    };
    for y in 0..ch {
        for x in 0..cw {
            let (r, g, b) = sample(x, y);
            src_y[y * cw + x] = (rgb_to_y_q13(r, g, b) << shift).clamp(0, max_val);
        }
    }
    // Chroma: average each sub_w×sub_h RGB region (rounded, matching the `yuv`
    // crate), then convert directly from RGB. 4:4:4 has a 1×1 region (no average).
    let region_shift = (sub_w * sub_h).trailing_zeros();
    let region_half = (1i32 << region_shift) >> 1; // 0 (4:4:4), 1 (4:2:2), 2 (4:2:0)
    for cy in 0..chc {
        for cx in 0..cwc {
            let (mut sr, mut sg, mut sb) = (0i32, 0i32, 0i32);
            for dy in 0..sub_h {
                for dx in 0..sub_w {
                    let (r, g, b) = sample(cx * sub_w + dx, cy * sub_h + dy);
                    sr += r;
                    sg += g;
                    sb += b;
                }
            }
            let r = (sr + region_half) >> region_shift;
            let g = (sg + region_half) >> region_shift;
            let b = (sb + region_half) >> region_shift;
            let cb = ((FWD_CB_R * r + FWD_CB_G * g + FWD_CB_B * b + FWD_BIAS) >> FWD_PREC) << shift;
            let cr = ((FWD_CR_R * r + FWD_CR_G * g + FWD_CR_B * b + FWD_BIAS) >> FWD_PREC) << shift;
            src_cb[cy * cwc + cx] = (mid + cb).clamp(0, max_val);
            src_cr[cy * cwc + cx] = (mid + cr).clamp(0, max_val);
        }
    }

    encode_planes(headers, src_y, src_cb, src_cr, threads, rdoq)
}

struct LeafDecision {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    luma_bdpcm: u8,
    best_mode: u8,
    luma_dir_mode: u8,
    luma_ts: bool,
    luma_levels: Vec<i32>,
    lfnst_idx: u8,
    luma_mts_idx: u8,
    chroma: Option<ChromaDecision>,
}

struct ChromaDecision {
    ccw: usize,
    cch: usize,
    chroma_mode: u8,
    chroma_bdpcm: u8,
    chroma_ts: bool,
    cb_levels: Vec<i32>,
    cr_levels: Vec<i32>,
}

struct ModeGrid {
    mode_of: Vec<AtomicU8>,
    cells_x: usize,
    coded_w: u32,
    coded_h: u32,
}

impl ModeGrid {
    fn new(coded_w: u32, coded_h: u32) -> Self {
        let cells_x = coded_w as usize / 4;
        let cells_y = coded_h as usize / 4;
        let mode_of = (0..cells_x * cells_y)
            .map(|_| AtomicU8::new(PLANAR_IDX))
            .collect();
        ModeGrid {
            mode_of,
            cells_x,
            coded_w,
            coded_h,
        }
    }
    #[inline]
    fn cell(&self, x: u32, y: u32) -> usize {
        (y as usize / 4) * self.cells_x + (x as usize / 4)
    }
    fn set_mode_rect(&self, x: u32, y: u32, w: u32, h: u32, mode: u8) {
        let mut cy = y;
        while cy < y + h && cy < self.coded_h {
            let mut cx = x;
            while cx < x + w && cx < self.coded_w {
                let idx = self.cell(cx, cy);
                self.mode_of[idx].store(mode, Relaxed);
                cx += 4;
            }
            cy += 4;
        }
    }
    fn left_mode_rect(&self, x: u32, y: u32, h: u32) -> Option<u8> {
        if x > 0 {
            Some(self.mode_of[self.cell(x - 1, y + h - 1)].load(Relaxed))
        } else {
            None
        }
    }
    fn above_mode_rect(&self, x: u32, y: u32, w: u32) -> Option<u8> {
        if y > 0 {
            Some(self.mode_of[self.cell(x + w - 1, y - 1)].load(Relaxed))
        } else {
            None
        }
    }
    /// Dual-tree chroma DM: luma mode at the centre of the co-located luma block
    /// `(x, y, w, h)` (luma coords). Matches VTM `getCoLocatedLumaPU`.
    fn chroma_dm(&self, x: u32, y: u32, w: u32, h: u32) -> u8 {
        let cx = (x + (w >> 1)).min(self.coded_w - 1);
        let cy = (y + (h >> 1)).min(self.coded_h - 1);
        self.mode_of[self.cell(cx, cy)].load(Relaxed)
    }
}

struct Planes {
    rec_y: Vec<AtomicI32>,
    av_y: Vec<AtomicU8>,
    rec_cb: Vec<AtomicI32>,
    av_cb: Vec<AtomicU8>,
    rec_cr: Vec<AtomicI32>,
    av_cr: Vec<AtomicU8>,
    modes: ModeGrid,
}

impl Planes {
    fn new(cw: usize, ch: usize, cwc: usize, chc: usize) -> Self {
        let zeros = |n: usize| (0..n).map(|_| AtomicI32::new(0)).collect();
        let avail = |n: usize| (0..n).map(|_| AtomicU8::new(0)).collect();
        Planes {
            rec_y: zeros(cw * ch),
            av_y: avail(cw * ch),
            rec_cb: zeros(cwc * chc),
            av_cb: avail(cwc * chc),
            rec_cr: zeros(cwc * chc),
            av_cr: avail(cwc * chc),
            modes: ModeGrid::new(cw as u32, ch as u32),
        }
    }
    /// Consume the atomic luma/chroma planes into plain `Vec<i32>` for output.
    fn into_recon(self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let conv = |v: Vec<AtomicI32>| v.into_iter().map(|a| a.into_inner()).collect();
        (conv(self.rec_y), conv(self.rec_cb), conv(self.rec_cr))
    }
}

/// Immutable per-picture inputs shared by every leaf analysis (read-only, so
/// freely shared across threads).
struct LeafShared<'a> {
    src_y: &'a [i32],
    src_cb: &'a [i32],
    src_cr: &'a [i32],
    cw: usize,
    ch: usize,
    cwc: usize,
    chc: usize,
    sub_w: usize,
    sub_h: usize,
    has_chroma: bool,
    bit_depth: u8,
    qp: u8,
    lossless: bool,
    rdoq: bool,
    /// Enable LFNST trial for eligible intra luma blocks.
    lfnst: bool,
    /// Enable dependent quantization for transformed (non-transform-skip) blocks.
    dep_quant: bool,
    /// Enable explicit MTS (DST7/DCT8) for eligible intra luma TUs.
    mts: bool,
    /// Enable CCLM chroma prediction (single tree: the collocated luma is
    /// committed to the recon plane before chroma analysis so CCLM can read it).
    cclm: bool,
    /// Adaptive quant: when set, each leaf quantizes at its CTU's target Qp'Y
    /// from [`ctu_qpprime`](Self::ctu_qpprime) rather than the uniform [`qp`].
    aq: bool,
    /// Quantization-group size in luma samples (sub-CTU under adaptive quant).
    qg: usize,
    /// QGs per row, for indexing [`qg_qpprime`](Self::qg_qpprime).
    qg_cols: usize,
    /// Per-QG quantization QP (Qp'Y). Indexed `(y / qg) * qg_cols + x / qg`.
    /// Equals `qp` everywhere when `!aq`.
    qg_qpprime: &'a [u8],
    max_val: i32,
    /// Fixed slice-initial CABAC contexts used purely for the RD *rate*
    /// estimate. Using a fixed state (rather than the evolving live contexts)
    /// makes each leaf's decision independent of coding order, which is what
    /// lets the analysis run as a parallel wavefront with identical results.
    rd_ctx: &'a Contexts,
}

/// Thread-local scratch reused across leaves (one set per worker).
struct LeafScratch {
    pred_buf: Vec<i32>,
    res_buf: Vec<i32>,
    ang_scratch: Vec<i32>,
    scored: Vec<(i64, u8)>,
    cpred_cb: Vec<i32>,
    cpred_cr: Vec<i32>,
    cres: Vec<i32>,
    cscratch: Vec<i32>,
    best_pred: Vec<i32>,
    sblk_y: Vec<i32>,
    sblk_cb: Vec<i32>,
    sblk_cr: Vec<i32>,
    inv_buf: Vec<i32>,
    /// Reused dequant + i16-level scratch for the stage-2 RD candidate loop
    /// (avoids a 16 KiB array zero-fill and a heap Vec per candidate).
    deq_buf: Vec<i32>,
    lv_buf: Vec<i16>,
}

impl LeafScratch {
    fn new() -> Self {
        const MAX_CU: usize = 64 * 64;
        LeafScratch {
            pred_buf: vec![0; MAX_CU],
            res_buf: vec![0; MAX_CU],
            ang_scratch: Vec::with_capacity(MAX_CU),
            scored: Vec::with_capacity(NUM_LUMA_MODE as usize),
            cpred_cb: vec![0; MAX_CU],
            cpred_cr: vec![0; MAX_CU],
            cres: vec![0; MAX_CU],
            cscratch: Vec::with_capacity(MAX_CU),
            best_pred: vec![0; MAX_CU],
            sblk_y: vec![0; MAX_CU],
            sblk_cb: vec![0; MAX_CU],
            sblk_cr: vec![0; MAX_CU],
            inv_buf: vec![0; MAX_CU],
            deq_buf: vec![0; MAX_CU],
            lv_buf: Vec::with_capacity(MAX_CU),
        }
    }
}

thread_local! {
    static RD_ENC: std::cell::RefCell<CabacEncoder> = std::cell::RefCell::new(CabacEncoder::new());
}

fn run_ctu_wavefront<R, F>(ctu_cols: usize, ctu_rows: usize, threads: usize, process: F) -> Vec<R>
where
    R: Send + Default,
    F: Fn(usize, &mut LeafScratch) -> R + Sync,
{
    let total = ctu_cols * ctu_rows;
    if threads <= 1 {
        // Row-major order already respects causality (left + above first).
        let mut sc = LeafScratch::new();
        return (0..total).map(|cidx| process(cidx, &mut sc)).collect();
    }
    use std::collections::VecDeque;
    use std::sync::atomic::{
        AtomicUsize,
        Ordering::{AcqRel, Acquire},
    };
    use std::sync::{Condvar, Mutex};
    let at = |r: usize, c: usize| r * ctu_cols + c;
    let remaining: Vec<AtomicUsize> = (0..total)
        .map(|i| {
            let (r, c) = (i / ctu_cols, i % ctu_cols);
            let mut n = 0usize;
            if c > 0 {
                n += 1;
            }
            if r > 0 {
                n += 1;
            }
            if r > 0 && c + 1 < ctu_cols {
                n += 1;
            }
            if r > 0 && c > 0 {
                n += 1;
            }
            AtomicUsize::new(n)
        })
        .collect();
    let queue = Mutex::new(VecDeque::<usize>::new());
    let cvar = Condvar::new();
    {
        let mut q = queue.lock().unwrap();
        for (i, rem) in remaining.iter().enumerate() {
            if rem.load(Relaxed) == 0 {
                q.push_back(i);
            }
        }
    }
    let done = AtomicUsize::new(0);
    let nthreads = threads.min(total).max(1);
    let remaining_ref = &remaining;
    let queue_ref = &queue;
    let cvar_ref = &cvar;
    let done_ref = &done;
    let process_ref = &process;
    let results: Vec<Vec<(usize, R)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|_| {
                s.spawn(move || {
                    let mut sc = LeafScratch::new();
                    let mut out: Vec<(usize, R)> = Vec::new();
                    loop {
                        let next = {
                            let mut q = queue_ref.lock().unwrap();
                            loop {
                                if let Some(c) = q.pop_front() {
                                    break Some(c);
                                }
                                if done_ref.load(Acquire) >= total {
                                    break None;
                                }
                                q = cvar_ref.wait(q).unwrap();
                            }
                        };
                        let Some(cidx) = next else { break };
                        let r = process_ref(cidx, &mut sc);
                        out.push((cidx, r));
                        let (rr, cc) = (cidx / ctu_cols, cidx % ctu_cols);
                        let deps = [
                            (cc + 1 < ctu_cols).then(|| at(rr, cc + 1)),
                            (rr + 1 < ctu_rows).then(|| at(rr + 1, cc)),
                            (rr + 1 < ctu_rows && cc > 0).then(|| at(rr + 1, cc - 1)),
                            (rr + 1 < ctu_rows && cc + 1 < ctu_cols).then(|| at(rr + 1, cc + 1)),
                        ];
                        for e in deps.into_iter().flatten() {
                            if remaining_ref[e].fetch_sub(1, AcqRel) == 1 {
                                queue_ref.lock().unwrap().push_back(e);
                                cvar_ref.notify_one();
                            }
                        }
                        if done_ref.fetch_add(1, AcqRel) + 1 >= total {
                            cvar_ref.notify_all();
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut assembled: Vec<R> = (0..total).map(|_| R::default()).collect();
    for worker_out in results {
        for (cidx, r) in worker_out {
            assembled[cidx] = r;
        }
    }
    assembled
}

#[allow(clippy::type_complexity)]
fn analyze_ctus_wavefront(
    shared: &LeafShared,
    planes: &Planes,
    ctu_leaves: &[Vec<(usize, usize, usize, usize)>],
    ctu_cols: usize,
    ctu_rows: usize,
    threads: usize,
) -> Vec<Vec<LeafDecision>> {
    run_ctu_wavefront(ctu_cols, ctu_rows, threads, |cidx, sc| {
        let mut decs = Vec::with_capacity(ctu_leaves[cidx].len());
        for &(x, y, w, h) in &ctu_leaves[cidx] {
            decs.push(analyze_leaf(
                shared,
                planes,
                sc,
                x,
                y,
                w,
                h,
                TreeType::Single,
            ));
        }
        decs
    })
}

#[allow(unused, clippy::type_complexity)]
fn analyze_leaf(
    sh: &LeafShared,
    pl: &Planes,
    sc: &mut LeafScratch,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    tree: TreeType,
) -> LeafDecision {
    let (cw, ch) = (sh.cw, sh.ch);
    let bit_depth = sh.bit_depth;
    let qp = if sh.aq {
        sh.qg_qpprime[(y / sh.qg) * sh.qg_cols + x / sh.qg]
    } else {
        sh.qp
    };
    let lossless = sh.lossless;
    let max_val = sh.max_val;
    let ctx = sh.rd_ctx;

    let refs_y = gather_refs_atomic_wh(&pl.rec_y, &pl.av_y, cw, ch, x, y, w, h, bit_depth);
    extract_block_wh_into(&mut sc.sblk_y[..w * h], sh.src_y, cw, x, y, w, h);
    let sblk_y = &sc.sblk_y[..w * h];
    let mpm = build_mpm(
        pl.modes.left_mode_rect(x as u32, y as u32, h as u32),
        pl.modes.above_mode_rect(x as u32, y as u32, w as u32),
    );

    // Stage 1: coarse-to-fine SATD pre-selection.
    let filt_y = if w >= 8 && h >= 8 {
        Some(crate::predict::filter_references(&refs_y, w, h))
    } else {
        None
    };
    let nn = w * h;
    sc.scored.clear();
    let mut done = [false; NUM_LUMA_MODE as usize];
    macro_rules! eval_mode {
        ($m:expr) => {{
            let mi = $m as usize;
            if mi < NUM_LUMA_MODE as usize && !done[mi] {
                done[mi] = true;
                let mode = mi as u8;
                predict_into(
                    &mut sc.pred_buf[..nn],
                    &mut sc.ang_scratch,
                    filt_y.as_ref(),
                    mode,
                    w,
                    h,
                    &refs_y,
                    bit_depth,
                    true,
                );
                for i in 0..nn {
                    sc.res_buf[i] = sblk_y[i] - sc.pred_buf[i];
                }
                sc.scored.push((satd_wh(&sc.res_buf[..nn], w, h), mode));
            }
        }};
    }
    eval_mode!(PLANAR_IDX);
    eval_mode!(DC_IDX);
    for &m in mpm.iter() {
        eval_mode!(m);
    }
    let mut a = 2u8;
    while a <= 66 {
        eval_mode!(a);
        a += 2;
    }
    sc.scored.sort_unstable_by_key(|&(s, _)| s);
    if let Some(bm) = sc.scored.iter().map(|&(_, m)| m).find(|&m| m >= 2) {
        if bm > 2 {
            eval_mode!(bm - 1);
        }
        if bm < 66 {
            eval_mode!(bm + 1);
        }
    }
    sc.scored.sort_unstable_by_key(|&(s, _)| s);
    let k = sc.scored.len().min(RDO_CANDIDATES);

    // Stage 2: rate-distortion refinement (rate uses the fixed rd context).
    let lambda = 0.57f64 * 2f64.powf((qp as f64 - 12.0) / 3.0);
    let mut best_mode = sc.scored[0].1;
    predict_into(
        &mut sc.best_pred[..nn],
        &mut sc.ang_scratch,
        filt_y.as_ref(),
        best_mode,
        w,
        h,
        &refs_y,
        bit_depth,
        true,
    );
    let mut best_rd = f64::MAX;
    let mut best_levels: Vec<i32> = Vec::new();
    for i in 0..k {
        if lossless {
            break;
        }
        let mode = sc.scored[i].1;
        predict_into(
            &mut sc.pred_buf[..nn],
            &mut sc.ang_scratch,
            filt_y.as_ref(),
            mode,
            w,
            h,
            &refs_y,
            bit_depth,
            true,
        );
        let levels = analyze_wh(
            sblk_y,
            &sc.pred_buf[..nn],
            w,
            h,
            qp,
            bit_depth,
            if sh.rdoq { lambda } else { 0.0 },
            0,
            0,
            sh.dep_quant,
        );
        if sh.dep_quant {
            crate::transform::dequantize_dq_wh_into(
                &mut sc.deq_buf[..nn],
                &levels,
                crate::residual::scan_coords(w, h),
                w,
                h,
                qp,
                bit_depth,
            );
        } else {
            sc.lv_buf.clear();
            sc.lv_buf.extend(levels.iter().map(|&l| l as i16));
            crate::transform::dequantize_wh_into(
                &mut sc.deq_buf[..nn],
                &sc.lv_buf,
                w,
                h,
                qp,
                bit_depth,
            );
        }
        inv_transform_wh_into(&mut sc.inv_buf[..nn], &sc.deq_buf[..nn], w, h, bit_depth);
        let ssd: i64 = (0..nn)
            .map(|i| {
                let d = (sblk_y[i] - (sc.pred_buf[i] + sc.inv_buf[i]).clamp(0, max_val)) as i64;
                d * d
            })
            .sum();
        let mut tctx = ctx.clone();
        let tu = TuCoeffs {
            tree: TreeType::Single,
            luma: &levels,
            lw: w,
            lh: h,
            chroma: None,
            lossless: false,
            luma_bdpcm: false,
            chroma_bdpcm: false,
            luma_ts: false,
            chroma_ts: false,
            code_dqp: None,
            dep_quant: sh.dep_quant,
        };
        let bits = RD_ENC.with(|c| {
            let tenc = &mut *c.borrow_mut();
            tenc.reset();
            encode_luma_mode(tenc, &mut tctx, &mpm, mode);
            encode_transform_tree(tenc, &mut tctx, &tu);
            tenc.flushed_len() * 8
        }) as f64;
        let rd = ssd as f64 + lambda * bits;
        if rd < best_rd {
            best_rd = rd;
            best_mode = mode;
            sc.best_pred[..nn].copy_from_slice(&sc.pred_buf[..nn]);
            best_levels = levels;
        }
    }
    // Lossy transform-skip trial.
    let mut luma_ts = false;
    if !lossless && w <= 32 && h <= 32 {
        let r: Vec<i32> = sblk_y
            .iter()
            .zip(&sc.best_pred[..nn])
            .map(|(&s, &p)| s - p)
            .collect();
        let ts_lv16 = quantize_ts_wh(&r, w, h, qp);
        let ts_levels: Vec<i32> = ts_lv16[..nn].iter().map(|&l| l as i32).collect();
        let deq = dequantize_ts_wh(&ts_lv16, w, h, qp);
        let ssd: i64 = (0..nn)
            .map(|i| {
                let d = (sblk_y[i] - (sc.best_pred[i] + deq[i]).clamp(0, max_val)) as i64;
                d * d
            })
            .sum();
        let mut tctx = ctx.clone();
        let tu = TuCoeffs {
            tree: TreeType::Single,
            luma: &ts_levels,
            lw: w,
            lh: h,
            chroma: None,
            lossless: false,
            luma_bdpcm: false,
            chroma_bdpcm: false,
            luma_ts: true,
            chroma_ts: false,
            code_dqp: None,
            dep_quant: sh.dep_quant,
        };
        let bits = RD_ENC.with(|c| {
            let tenc = &mut *c.borrow_mut();
            tenc.reset();
            encode_luma_mode(tenc, &mut tctx, &mpm, best_mode);
            encode_transform_tree(tenc, &mut tctx, &tu);
            tenc.flushed_len() * 8
        }) as f64;
        let ts_rd = ssd as f64 + lambda * bits;
        if ts_rd < best_rd {
            best_levels = ts_levels;
            luma_ts = true;
        }
    }
    // Lossless luma BDPCM / mode by true coded cost (fixed rd context).
    let mut luma_bdpcm = 0u8;
    let mut luma_dir_mode = best_mode;
    let luma_levels: Vec<i32> = if lossless {
        let leaf_bits = |bdpcm: u8, mode: u8, levels: &[i32]| -> usize {
            let mut tctx = ctx.clone();
            let tu = TuCoeffs {
                tree: TreeType::Single,
                luma: levels,
                lw: w,
                lh: h,
                chroma: None,
                lossless: true,
                luma_bdpcm: bdpcm != 0,
                chroma_bdpcm: false,
                luma_ts: false,
                chroma_ts: false,
                code_dqp: None,
                dep_quant: sh.dep_quant,
            };
            RD_ENC.with(|c| {
                let tenc = &mut *c.borrow_mut();
                tenc.reset();
                encode_bdpcm_mode(tenc, &mut tctx, bdpcm, true);
                if bdpcm == 0 {
                    encode_luma_mode(tenc, &mut tctx, &mpm, mode);
                }
                encode_transform_tree(tenc, &mut tctx, &tu);
                tenc.flushed_len()
            })
        };
        let mut best_bits = usize::MAX;
        let mut chosen: Vec<i32> = Vec::new();
        for i in 0..k {
            let mode = sc.scored[i].1;
            predict_into(
                &mut sc.pred_buf[..nn],
                &mut sc.ang_scratch,
                filt_y.as_ref(),
                mode,
                w,
                h,
                &refs_y,
                bit_depth,
                true,
            );
            let levels: Vec<i32> = sblk_y
                .iter()
                .zip(&sc.pred_buf[..nn])
                .map(|(&s, &p)| s - p)
                .collect();
            let bits = leaf_bits(0, mode, &levels);
            if bits < best_bits {
                best_bits = bits;
                best_mode = mode;
                luma_bdpcm = 0;
                luma_dir_mode = mode;
                chosen = levels;
            }
        }
        let hor = dpcm_levels_wh(sblk_y, &refs_y, w, h, 1);
        let bh = leaf_bits(1, HOR_IDX, &hor);
        if bh < best_bits {
            best_bits = bh;
            luma_bdpcm = 1;
            luma_dir_mode = HOR_IDX;
            chosen = hor;
        }
        let ver = dpcm_levels_wh(sblk_y, &refs_y, w, h, 2);
        if leaf_bits(2, VER_IDX, &ver) < best_bits {
            luma_bdpcm = 2;
            luma_dir_mode = VER_IDX;
            chosen = ver;
        }
        chosen
    } else {
        best_levels
    };

    let mut lfnst_idx = 0u8;
    let mut lfnst_mode = 0usize;
    let mut lfnst_levels: Vec<i32> = Vec::new();
    // ---- MTS (explicit DST-VII/DCT-VIII) trial state ----
    // Chosen per-TU transform: 0 = DCT-II (default), 1..=4 = the DST7/DCT8 pairs.
    // Mutually exclusive with LFNST and transform-skip. Filled by the MTS trial
    // below; the winning levels (if any) replace the DCT-II luma levels.
    let mut luma_mts_idx = 0u8;
    let mut mts_levels: Vec<i32> = Vec::new();
    if sh.lfnst && !lossless && !luma_ts && w <= 32 && h <= 32 {
        thread_local! {
            static LF_TRIAL: std::cell::RefCell<(Box<[i32; 4096]>, Box<[i32; 4096]>,Box<[i32; 4096]>)> =
                std::cell::RefCell::new((Box::new([0; 64 * 64]), Box::new([0; 64 * 64]), Box::new([0; 64 * 64])));
        }
        let lf_mode = crate::lfnst::lfnst_intra_mode(crate::predict::lfnst_wide_angle(
            w,
            h,
            best_mode as i32,
        ));
        let lf_lambda = if sh.rdoq { lambda } else { 0.0 };
        let mut best_lf_rd = best_rd;
        LF_TRIAL.with(|c| {
            let (res, dct, coeff) = &mut *c.borrow_mut();
            // Forward DCT of the chosen mode's residual, computed once.
            for (r, (&s, &p)) in res[..nn]
                .iter_mut()
                .zip(sblk_y.iter().zip(&sc.best_pred[..nn]))
            {
                *r = s - p;
            }
            fwd_transform_wh_into(&mut dct[..nn], &res[..nn], w, h, bit_depth);
            for idx in 1u8..=2 {
                coeff[..nn].copy_from_slice(&dct[..nn]);
                crate::lfnst::apply_fwd_lfnst(&mut coeff[..nn], w, h, lf_mode, idx as usize);
                let lv16 = if lf_lambda > 0.0 {
                    rdoq_wh(&coeff[..nn], w, h, qp, bit_depth, lf_lambda)
                } else {
                    quantize_wh(&coeff[..nn], w, h, qp, bit_depth)
                };
                let lv: Vec<i32> = lv16[..nn].iter().map(|&l| l as i32).collect();
                // Need a coeff beyond DC, else the LFNST presence condition
                // (lfnstLastScanPos) can't be met and the index can't be signalled.
                if crate::residual::last_sig_scan_pos(&lv, w, h, crate::residual::Component::Luma)
                    < 1
                {
                    continue;
                }
                let rec = reconstruct_block_wh(
                    &sc.best_pred[..nn],
                    &lv,
                    w,
                    h,
                    qp,
                    bit_depth,
                    max_val,
                    false,
                    false,
                    idx,
                    lf_mode,
                    sh.dep_quant,
                    0,
                );
                let ssd: i64 = (0..nn)
                    .map(|i| {
                        let d = (sblk_y[i] - rec[i]) as i64;
                        d * d
                    })
                    .sum();
                let mut tctx = ctx.clone();
                let tu = TuCoeffs {
                    tree: TreeType::Single,
                    luma: &lv,
                    lw: w,
                    lh: h,
                    chroma: None,
                    lossless: false,
                    luma_bdpcm: false,
                    chroma_bdpcm: false,
                    luma_ts: false,
                    chroma_ts: false,
                    code_dqp: None,
                    dep_quant: sh.dep_quant,
                };
                let bits = RD_ENC.with(|c| {
                    let tenc = &mut *c.borrow_mut();
                    tenc.reset();
                    encode_luma_mode(tenc, &mut tctx, &mpm, best_mode);
                    encode_transform_tree(tenc, &mut tctx, &tu);
                    tenc.flushed_len() * 8
                }) as f64;
                // Approximate the residual_lfnst_mode cost (1 bin for idx 1, 2 for idx 2).
                let rd = ssd as f64 + lambda * (bits + idx as f64);
                if rd < best_lf_rd {
                    best_lf_rd = rd;
                    lfnst_idx = idx;
                    lfnst_mode = lf_mode;
                    lfnst_levels = lv;
                } else if idx == 1 && lfnst_idx == 0 {
                    // The first kernel did not improve on no-LFNST; the second
                    // (a different basis on the same residual) almost never does,
                    // so skip it rather than pay another full RD trial.
                    break;
                }
            }
        });
    }

    if sh.mts
        && !lossless
        && !luma_ts
        && lfnst_idx == 0
        && w <= 32
        && h <= 32
        && matches!(w, 4 | 8 | 16 | 32)
        && matches!(h, 4 | 8 | 16 | 32)
    {
        thread_local! {
            static MTS_TRIAL: std::cell::RefCell<(Box<[i32; 4096]>, Box<[i32; 4096]>, Box<[i32; 4096]>, Box<[i32; 4096]>)> =
                std::cell::RefCell::new((Box::new([0; 64 * 64]), Box::new([0; 64 * 64]), Box::new([0; 64 * 64]), Box::new([0; 64 * 64])));
        }
        let mts_lambda = if sh.rdoq { lambda } else { 0.0 };
        // Skip the trial for blocks the DCT-II path already codes very cheaply
        // (few coefficients): MTS cannot help and would usually fail the
        // signallable gate anyway.
        let dct2_nz = luma_levels.iter().filter(|&&v| v != 0).count();
        if dct2_nz >= 2 {
            let escale = crate::transform::err_scale(w, h, bit_depth);
            // Cheap rate proxy (ranking only): a VLC-like per-coefficient cost.
            let proxy_bits = |lv: &[i32]| -> f64 {
                let mut b = 0.0f64;
                for &l in &lv[..nn] {
                    if l != 0 {
                        let a = l.unsigned_abs();
                        b += 3.0 + 2.0 * (32 - a.leading_zeros()) as f64;
                    }
                }
                b
            };
            let mut best_cand = 0u8;
            let mut best_proxy = f64::MAX;
            let mut best_lv: Vec<i32> = Vec::new();
            MTS_TRIAL.with(|c| {
                let (res, coeff, tmp_dst7, tmp_dct8) = &mut *c.borrow_mut();
                for (r, (&s, &p)) in res[..nn]
                    .iter_mut()
                    .zip(sblk_y.iter().zip(&sc.best_pred[..nn]))
                {
                    *r = s - p;
                }
                crate::transform::fwd_mts_pass1_into(
                    &mut tmp_dst7[..nn],
                    &res[..nn],
                    w,
                    h,
                    bit_depth,
                    1,
                );
                crate::transform::fwd_mts_pass1_into(
                    &mut tmp_dct8[..nn],
                    &res[..nn],
                    w,
                    h,
                    bit_depth,
                    2,
                );
                for cand in 1u8..=4 {
                    let (th, tv) = crate::transform::mts_to_types(cand);
                    let tmp = if th == 1 {
                        &tmp_dst7[..nn]
                    } else {
                        &tmp_dct8[..nn]
                    };
                    crate::transform::fwd_mts_pass2_into(&mut coeff[..nn], tmp, w, h, th, tv);
                    let lv16 = quantize_wh(&coeff[..nn], w, h, qp, bit_depth);
                    let lv: Vec<i32> = lv16[..nn].iter().map(|&l| l as i32).collect();
                    if !crate::residual::mts_signallable(&lv, w, h) {
                        continue;
                    }
                    let deq = dequantize_wh(&lv16, w, h, qp, bit_depth);
                    let dist: f64 = coeff[..nn]
                        .iter()
                        .zip(&deq[..nn])
                        .map(|(&cc, &dd)| {
                            let e = (cc - dd) as f64;
                            e * e
                        })
                        .sum::<f64>()
                        * escale;
                    let proxy = dist + lambda * proxy_bits(&lv);
                    if proxy < best_proxy {
                        best_proxy = proxy;
                        best_cand = cand;
                        best_lv = lv;
                    }
                }
                // Refine only the winning pair with RDOQ (when enabled) for the
                // actual coded levels, reusing the shared pass-1 buffer.
                if best_cand > 0 && mts_lambda > 0.0 {
                    let (th, tv) = crate::transform::mts_to_types(best_cand);
                    let tmp = if th == 1 {
                        &tmp_dst7[..nn]
                    } else {
                        &tmp_dct8[..nn]
                    };
                    crate::transform::fwd_mts_pass2_into(&mut coeff[..nn], tmp, w, h, th, tv);
                    let lv16 = rdoq_wh(&coeff[..nn], w, h, qp, bit_depth, mts_lambda);
                    let lv: Vec<i32> = lv16[..nn].iter().map(|&l| l as i32).collect();
                    // RDOQ may zero the block below the signallable threshold.
                    if crate::residual::mts_signallable(&lv, w, h) {
                        best_lv = lv;
                    } else {
                        best_cand = 0;
                    }
                }
            });
            // Pass 2: only the winning pair pays the full reconstruction + RD encode.
            if best_cand > 0 {
                let rec = reconstruct_block_wh(
                    &sc.best_pred[..nn],
                    &best_lv,
                    w,
                    h,
                    qp,
                    bit_depth,
                    max_val,
                    false,
                    false,
                    0,
                    0,
                    sh.dep_quant,
                    best_cand,
                );
                let ssd: i64 = (0..nn)
                    .map(|i| {
                        let d = (sblk_y[i] - rec[i]) as i64;
                        d * d
                    })
                    .sum();
                let mut tctx = ctx.clone();
                let tu = TuCoeffs {
                    tree: TreeType::Single,
                    luma: &best_lv,
                    lw: w,
                    lh: h,
                    chroma: None,
                    lossless: false,
                    luma_bdpcm: false,
                    chroma_bdpcm: false,
                    luma_ts: false,
                    chroma_ts: false,
                    code_dqp: None,
                    dep_quant: sh.dep_quant,
                };
                let bits = RD_ENC.with(|c| {
                    let tenc = &mut *c.borrow_mut();
                    tenc.reset();
                    encode_luma_mode(tenc, &mut tctx, &mpm, best_mode);
                    encode_transform_tree(tenc, &mut tctx, &tu);
                    tenc.flushed_len() * 8
                }) as f64;
                let mts_bin_len = [2u32, 3, 4, 4][(best_cand - 1) as usize] as f64;
                let rd = ssd as f64 + lambda * (bits + mts_bin_len);
                if rd < best_rd {
                    luma_mts_idx = best_cand;
                    mts_levels = best_lv;
                }
            }
        }
    }

    // ---- Luma reconstruction + chroma ----
    // Single-tree CCLM predicts chroma from the *current* CU's reconstructed
    // luma, so in that path the luma recon is committed to the recon plane
    // before chroma analysis. The LFNST single-tree gate (luma + chroma must
    // both be low-frequency-confined) is evaluated after chroma; if it reverts
    // lfnst_idx the luma recon changes, so the luma is re-committed and — when
    // chroma used CCLM — chroma is re-analysed against the corrected luma.
    macro_rules! commit_luma {
        ($lf:expr) => {{
            let lf: u8 = $lf;
            let lv: &[i32] = if luma_mts_idx > 0 {
                &mts_levels
            } else if lf > 0 {
                &lfnst_levels
            } else {
                &luma_levels
            };
            if lossless {
                place_atomic_wh(&pl.rec_y, &pl.av_y, cw, x, y, w, h, &sc.sblk_y[..w * h]);
            } else {
                reconstruct_atomic_wh(
                    &pl.rec_y,
                    &pl.av_y,
                    cw,
                    x,
                    y,
                    w,
                    h,
                    &sc.best_pred[..nn],
                    lv,
                    qp,
                    bit_depth,
                    max_val,
                    lossless,
                    luma_ts,
                    lf,
                    lfnst_mode,
                    sh.dep_quant,
                    luma_mts_idx,
                );
            }
        }};
    }
    let chroma_dec = if tree == TreeType::Luma {
        // Dual-tree luma phase: no chroma. LFNST gate on luma alone, then commit.
        if lfnst_idx > 0
            && !crate::residual::lfnst_present(
                sh.lfnst,
                w,
                h,
                (lfnst_levels.as_slice(), false),
                None,
            )
        {
            lfnst_idx = 0;
        }
        commit_luma!(lfnst_idx);
        None
    } else {
        // Single tree: commit luma first so CCLM can read the collocated luma.
        commit_luma!(lfnst_idx);
        let mut cd = analyze_chroma(
            sh,
            pl,
            sc,
            x,
            y,
            w,
            h,
            qp,
            lambda,
            max_val,
            luma_dir_mode,
            sh.cclm,
        );
        // LFNST single-tree gate: the index can only be signalled if every coded
        // component (luma + chroma) satisfies the presence conditions.
        if lfnst_idx > 0 {
            let chroma = cd.as_ref().map(|c| {
                (
                    c.cb_levels.as_slice(),
                    c.cr_levels.as_slice(),
                    c.ccw,
                    c.cch,
                    c.chroma_ts || c.chroma_bdpcm != 0,
                )
            });
            if !crate::residual::lfnst_present(
                sh.lfnst,
                w,
                h,
                (lfnst_levels.as_slice(), false),
                chroma,
            ) {
                lfnst_idx = 0;
                commit_luma!(0); // luma recon changed; re-commit without LFNST
                if cd
                    .as_ref()
                    .is_some_and(|c| crate::intra::is_cclm_mode(c.chroma_mode))
                {
                    // CCLM read the now-stale LFNST luma; re-fit against the corrected recon.
                    cd = analyze_chroma(
                        sh,
                        pl,
                        sc,
                        x,
                        y,
                        w,
                        h,
                        qp,
                        lambda,
                        max_val,
                        luma_dir_mode,
                        sh.cclm,
                    );
                }
            }
        }
        cd
    };
    pl.modes
        .set_mode_rect(x as u32, y as u32, w as u32, h as u32, luma_dir_mode);

    let luma_levels = if luma_mts_idx > 0 {
        mts_levels
    } else if lfnst_idx > 0 {
        lfnst_levels
    } else {
        luma_levels
    };
    LeafDecision {
        x,
        y,
        w,
        h,
        luma_bdpcm,
        best_mode,
        luma_dir_mode,
        luma_ts,
        luma_levels,
        lfnst_idx,
        luma_mts_idx,
        chroma: chroma_dec,
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze_chroma(
    sh: &LeafShared,
    pl: &Planes,
    sc: &mut LeafScratch,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    qp: u8,
    lambda: f64,
    max_val: i32,
    dm_mode: u8,
    cclm: bool,
) -> Option<ChromaDecision> {
    if !sh.has_chroma {
        return None;
    }
    // Reused dequant/inverse/level scratch for the chroma RD distortion checks
    // (avoids two 16 KiB fixed-array returns per component, per candidate).
    thread_local! {
        static CHR: std::cell::RefCell<(Vec<i16>, Vec<i32>, Vec<i32>)> =
            std::cell::RefCell::new((Vec::with_capacity(64 * 64), vec![0; 64 * 64], vec![0; 64 * 64]));
    }
    let bit_depth = sh.bit_depth;
    let lossless = sh.lossless;
    let (sub_w, sub_h) = (sh.sub_w, sh.sub_h);
    let (cwc, chc) = (sh.cwc, sh.chc);
    let ctx = sh.rd_ctx;
    let nn = w * h;
    let (ccw, cch) = (w / sub_w, h / sub_h);
    let (cx, cy) = (x / sub_w, y / sub_h);
    let refs_cb =
        gather_refs_atomic_wh(&pl.rec_cb, &pl.av_cb, cwc, chc, cx, cy, ccw, cch, bit_depth);
    let refs_cr =
        gather_refs_atomic_wh(&pl.rec_cr, &pl.av_cr, cwc, chc, cx, cy, ccw, cch, bit_depth);
    extract_block_wh_into(
        &mut sc.sblk_cb[..ccw * cch],
        sh.src_cb,
        cwc,
        cx,
        cy,
        ccw,
        cch,
    );
    extract_block_wh_into(
        &mut sc.sblk_cr[..ccw * cch],
        sh.src_cr,
        cwc,
        cx,
        cy,
        ccw,
        cch,
    );
    let sblk_cb = &sc.sblk_cb[..ccw * cch];
    let sblk_cr = &sc.sblk_cr[..ccw * cch];

    let mut chroma_cands = vec![dm_mode];
    for m in chroma_cand_modes(dm_mode) {
        chroma_cands.push(m);
    }
    let mut chroma_mode = dm_mode;
    let mut best_chroma_cost = i64::MAX;
    let ncc = ccw * cch;
    let is_422 = sub_w == 2 && sub_h == 1;
    let pmode = |m: u8| if is_422 { chroma_422_mode(m) } else { m };
    // CCLM candidates (LT always; MDLM-T when an above template exists; MDLM-L
    // when a left template exists). Each is fitted from the reconstructed
    // co-located luma + chroma neighbours and scored by SATD; the best is
    // carried as (mode, cost, pred_cb, pred_cr) for comparison with the
    // angular/DM candidates below.
    let cclm_eval: Option<(u8, i64, Vec<i32>, Vec<i32>)> = if cclm {
        let (lw, lh) = (sh.cw, sh.ch);
        let (cwc2, chc2) = (cwc, chc);
        let luma = |xx: isize, yy: isize| {
            let xx = xx.clamp(0, lw as isize - 1) as usize;
            let yy = yy.clamp(0, lh as isize - 1) as usize;
            pl.rec_y[yy * lw + xx].load(std::sync::atomic::Ordering::Relaxed)
        };
        let cba = |xx: isize, yy: isize| {
            let xx = xx.clamp(0, cwc2 as isize - 1) as usize;
            let yy = yy.clamp(0, chc2 as isize - 1) as usize;
            pl.rec_cb[yy * cwc2 + xx].load(std::sync::atomic::Ordering::Relaxed)
        };
        let cra = |xx: isize, yy: isize| {
            let xx = xx.clamp(0, cwc2 as isize - 1) as usize;
            let yy = yy.clamp(0, chc2 as isize - 1) as usize;
            pl.rec_cr[yy * cwc2 + xx].load(std::sync::atomic::Ordering::Relaxed)
        };
        let avail = |xx: usize, yy: usize| {
            pl.av_cb[yy * cwc2 + xx].load(std::sync::atomic::Ordering::Relaxed) != 0
        };
        let first_row = (y & (crate::partition::CTU_SIZE as usize - 1)) == 0;
        let (above, left) = (cy > 0, cx > 0);
        let uw = 4 >> if sub_w == 2 { 1 } else { 0 };
        let uh = 4 >> if sub_h == 2 { 1 } else { 0 };
        // Consecutive available above-right / below-left chroma units.
        let avai_ar = if above {
            let mut n = 0;
            for u in 0..ccw / uw {
                let col = cx + ccw + u * uw;
                if col >= cwc2 || !avail(col, cy - 1) {
                    break;
                }
                n += 1;
            }
            n
        } else {
            0
        };
        let avai_bl = if left {
            let mut n = 0;
            for u in 0..cch / uh {
                let row = cy + cch + u * uh;
                if row >= chc2 || !avail(cx - 1, row) {
                    break;
                }
                n += 1;
            }
            n
        } else {
            0
        };
        // CCLM candidates: LT always, plus the directional MDLM modes when the
        // corresponding template exists (MDLM-T needs an above template, MDLM-L a
        // left one). All three are VTM bit-exact.
        let mut cands = vec![crate::intra::CCLM_LT_MODE];
        if above {
            cands.push(crate::intra::CCLM_T_MODE);
        }
        if left {
            cands.push(crate::intra::CCLM_L_MODE);
        }
        let mut best: Option<(u8, i64, Vec<i32>, Vec<i32>)> = None;
        for &m in &cands {
            let (pcb, pcr) = crate::cclm::cclm_predict(
                luma, cba, cra, x, y, cx, cy, ccw, cch, sub_w, sub_h, above, left, first_row, m,
                avai_ar, avai_bl, max_val, bit_depth,
            );
            let mut res = vec![0i32; ncc];
            for i in 0..ncc {
                res[i] = sblk_cb[i] - pcb[i];
            }
            let mut cost = satd_wh(&res, ccw, cch);
            for i in 0..ncc {
                res[i] = sblk_cr[i] - pcr[i];
            }
            cost += satd_wh(&res, ccw, cch);
            if best.as_ref().is_none_or(|b| cost < b.1) {
                best = Some((m, cost, pcb, pcr));
            }
        }
        best
    } else {
        None
    };
    for &cm in &chroma_cands {
        let pm = pmode(cm);
        predict_into(
            &mut sc.cpred_cb[..ncc],
            &mut sc.cscratch,
            None,
            pm,
            ccw,
            cch,
            &refs_cb,
            bit_depth,
            false,
        );
        predict_into(
            &mut sc.cpred_cr[..ncc],
            &mut sc.cscratch,
            None,
            pm,
            ccw,
            cch,
            &refs_cr,
            bit_depth,
            false,
        );
        #[allow(clippy::needless_range_loop)]
        for i in 0..ncc {
            sc.cres[i] = sblk_cb[i] - sc.cpred_cb[i];
        }
        let mut cost = satd_wh(&sc.cres[..ncc], ccw, cch);
        #[allow(clippy::needless_range_loop)]
        for i in 0..ncc {
            sc.cres[i] = sblk_cr[i] - sc.cpred_cr[i];
        }
        cost += satd_wh(&sc.cres[..ncc], ccw, cch);
        if cost < best_chroma_cost {
            best_chroma_cost = cost;
            chroma_mode = cm;
        }
    }

    if let Some((m, ccost, _, _)) = &cclm_eval
        && *ccost < best_chroma_cost
    {
        chroma_mode = *m;
    }

    if crate::intra::is_cclm_mode(chroma_mode) {
        let (_, _, pcb, pcr) = cclm_eval.unwrap();
        sc.cpred_cb[..ncc].copy_from_slice(&pcb);
        sc.cpred_cr[..ncc].copy_from_slice(&pcr);
    } else {
        predict_into(
            &mut sc.cpred_cb[..ncc],
            &mut sc.cscratch,
            None,
            pmode(chroma_mode),
            ccw,
            cch,
            &refs_cb,
            bit_depth,
            false,
        );
        predict_into(
            &mut sc.cpred_cr[..ncc],
            &mut sc.cscratch,
            None,
            pmode(chroma_mode),
            ccw,
            cch,
            &refs_cr,
            bit_depth,
            false,
        );
    }
    let mut chroma_bdpcm = 0u8;
    let mut chroma_ts = false;
    let (cb_levels, cr_levels) = if lossless {
        let cb_n: Vec<i32> = sblk_cb
            .iter()
            .zip(&sc.cpred_cb[..ncc])
            .map(|(&s, &p)| s - p)
            .collect();
        let cr_n: Vec<i32> = sblk_cr
            .iter()
            .zip(&sc.cpred_cr[..ncc])
            .map(|(&s, &p)| s - p)
            .collect();
        let cb_h = dpcm_levels_wh(sblk_cb, &refs_cb, ccw, cch, 1);
        let cr_h = dpcm_levels_wh(sblk_cr, &refs_cr, ccw, cch, 1);
        let cb_v = dpcm_levels_wh(sblk_cb, &refs_cb, ccw, cch, 2);
        let cr_v = dpcm_levels_wh(sblk_cr, &refs_cr, ccw, cch, 2);
        let cost = |a: &[i32], b: &[i32]| {
            a.iter()
                .chain(b)
                .map(|&v| v.unsigned_abs() as u64)
                .sum::<u64>()
        };
        let (c_n, c_h, c_v) = (cost(&cb_n, &cr_n), cost(&cb_h, &cr_h), cost(&cb_v, &cr_v));
        if c_h <= c_n && c_h <= c_v {
            chroma_bdpcm = 1;
            (cb_h, cr_h)
        } else if c_v <= c_n {
            chroma_bdpcm = 2;
            (cb_v, cr_v)
        } else {
            (cb_n, cr_n)
        }
    } else {
        let rlambda = if sh.rdoq { lambda } else { 0.0 };
        let cb_dct = residual_levels_wh(
            sblk_cb,
            &sc.cpred_cb[..ncc],
            ccw,
            cch,
            qp,
            bit_depth,
            false,
            rlambda,
            sh.dep_quant,
        );
        let cr_dct = residual_levels_wh(
            sblk_cr,
            &sc.cpred_cr[..ncc],
            ccw,
            cch,
            qp,
            bit_depth,
            false,
            rlambda,
            sh.dep_quant,
        );
        let ssd_dct = CHR.with(|c| {
            let (l16, deq, inv) = &mut *c.borrow_mut();
            let mut s = 0i64;
            for (sblk, pred, lv) in [
                (sblk_cb, &sc.cpred_cb[..ncc], &cb_dct),
                (sblk_cr, &sc.cpred_cr[..ncc], &cr_dct),
            ] {
                l16.clear();
                l16.extend(lv.iter().map(|&x| x as i16));
                crate::transform::dequantize_wh_into(&mut deq[..ncc], l16, ccw, cch, qp, bit_depth);
                inv_transform_wh_into(&mut inv[..ncc], &deq[..ncc], ccw, cch, bit_depth);
                for i in 0..ncc {
                    let d = (sblk[i] - (pred[i] + inv[i]).clamp(0, max_val)) as i64;
                    s += d * d;
                }
            }
            s
        });
        let zero_luma = vec![0i32; nn];
        let chroma_bits = |cb_lv: &[i32], cr_lv: &[i32], ts: bool| -> f64 {
            let mut tctx = ctx.clone();
            let tu = TuCoeffs {
                tree: TreeType::Single,
                luma: &zero_luma,
                lw: w,
                lh: h,
                chroma: Some((cb_lv, cr_lv, ccw, cch)),
                lossless: false,
                luma_bdpcm: false,
                chroma_bdpcm: false,
                luma_ts: false,
                chroma_ts: ts,
                code_dqp: None,
                dep_quant: sh.dep_quant,
            };
            RD_ENC.with(|c| {
                let tenc = &mut *c.borrow_mut();
                tenc.reset();
                encode_transform_tree(tenc, &mut tctx, &tu);
                tenc.flushed_len() * 8
            }) as f64
        };
        let rd_dct = ssd_dct as f64 + lambda * chroma_bits(&cb_dct, &cr_dct, false);
        // Transform-skip is only defined for blocks up to 32 in each
        // dimension; for a 64-wide chroma block (4:4:4 CTU-128) it must be
        // neither chosen *nor* evaluated — its un-zeroed spatial levels are
        // invalid input to the DCT residual coder. Restrict the whole trial.
        let ts_pick = if ccw <= 32 && cch <= 32 {
            let cb_res: Vec<i32> = sblk_cb
                .iter()
                .zip(&sc.cpred_cb[..ncc])
                .map(|(&s, &p)| s - p)
                .collect();
            let cr_res: Vec<i32> = sblk_cr
                .iter()
                .zip(&sc.cpred_cr[..ncc])
                .map(|(&s, &p)| s - p)
                .collect();
            let cb_ts16 = quantize_ts_wh(&cb_res, ccw, cch, qp);
            let cr_ts16 = quantize_ts_wh(&cr_res, ccw, cch, qp);
            let cb_ts: Vec<i32> = cb_ts16[..ncc].iter().map(|&l| l as i32).collect();
            let cr_ts: Vec<i32> = cr_ts16[..ncc].iter().map(|&l| l as i32).collect();
            let ssd_ts = {
                let mut s = 0i64;
                for (sblk, pred, lv16) in [
                    (sblk_cb, &sc.cpred_cb[..ncc], &cb_ts16),
                    (sblk_cr, &sc.cpred_cr[..ncc], &cr_ts16),
                ] {
                    let deq = dequantize_ts_wh(lv16, ccw, cch, qp);
                    for i in 0..ncc {
                        let d = (sblk[i] - (pred[i] + deq[i]).clamp(0, max_val)) as i64;
                        s += d * d;
                    }
                }
                s
            };
            let rd_ts = ssd_ts as f64 + lambda * chroma_bits(&cb_ts, &cr_ts, true);
            if rd_ts < rd_dct {
                Some((cb_ts, cr_ts))
            } else {
                None
            }
        } else {
            None
        };
        match ts_pick {
            Some(lv) => {
                chroma_ts = true;
                lv
            }
            None => (cb_dct, cr_dct),
        }
    };

    // Reconstruct chroma into the planes.
    if lossless {
        place_atomic_wh(&pl.rec_cb, &pl.av_cb, cwc, cx, cy, ccw, cch, sblk_cb);
        place_atomic_wh(&pl.rec_cr, &pl.av_cr, cwc, cx, cy, ccw, cch, sblk_cr);
    } else {
        reconstruct_atomic_wh(
            &pl.rec_cb,
            &pl.av_cb,
            cwc,
            cx,
            cy,
            ccw,
            cch,
            &sc.cpred_cb[..ncc],
            &cb_levels,
            qp,
            bit_depth,
            max_val,
            lossless,
            chroma_ts,
            0,
            0,
            sh.dep_quant,
            0,
        );
        reconstruct_atomic_wh(
            &pl.rec_cr,
            &pl.av_cr,
            cwc,
            cx,
            cy,
            ccw,
            cch,
            &sc.cpred_cr[..ncc],
            &cr_levels,
            qp,
            bit_depth,
            max_val,
            lossless,
            chroma_ts,
            0,
            0,
            sh.dep_quant,
            0,
        );
    }
    let _ = (cx, cy);
    Some(ChromaDecision {
        ccw,
        cch,
        chroma_mode,
        chroma_bdpcm,
        chroma_ts,
        cb_levels,
        cr_levels,
    })
}

fn leaf_has_cbf(d: &LeafDecision) -> bool {
    d.luma_levels.iter().any(|&v| v != 0)
        || d.chroma.as_ref().is_some_and(|c| {
            c.cb_levels.iter().any(|&v| v != 0) || c.cr_levels.iter().any(|&v| v != 0)
        })
}

#[allow(clippy::too_many_arguments)]
fn replay_luma(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    grid: &mut crate::partition::NeighborGrid,
    lossless: bool,
    sps_lfnst: bool,
    dep_quant: bool,
    sps_mts: bool,
    code_dqp: Option<i32>,
    d: &LeafDecision,
) {
    let mpm = build_mpm(
        grid.left_mode_rect(d.x as u32, d.y as u32, d.h as u32),
        grid.above_mode_rect(d.x as u32, d.y as u32, d.w as u32),
    );
    if lossless {
        encode_bdpcm_mode(enc, ctx, d.luma_bdpcm, true);
    }
    if d.luma_bdpcm == 0 {
        encode_luma_mode(enc, ctx, &mpm, d.best_mode);
    }
    grid.set_mode_rect(
        d.x as u32,
        d.y as u32,
        d.w as u32,
        d.h as u32,
        d.luma_dir_mode,
    );
    let tu = TuCoeffs {
        tree: TreeType::Luma,
        luma: &d.luma_levels,
        lw: d.w,
        lh: d.h,
        chroma: None,
        lossless,
        luma_bdpcm: d.luma_bdpcm != 0,
        chroma_bdpcm: false,
        luma_ts: d.luma_ts,
        chroma_ts: false,
        code_dqp,
        dep_quant,
    };
    encode_transform_tree(enc, ctx, &tu);
    if sps_lfnst && !lossless {
        let lts = d.luma_ts || d.luma_bdpcm != 0;
        if crate::residual::lfnst_present(
            sps_lfnst,
            d.w,
            d.h,
            (d.luma_levels.as_slice(), lts),
            None,
        ) {
            encode_lfnst_idx(enc, ctx, d.lfnst_idx, true);
        }
    }
    if sps_mts
        && !lossless
        && d.lfnst_idx == 0
        && !d.luma_ts
        && d.luma_bdpcm == 0
        && d.w <= 32
        && d.h <= 32
        && crate::residual::mts_signallable(&d.luma_levels, d.w, d.h)
    {
        encode_mts_idx(enc, ctx, d.luma_mts_idx);
    }
}

/// Replay the chroma half for a dual-tree chroma pass: chroma mode (DM source =
/// `dm_mode`, the co-located luma mode) + BDPCM in lossless, then the chroma-only
/// transform unit. No luma, no LFNST/MTS (chroma LFNST is out of scope for v1).
fn replay_chroma(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    lossless: bool,
    sps_lfnst: bool,
    dep_quant: bool,
    dm_mode: u8,
    cclm: bool,
    c: &ChromaDecision,
) {
    if lossless {
        encode_bdpcm_mode(enc, ctx, c.chroma_bdpcm, false);
    }
    if c.chroma_bdpcm == 0 {
        encode_chroma_mode(enc, ctx, dm_mode, c.chroma_mode, cclm);
    }
    let tu = TuCoeffs {
        tree: TreeType::Chroma,
        luma: &[],
        lw: c.ccw,
        lh: c.cch,
        chroma: Some((c.cb_levels.as_slice(), c.cr_levels.as_slice(), c.ccw, c.cch)),
        lossless,
        luma_bdpcm: false,
        chroma_bdpcm: c.chroma_bdpcm != 0,
        luma_ts: false,
        chroma_ts: c.chroma_ts,
        code_dqp: None,
        dep_quant,
    };
    encode_transform_tree(enc, ctx, &tu);
    // In the dual tree, chroma carries its own LFNST index (separate from luma).
    // The encoder applies no chroma LFNST, but a conformant decoder still parses
    // lfnst_idx whenever the chroma block's last significant coefficient lies in
    // the low-frequency region, so it must be signalled (as 0) to stay in sync.
    // Uses the separate-tree context (1). Lossless/transform-skip disables LFNST.
    if sps_lfnst && !lossless {
        let cts = c.chroma_ts || c.chroma_bdpcm != 0;
        let chroma = Some((
            c.cb_levels.as_slice(),
            c.cr_levels.as_slice(),
            c.ccw,
            c.cch,
            cts,
        ));
        if crate::residual::lfnst_present(sps_lfnst, c.ccw, c.cch, (&[], false), chroma) {
            encode_lfnst_idx(enc, ctx, 0, true);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_leaf(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    grid: &mut crate::partition::NeighborGrid,
    lossless: bool,
    sps_lfnst: bool,
    dep_quant: bool,
    sps_mts: bool,
    sps_cclm: bool,
    code_dqp: Option<i32>,
    d: &LeafDecision,
) {
    let mpm = build_mpm(
        grid.left_mode_rect(d.x as u32, d.y as u32, d.h as u32),
        grid.above_mode_rect(d.x as u32, d.y as u32, d.w as u32),
    );
    if lossless {
        encode_bdpcm_mode(enc, ctx, d.luma_bdpcm, true);
    }
    if d.luma_bdpcm == 0 {
        encode_luma_mode(enc, ctx, &mpm, d.best_mode);
    }
    grid.set_mode_rect(
        d.x as u32,
        d.y as u32,
        d.w as u32,
        d.h as u32,
        d.luma_dir_mode,
    );
    if let Some(c) = &d.chroma {
        if lossless {
            encode_bdpcm_mode(enc, ctx, c.chroma_bdpcm, false);
        }
        if c.chroma_bdpcm == 0 {
            encode_chroma_mode(enc, ctx, d.luma_dir_mode, c.chroma_mode, sps_cclm);
        }
    }
    let tu = TuCoeffs {
        tree: TreeType::Single,
        luma: &d.luma_levels,
        lw: d.w,
        lh: d.h,
        chroma: d
            .chroma
            .as_ref()
            .map(|c| (c.cb_levels.as_slice(), c.cr_levels.as_slice(), c.ccw, c.cch)),
        lossless,
        luma_bdpcm: d.luma_bdpcm != 0,
        chroma_bdpcm: d.chroma.as_ref().is_some_and(|c| c.chroma_bdpcm != 0),
        luma_ts: d.luma_ts,
        chroma_ts: d.chroma.as_ref().is_some_and(|c| c.chroma_ts),
        code_dqp,
        dep_quant,
    };
    encode_transform_tree(enc, ctx, &tu);

    // residual_lfnst_mode: coded at the end of the coding unit, after the
    // transform tree, iff the coded coefficients satisfy the VTM presence
    // conditions (single tree, intra, no ISP/MIP). Lossless is transform-skip
    // coded, which disables LFNST.
    if sps_lfnst && !lossless {
        let lts = d.luma_ts || d.luma_bdpcm != 0;
        let chroma = d.chroma.as_ref().map(|c| {
            (
                c.cb_levels.as_slice(),
                c.cr_levels.as_slice(),
                c.ccw,
                c.cch,
                c.chroma_ts || c.chroma_bdpcm != 0,
            )
        });
        if crate::residual::lfnst_present(
            sps_lfnst,
            d.w,
            d.h,
            (d.luma_levels.as_slice(), lts),
            chroma,
        ) {
            encode_lfnst_idx(enc, ctx, d.lfnst_idx, false);
        }
    }

    // mts_idx: coded at the coding-unit level after residual_lfnst_mode, iff MTS
    // is enabled, the luma TU is ≤32 with lfnst_idx == 0 and is not transform-
    // skip, and the coded levels are signallable (not DC-only, confined to the
    // top-left 16×16). The decoder reconstructs this same gate from the levels.
    if sps_mts
        && !lossless
        && d.lfnst_idx == 0
        && !d.luma_ts
        && d.luma_bdpcm == 0
        && d.w <= 32
        && d.h <= 32
        && crate::residual::mts_signallable(&d.luma_levels, d.w, d.h)
    {
        encode_mts_idx(enc, ctx, d.luma_mts_idx);
    }
}

/// Code `residual_lfnst_mode` (VTM binarization): a context-coded first bin for
/// `idx > 0`, then a second context-coded bin selecting index 2 over 1.
fn encode_lfnst_idx(enc: &mut CabacEncoder, ctx: &mut Contexts, idx: u8, sep_tree: bool) {
    // VVC: the first bin uses context 1 in a separate (dual) tree, 0 otherwise;
    // the second bin always uses context 2.
    let c0 = if sep_tree { 1 } else { 0 };
    enc.encode_bin((idx > 0) as u8, &mut ctx.lfnst_idx[c0]);
    if idx > 0 {
        enc.encode_bin((idx == 2) as u8, &mut ctx.lfnst_idx[2]);
    }
}

/// Code `mts_idx` (VVC binarization): a context-coded first bin (`idx > 0`),
/// then for a non-DCT2 index a truncated-unary tail over the three remaining
/// contexts — `"10"/"110"/"1110"/"1111"` select idx 1..=4 (the DST7/DCT8 pairs).
fn encode_mts_idx(enc: &mut CabacEncoder, ctx: &mut Contexts, idx: u8) {
    enc.encode_bin((idx > 0) as u8, &mut ctx.mts_idx[0]);
    if idx > 0 {
        let k = idx - 1;
        for i in 0..3u8 {
            let bin = (k > i) as u8;
            enc.encode_bin(bin, &mut ctx.mts_idx[1 + i as usize]);
            if bin == 0 {
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mtt_rd_decide(
    src: &[i32],
    cw: usize,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    qp: u8,
    can: &[bool; 6],
    min_w: u32,
    min_h: u32,
) -> crate::partition::MttSplit {
    use crate::partition::MttSplit::*;
    let allowed = |s: crate::partition::MttSplit| match s {
        None => can[0],
        Quad => can[1],
        BinH => can[2],
        BinV => can[3],
        TriH => can[4],
        TriV => can[5],
    };
    let usable = |s: crate::partition::MttSplit| {
        allowed(s)
            && match s {
                None => true,
                Quad => w / 2 >= min_w && h / 2 >= min_h,
                BinV => w / 2 >= min_w,
                BinH => h / 2 >= min_h,
                TriV => w / 4 >= min_w,
                TriH => h / 4 >= min_h,
            }
    };
    let must_split = w > 64 || h > 64;

    // Block statistics over the in-picture region.
    let ch = src.len() / cw;
    let (xx, yy) = (x as usize, y as usize);
    let bw = (w as usize).min(cw - xx);
    let bh = (h as usize).min(ch - yy);
    let (mut sum, mut sumsq, mut gx, mut gy, mut n) = (0i64, 0i64, 0i64, 0i64, 0i64);
    for j in 0..bh {
        let row = (yy + j) * cw + xx;
        for i in 0..bw {
            let v = src[row + i] as i64;
            sum += v;
            sumsq += v * v;
            n += 1;
            if i + 1 < bw {
                gx += (src[row + i + 1] as i64 - v).abs();
            }
            if j + 1 < bh {
                gy += (src[row + i + cw] as i64 - v).abs();
            }
        }
    }
    let var = if n > 0 {
        sumsq / n - (sum / n) * (sum / n)
    } else {
        0
    };
    let thr = 64 + (qp as i64) * (qp as i64);
    if !must_split && var < thr && allowed(None) {
        return None;
    }

    // Orientation from gradient anisotropy and aspect ratio.
    let vert = (gx as i128) * 4 > (gy as i128) * 5;
    let horz = (gy as i128) * 4 > (gx as i128) * 5;
    let order = if w >= 2 * h {
        [BinV, TriV, Quad, BinH, TriH]
    } else if h >= 2 * w {
        [BinH, TriH, Quad, BinV, TriV]
    } else if vert {
        [BinV, TriV, Quad, BinH, TriH]
    } else if horz {
        [BinH, TriH, Quad, BinV, TriV]
    } else {
        [Quad, BinV, BinH, TriV, TriH]
    };
    for s in order {
        if usable(s) {
            return s;
        }
    }
    if allowed(None) && !must_split {
        return None;
    }
    for s in [Quad, BinV, BinH, TriV, TriH] {
        if allowed(s) {
            return s;
        }
    }
    None
}

fn encode_planes(
    headers: Headers,
    src_y: Vec<i32>,
    src_cb: Vec<i32>,
    src_cr: Vec<i32>,
    threads: usize,
    rdoq: bool,
) -> CoreOutput {
    let chroma = headers.chroma;
    let bit_depth = headers.bit_depth.bits();
    let qp = headers.qp;
    let lossless = headers.lossless;
    let cw = headers.coded_width() as usize;
    let ch = headers.coded_height() as usize;
    let (sub_w, sub_h) = (chroma.sub_w(), chroma.sub_h());
    let has_chroma = !chroma.is_monochrome();
    let cwc = if has_chroma { cw / sub_w } else { 0 };
    let chc = if has_chroma { ch / sub_h } else { 0 };
    let max_val = (1i32 << bit_depth) - 1;

    // ---- Setup: fixed RD context + variance-only split decision ----
    let slice_qp = (qp as i32 - 6 * (bit_depth as i32 - 8)).clamp(0, 63) as u8;
    let rd_ctx = Contexts::new_intra(slice_qp);

    // The quadtree split is a pure function of luma source variance — no
    // reconstruction or CABAC dependency — so the partition is derived up front
    // and is identical in every phase.
    let split_threshold = 64 + (qp as i64) * (qp as i64);
    let decide_fn = |x: u32, y: u32, size: u32| -> bool {
        if size > 64 {
            return true;
        }
        if lossless && size > 32 {
            return true;
        }
        size > 8
            && block_variance(&src_y, cw, x as usize, y as usize, size as usize) > split_threshold
    };

    // ---- Phase 0: partition geometry, leaves grouped by CTU (emission order) ----
    let ctu = CTU_SIZE as usize;
    let ctu_cols = cw.div_ceil(ctu);
    let ctu_rows = ch.div_ceil(ctu);
    let mut ctu_leaves: Vec<Vec<(usize, usize, usize, usize)>> =
        vec![Vec::new(); ctu_cols * ctu_rows];
    if !headers.mtt {
        let mut throwaway = CabacEncoder::new();
        let mut tctx = Contexts::new_intra(slice_qp);
        let _ = code_partitions(
            &mut throwaway,
            &mut tctx,
            cw as u32,
            ch as u32,
            &decide_fn,
            |_enc, _ctx, _grid, x, y, size| {
                let (xu, yu, nu) = (x as usize, y as usize, size as usize);
                let cidx = (yu / ctu) * ctu_cols + (xu / ctu);
                ctu_leaves[cidx].push((xu, yu, nu, nu));
            },
        );
    }
    let total_leaves: usize = ctu_leaves.iter().map(|v| v.len()).sum();

    let qp_bd_offset = 6 * (bit_depth as i32 - 8);
    let base_qpy = qp as i32 - qp_bd_offset; // SliceQpY, unclamped
    let qg = if headers.aq && headers.dual_tree {
        64 // one QG per 64×64 region: each region's luma (incl. cu_qp_delta) is
    // fully coded before its single chroma CU, so chroma QP is unambiguous
    } else if headers.aq && !headers.mtt {
        AQ_QG
    } else {
        ctu
    };
    let qg_cols = cw.div_ceil(qg);
    let qg_rows = ch.div_ceil(qg);
    let qg_dq: Vec<i32> = if headers.aq {
        aq_qg_targets(&src_y, cw, ch, qg, qg_cols, qg_rows, qp)
    } else {
        vec![qp as i32; qg_cols * qg_rows]
    };
    let qg_target_qpy: Vec<i32> = qg_dq.iter().map(|&d| d - qp_bd_offset).collect();
    let qg_qpprime: Vec<u8> = qg_dq
        .iter()
        .map(|&d| d.clamp(0, 63 + qp_bd_offset) as u8)
        .collect();

    let shared = LeafShared {
        src_y: &src_y,
        src_cb: &src_cb,
        src_cr: &src_cr,
        cw,
        ch,
        cwc,
        chc,
        sub_w,
        sub_h,
        has_chroma,
        bit_depth,
        qp,
        lossless,
        rdoq,
        aq: headers.aq,
        lfnst: headers.lfnst,
        dep_quant: headers.dep_quant,
        mts: headers.mts,
        cclm: headers.cclm,
        qg,
        qg_cols,
        qg_qpprime: &qg_qpprime,
        max_val,
        rd_ctx: &rd_ctx,
    };

    // ---- Dual-tree path (intra separate luma/chroma trees) ----
    // Same three-phase structure as the single-tree paths: recon-independent
    // partition geometry is captured up front (Phase 0), the per-leaf analysis +
    // reconstruction runs across the parallel CTU wavefront — each CTU does its
    // luma subtree then its chroma, so chroma DM / CCLM read the just-written luma
    // modes and reconstruction (Phase A) — and a sequential pass replays the dual
    // partition syntax and the captured decisions into CABAC (Phase B). QT-only
    // luma partition, chroma one CU per 64×64, AQ off.
    if headers.dual_tree && has_chroma {
        use crate::partition::{MttCfg, MttSplit, code_partitions_dual};
        fn first_legal(can: &[bool; 6]) -> MttSplit {
            if can[1] {
                return MttSplit::Quad;
            }
            if can[2] {
                return MttSplit::BinH;
            }
            if can[3] {
                return MttSplit::BinV;
            }
            if can[4] {
                return MttSplit::TriH;
            }
            if can[5] {
                return MttSplit::TriV;
            }
            MttSplit::None
        }
        let planes = Planes::new(cw, ch, cwc, chc);
        let lambda = 0.57f64 * 2f64.powf((qp as f64 - 12.0) / 3.0);
        // Luma may use the full QT+MTT partition (same heuristic and SPS limits as
        // the single tree) when MTT is enabled; chroma stays one CU per 64×64.
        let luma_cfg = if headers.mtt {
            crate::partition::mtt_cfg()
        } else {
            MttCfg {
                max_mtt_depth: 0,
                min_qt_size: 1 << crate::headers::LOG2_MIN_CB_SIZE,
                max_bt_size: 64,
                max_tt_size: 64,
            }
        };
        let chroma_cfg = MttCfg {
            max_mtt_depth: 0,
            min_qt_size: 64,
            max_bt_size: 64,
            max_tt_size: 64,
        };
        // Both trees' partition geometry is recon-independent (luma = source
        // statistics, chroma = forced boundary splits), so the split closures are
        // shared verbatim by every phase below — guaranteeing the analysis and
        // the CABAC replay traverse identical structures.
        let use_mtt = headers.mtt;
        let luma_decide = |x: u32, y: u32, w: u32, h: u32, can: &[bool; 6]| -> MttSplit {
            if use_mtt {
                // Dual-tree luma is independent of chroma, so the minimum CB is 4.
                return mtt_rd_decide(&src_y, cw, x, y, w, h, qp, can, 4, 4);
            }
            if w > 64 || h > 64 {
                return MttSplit::Quad;
            }
            if !can[0] {
                return first_legal(can);
            }
            if can[1]
                && w > 8
                && block_variance(&src_y, cw, x as usize, y as usize, w as usize) > split_threshold
            {
                MttSplit::Quad
            } else {
                MttSplit::None
            }
        };
        let chroma_decide = |_x: u32, _y: u32, _w: u32, _h: u32, can: &[bool; 6]| -> MttSplit {
            if can[0] {
                MttSplit::None
            } else {
                first_legal(can)
            }
        };

        // ---- Phase 0: per-CTU luma + chroma leaf geometry, in coding order ----
        let nctu = ctu_cols * ctu_rows;
        let mut luma_geom: Vec<Vec<(u32, u32, u32, u32)>> = vec![Vec::new(); nctu];
        let mut chroma_geom: Vec<Vec<(u32, u32, u32, u32)>> = vec![Vec::new(); nctu];
        {
            let mut tenc = CabacEncoder::new();
            let mut tctx = Contexts::new_intra(slice_qp);
            let lg = std::cell::RefCell::new(&mut luma_geom);
            let cg = std::cell::RefCell::new(&mut chroma_geom);
            let ctu_u = ctu as u32;
            code_partitions_dual(
                &mut tenc,
                &mut tctx,
                cw as u32,
                ch as u32,
                luma_cfg,
                chroma_cfg,
                luma_decide,
                |_enc, _ctx, _grid, x, y, w, h| {
                    let cidx = (y as usize / ctu) * ctu_cols + x as usize / ctu;
                    lg.borrow_mut()[cidx].push((x, y, w, h));
                    let _ = ctu_u;
                },
                chroma_decide,
                |_enc, _ctx, _grid, x, y, w, h| {
                    let cidx = (y as usize / ctu) * ctu_cols + x as usize / ctu;
                    cg.borrow_mut()[cidx].push((x, y, w, h));
                },
            );
        }

        // ---- Phase A (luma): parallel wavefront, analyse + reconstruct luma ----
        let luma_res: Vec<Vec<LeafDecision>> =
            run_ctu_wavefront(ctu_cols, ctu_rows, threads, |cidx, sc| {
                let mut ld = Vec::with_capacity(luma_geom[cidx].len());
                for &(x, y, w, h) in &luma_geom[cidx] {
                    ld.push(analyze_leaf(
                        &shared,
                        &planes,
                        sc,
                        x as usize,
                        y as usize,
                        w as usize,
                        h as usize,
                        TreeType::Luma,
                    ));
                }
                ld
            });

        // ---- Adaptive-quant QG resolution (one QG per 64×64 region) ----
        // In the dual tree cu_qp_delta rides in the LUMA tree, coded at the QG's
        // first coefficient-bearing luma CU; chroma QP is derived from that
        // resolved luma QG QP. A QG with no luma cbf resolves to the QP predictor
        // (no delta), and its chroma must be quantised at that predictor — hence
        // chroma analysis waits for this sequential pass. QGs are walked in coding
        // order (CTUs in raster, the four 64×64 regions in z-order) so each QG's
        // left/above neighbours are finalised first. `qg_delta` is what Phase B
        // codes; `qg_qpprime` is the resolved dequant Qp'Y for that QG's chroma.
        let mut qg_map: Vec<i32> = vec![base_qpy; qg_cols * qg_rows];
        let mut qg_delta: Vec<Option<i32>> = vec![None; qg_cols * qg_rows];
        // Per-4×4 luma Qp'Y grid. Mirrors VTM: each luma CU stores the QG QP in
        // effect when it is coded — the predictor for CUs preceding the QG's
        // cu_qp_delta, the resolved target for the delta CU and those after it.
        // A dual-tree chroma CU then takes the QP of the luma CU collocated with
        // its centre (H.266 / VTM `colLumaCu->qp`), not the QG's final QP.
        let cw4 = cw.div_ceil(4);
        let mut luma_qp: Vec<u8> = vec![qp; cw4 * ch.div_ceil(4)];
        if headers.aq {
            // Walk luma leaves in coding order (CTUs raster, regions z-order,
            // leaves within), tracking the QG QP exactly as the decoder will.
            let mut cur_qg = usize::MAX;
            let mut prev = base_qpy;
            let mut cur_qp_y = base_qpy;
            let mut dqp_done = false;
            #[allow(clippy::needless_range_loop)]
            for cidx in 0..nctu {
                for d in &luma_res[cidx] {
                    let (qx, qy) = (d.x & !(qg - 1), d.y & !(qg - 1));
                    let qgi = (qy / qg) * qg_cols + qx / qg;
                    if qgi != cur_qg {
                        prev = cur_qp_y;
                        cur_qp_y = aq_predict_qp(&qg_map, qg_cols, qg, ctu, qx, qy, prev);
                        aq_fill_qg(&mut qg_map, qg_cols, qg, d.x, d.y, d.w, cur_qp_y);
                        dqp_done = false;
                        cur_qg = qgi;
                    }
                    if !dqp_done && leaf_has_cbf(d) {
                        let delta = qg_target_qpy[qgi] - cur_qp_y;
                        qg_delta[qgi] = Some(delta);
                        cur_qp_y = if delta == 0 {
                            cur_qp_y
                        } else {
                            ((cur_qp_y + delta + 64 + 2 * qp_bd_offset) % (64 + qp_bd_offset))
                                - qp_bd_offset
                        };
                        aq_fill_qg(&mut qg_map, qg_cols, qg, d.x, d.y, d.w, cur_qp_y);
                        dqp_done = true;
                    }
                    let lqp = (cur_qp_y + qp_bd_offset).clamp(0, 63 + qp_bd_offset) as u8;
                    let _ = prev;
                    for yy in (d.y / 4)..((d.y + d.h).div_ceil(4)) {
                        for xx in (d.x / 4)..((d.x + d.w).div_ceil(4)) {
                            luma_qp[yy * cw4 + xx] = lqp;
                        }
                    }
                }
            }
        }
        // Collocated luma QP at a chroma CU's centre (luma coords).
        let chroma_qp_at = |x: usize, y: usize, w: usize, h: usize| -> u8 {
            let lx = ((x / sub_w + ((w / sub_w) >> 1)) * sub_w).min(cw - 1);
            let ly = ((y / sub_h + ((h / sub_h) >> 1)) * sub_h).min(ch - 1);
            luma_qp[(ly / 4) * cw4 + lx / 4]
        };

        let chroma_res: Vec<Vec<Option<ChromaDecision>>> =
            run_ctu_wavefront(ctu_cols, ctu_rows, threads, |cidx, sc| {
                let mut cd = Vec::with_capacity(chroma_geom[cidx].len());
                for &(x, y, w, h) in &chroma_geom[cidx] {
                    let cqp = chroma_qp_at(x as usize, y as usize, w as usize, h as usize);
                    let dm = planes.modes.chroma_dm(x, y, w, h);
                    cd.push(analyze_chroma(
                        &shared,
                        &planes,
                        sc,
                        x as usize,
                        y as usize,
                        w as usize,
                        h as usize,
                        cqp,
                        lambda,
                        max_val,
                        dm,
                        headers.cclm,
                    ));
                }
                cd
            });

        let mut enc = CabacEncoder::new();
        let mut ctx = Contexts::new_intra(slice_qp);
        let lcur = std::cell::RefCell::new(vec![0usize; nctu]);
        let ccur = std::cell::RefCell::new(vec![0usize; nctu]);
        let dqp_coded = std::cell::RefCell::new(vec![false; qg_cols * qg_rows]);
        let aq = headers.aq;
        let (luma_leaves, _chroma_leaves) = code_partitions_dual(
            &mut enc,
            &mut ctx,
            cw as u32,
            ch as u32,
            luma_cfg,
            chroma_cfg,
            luma_decide,
            |enc, ctx, grid, x, y, _w, _h| {
                let cidx = (y as usize / ctu) * ctu_cols + x as usize / ctu;
                let k = {
                    let mut c = lcur.borrow_mut();
                    let k = c[cidx];
                    c[cidx] += 1;
                    k
                };
                let d = &luma_res[cidx][k];
                // cu_qp_delta is coded once per QG, at its first cbf luma CU.
                let code_dqp = if aq && leaf_has_cbf(d) {
                    let qgi = (y as usize / qg) * qg_cols + x as usize / qg;
                    let mut dc = dqp_coded.borrow_mut();
                    if !dc[qgi] {
                        dc[qgi] = true;
                        qg_delta[qgi]
                    } else {
                        None
                    }
                } else {
                    None
                };
                replay_luma(
                    enc,
                    ctx,
                    grid,
                    lossless,
                    headers.lfnst,
                    headers.dep_quant,
                    headers.mts,
                    code_dqp,
                    d,
                );
            },
            chroma_decide,
            |enc, ctx, _grid, x, y, w, h| {
                let cidx = (y as usize / ctu) * ctu_cols + x as usize / ctu;
                let k = {
                    let mut c = ccur.borrow_mut();
                    let k = c[cidx];
                    c[cidx] += 1;
                    k
                };
                if let Some(c) = &chroma_res[cidx][k] {
                    let dm = planes.modes.chroma_dm(x, y, w, h);
                    replay_chroma(
                        enc,
                        ctx,
                        lossless,
                        headers.lfnst,
                        headers.dep_quant,
                        dm,
                        headers.cclm,
                        c,
                    );
                }
            },
        );
        enc.encode_terminate(1);
        let slice_data = enc.finish();
        let (rec_y, rec_cb, rec_cr) = planes.into_recon();
        return CoreOutput {
            stream: headers.write_still_picture(&slice_data),
            slice_data,
            rec_y,
            rec_cb,
            rec_cr,
            cw,
            ch,
            cwc,
            chc,
            qp,
            bit_depth,
            leaf_count: luma_leaves.len(),
        };
    }

    if headers.mtt {
        let cfg = crate::partition::mtt_cfg();
        let (min_w, min_h) = (4 * sub_w as u32, 4 * sub_h as u32);
        // Phase 0: per-CTU MTT leaf geometry (x, y, w, h) in z-order.
        let mut mtt_leaves: Vec<Vec<(usize, usize, usize, usize)>> =
            vec![Vec::new(); ctu_cols * ctu_rows];
        {
            let mut throwaway = CabacEncoder::new();
            let mut tctx = Contexts::new_intra(slice_qp);
            let _ = crate::partition::code_partitions_mtt(
                &mut throwaway,
                &mut tctx,
                cw as u32,
                ch as u32,
                cfg,
                |x, y, w, h, can| mtt_rd_decide(&src_y, cw, x, y, w, h, qp, can, min_w, min_h),
                |_enc, _ctx, _grid, x, y, w, h| {
                    let (xu, yu) = (x as usize, y as usize);
                    let cidx = (yu / ctu) * ctu_cols + xu / ctu;
                    mtt_leaves[cidx].push((xu, yu, w as usize, h as usize));
                },
            );
        }
        // Phase A: parallel analysis + reconstruction across the CTU wavefront.
        let planes = Planes::new(cw, ch, cwc, chc);
        let ctu_decisions =
            analyze_ctus_wavefront(&shared, &planes, &mtt_leaves, ctu_cols, ctu_rows, threads);

        // Phase B: sequential CABAC replay + per-CTU-QG cu_qp_delta (subdiv 0).
        let mut enc = CabacEncoder::new();
        let mut ctx = Contexts::new_intra(slice_qp);
        let mut cursor = vec![0usize; ctu_cols * ctu_rows];
        let aq = headers.aq;
        let mut qg_map: Vec<i32> = vec![base_qpy; qg_cols * qg_rows];
        let mut cur_qg: usize = usize::MAX;
        let mut prev_qp: i32 = base_qpy;
        let mut cur_qg_qp_y: i32 = base_qpy;
        let mut qg_dqp_coded = false;
        let leaves = crate::partition::code_partitions_mtt(
            &mut enc,
            &mut ctx,
            cw as u32,
            ch as u32,
            cfg,
            |x, y, w, h, can| mtt_rd_decide(&src_y, cw, x, y, w, h, qp, can, min_w, min_h),
            |enc, ctx, grid, x, y, w, h| {
                let (xu, yu, wu) = (x as usize, y as usize, w as usize);
                let cidx = (yu / ctu) * ctu_cols + xu / ctu;
                let k = cursor[cidx];
                cursor[cidx] += 1;
                let d = &ctu_decisions[cidx][k];
                let _ = (w, h);
                let mut code_dqp = None;
                if aq {
                    let (qx, qy) = (xu & !(qg - 1), yu & !(qg - 1));
                    let qidx = (qy / qg) * qg_cols + qx / qg;
                    if qidx != cur_qg {
                        prev_qp = cur_qg_qp_y;
                        cur_qg_qp_y = aq_predict_qp(&qg_map, qg_cols, qg, ctu, qx, qy, prev_qp);
                        aq_fill_qg(&mut qg_map, qg_cols, qg, xu, yu, wu, cur_qg_qp_y);
                        qg_dqp_coded = false;
                        cur_qg = qidx;
                    }
                    if !qg_dqp_coded && leaf_has_cbf(d) {
                        let target = qg_target_qpy[qidx];
                        let delta = target - cur_qg_qp_y;
                        cur_qg_qp_y = if delta == 0 {
                            cur_qg_qp_y
                        } else {
                            ((cur_qg_qp_y + delta + 64 + 2 * qp_bd_offset) % (64 + qp_bd_offset))
                                - qp_bd_offset
                        };
                        aq_fill_qg(&mut qg_map, qg_cols, qg, xu, yu, wu, cur_qg_qp_y);
                        qg_dqp_coded = true;
                        code_dqp = Some(delta);
                    }
                }
                replay_leaf(
                    enc,
                    ctx,
                    grid,
                    lossless,
                    headers.lfnst,
                    headers.dep_quant,
                    headers.mts,
                    headers.cclm,
                    code_dqp,
                    d,
                );
            },
        );
        enc.encode_terminate(1);
        let slice_data = enc.finish();
        let (rec_y, rec_cb, rec_cr) = planes.into_recon();
        return CoreOutput {
            stream: headers.write_still_picture(&slice_data),
            slice_data,
            rec_y,
            rec_cb,
            rec_cr,
            cw,
            ch,
            cwc,
            chc,
            qp,
            bit_depth,
            leaf_count: leaves.len(),
        };
    }

    // ---- Phase A: analyse + reconstruct every leaf (parallel CTU wavefront) ----
    let planes = Planes::new(cw, ch, cwc, chc);
    let ctu_decisions =
        analyze_ctus_wavefront(&shared, &planes, &ctu_leaves, ctu_cols, ctu_rows, threads);

    let mut enc = CabacEncoder::new();
    let mut ctx = Contexts::new_intra(slice_qp);
    let mut cursor = vec![0usize; ctu_cols * ctu_rows];
    // Adaptive-quant QP state in the SliceQpY (pre-QpBdOffset) domain, mirroring
    // the decoder. `qg_map` holds each resolved QG's QpY for neighbour
    // prediction; QGs are walked in decode order so a QG's left/above neighbours
    // are always finalised before it. The first coefficient-bearing TU of each QG
    // codes cu_qp_delta = target - predictor.
    let aq = headers.aq;
    let mut qg_map: Vec<i32> = vec![base_qpy; qg_cols * qg_rows];
    let mut cur_qg: usize = usize::MAX;
    let mut prev_qp: i32 = base_qpy;
    let mut cur_qg_qp_y: i32 = base_qpy;
    let mut qg_dqp_coded = false;
    let _leaves = code_partitions(
        &mut enc,
        &mut ctx,
        cw as u32,
        ch as u32,
        decide_fn,
        |enc, ctx, grid, x, y, size| {
            let (xu, yu, nu) = (x as usize, y as usize, size as usize);
            let cidx = (yu / ctu) * ctu_cols + (xu / ctu);
            let k = cursor[cidx];
            cursor[cidx] += 1;
            let d = &ctu_decisions[cidx][k];

            // Quantization-group entry: derive the QP predictor (same rule as the
            // decoder) and fill the map tentatively. Then, if this leaf is the
            // QG's first coefficient-bearing TU, emit cu_qp_delta and resolve.
            let mut code_dqp = None;
            if aq {
                let (qx, qy) = (xu & !(qg - 1), yu & !(qg - 1));
                let qidx = (qy / qg) * qg_cols + qx / qg;
                if qidx != cur_qg {
                    prev_qp = cur_qg_qp_y;
                    cur_qg_qp_y = aq_predict_qp(&qg_map, qg_cols, qg, ctu, qx, qy, prev_qp);
                    aq_fill_qg(&mut qg_map, qg_cols, qg, xu, yu, nu, cur_qg_qp_y);
                    qg_dqp_coded = false;
                    cur_qg = qidx;
                }
                if !qg_dqp_coded && leaf_has_cbf(d) {
                    let target = qg_target_qpy[qidx];
                    let delta = target - cur_qg_qp_y;
                    cur_qg_qp_y = if delta == 0 {
                        cur_qg_qp_y
                    } else {
                        ((cur_qg_qp_y + delta + 64 + 2 * qp_bd_offset) % (64 + qp_bd_offset))
                            - qp_bd_offset
                    };
                    aq_fill_qg(&mut qg_map, qg_cols, qg, xu, yu, nu, cur_qg_qp_y);
                    qg_dqp_coded = true;
                    code_dqp = Some(delta);
                }
            }
            replay_leaf(
                enc,
                ctx,
                grid,
                lossless,
                headers.lfnst,
                headers.dep_quant,
                headers.mts,
                headers.cclm,
                code_dqp,
                d,
            );
        },
    );
    enc.encode_terminate(1);

    let slice_data = enc.finish();
    let (mut rec_y, mut rec_cb, mut rec_cr) = planes.into_recon();
    if headers.deblock {
        let mut grid = crate::deblock::Grid::new(cw, ch);
        for leaves in &ctu_leaves {
            for &(x, y, w, h) in leaves {
                grid.set_cu(x, y, w, h, qp);
            }
        }
        crate::deblock::deblock_luma(&mut rec_y, cw, ch, &grid, bit_depth, ctu);
        if has_chroma {
            let subx = sub_w.trailing_zeros() as usize;
            let suby = sub_h.trailing_zeros() as usize;
            crate::deblock::deblock_chroma(
                &mut rec_cb,
                cwc,
                chc,
                &grid,
                subx,
                suby,
                bit_depth,
                ctu,
            );
            crate::deblock::deblock_chroma(
                &mut rec_cr,
                cwc,
                chc,
                &grid,
                subx,
                suby,
                bit_depth,
                ctu,
            );
        }
    }
    CoreOutput {
        stream: headers.write_still_picture(&slice_data),
        slice_data,
        rec_y,
        rec_cb,
        rec_cr,
        cw,
        ch,
        cwc,
        chc,
        qp,
        bit_depth,
        leaf_count: total_leaves,
    }
}

/// Encode a packed planar YCbCr image (the layout produced by the decoder: luma
/// `width×height`, then for non-monochrome Cb and Cr each `⌈width/sub_w⌉ ×
/// ⌈height/sub_h⌉`; samples are bytes at 8-bit and little-endian `u16` above).
/// Each plane is placed at the coded size with edge replication, then coded.
pub(crate) fn encode_still_yuv(
    planes: &[u8],
    width: u32,
    height: u32,
    qp: u8,
    bit_depth: u8,
    lossless: bool,
    chroma: ChromaFormat,
    threads: usize,
    rdoq: bool,
    aq: bool,
    mtt: bool,
    lfnst: bool,
    dep_quant: bool,
    mts: bool,
    dual_tree: bool,
    cclm: bool,
    deblock: bool,
) -> Result<Vec<u8>, crate::error::EncodeError> {
    // Dual tree falls back to single tree for lossless non-4:2:0; AQ's cu_qp_delta
    // is not implemented for the dual tree, so it is disabled there.
    let dual = dual_tree && !lossless;
    // LFNST is validated for 8-bit (single and dual tree, every chroma format and
    // picture size). In the dual tree luma and chroma carry separate LFNST
    // indices; both are signalled (see replay_luma / replay_chroma).
    let lfnst_eff = lfnst;
    let dep_quant_eff = dep_quant;
    let headers = Headers {
        width,
        height,
        chroma,
        bit_depth: crate::fmt::BitDepth::from_bits(bit_depth),
        qp,
        lossless,
        // AQ falls back to a flat QP for lossless: cu_qp_delta interacts with
        // the transquant-bypass QG handling in a way the reference decoder
        // does not accept (same rationale as the dual tree, where it is also
        // disabled).
        aq: aq && !lossless,
        // MTT (like the dual tree) falls back to the plain quadtree for
        // lossless: only the QT path forces CUs <= 32 so transform-skip
        // (which lossless requires) is always available.
        mtt: mtt && !lossless,
        lfnst: lfnst_eff,
        dep_quant: dep_quant_eff,
        mts,
        dual_tree: dual,
        cclm: cclm && !lossless,
        deblock: deblock && bit_depth == 8 && !aq && !dual && !lossless && !mtt,
    };
    let cw = headers.coded_width() as usize;
    let ch = headers.coded_height() as usize;
    let (w, h) = (width as usize, height as usize);
    let (sub_w, sub_h) = (chroma.sub_w(), chroma.sub_h());
    let has_chroma = !chroma.is_monochrome();
    let cwc = if has_chroma { cw / sub_w } else { 0 };
    let chc = if has_chroma { ch / sub_h } else { 0 };
    let dcw = if has_chroma { w.div_ceil(sub_w) } else { 0 };
    let dch = if has_chroma { h.div_ceil(sub_h) } else { 0 };
    let two_byte = bit_depth > 8;
    let bpp = if two_byte { 2 } else { 1 };
    let max_val = (1i32 << bit_depth) - 1;

    let expect = (w * h + 2 * dcw * dch) * bpp;
    if planes.len() != expect {
        return Err(crate::error::EncodeError::Unsupported(
            "YCbCr buffer length does not match the given dimensions, chroma format and bit depth",
        ));
    }

    let rd = |plane: &[u8], idx: usize| -> i32 {
        if two_byte {
            u16::from_le_bytes([plane[idx * 2], plane[idx * 2 + 1]]) as i32
        } else {
            plane[idx] as i32
        }
    };
    let y_plane = &planes[..w * h * bpp];
    let cb_plane = &planes[w * h * bpp..(w * h + dcw * dch) * bpp];
    let cr_plane = &planes[(w * h + dcw * dch) * bpp..];

    let mut src_y = vec![0i32; cw * ch];
    for y in 0..ch {
        let sy = y.min(h - 1);
        for x in 0..cw {
            let sx = x.min(w - 1);
            src_y[y * cw + x] = rd(y_plane, sy * w + sx).clamp(0, max_val);
        }
    }
    let mut src_cb = vec![0i32; cwc * chc];
    let mut src_cr = vec![0i32; cwc * chc];
    if has_chroma {
        for cy in 0..chc {
            let sy = cy.min(dch - 1);
            for cx in 0..cwc {
                let sx = cx.min(dcw - 1);
                src_cb[cy * cwc + cx] = rd(cb_plane, sy * dcw + sx).clamp(0, max_val);
                src_cr[cy * cwc + cx] = rd(cr_plane, sy * dcw + sx).clamp(0, max_val);
            }
        }
    }
    Ok(encode_planes(headers, src_y, src_cb, src_cr, threads, rdoq).stream)
}

/// Encode packed 8-bit RGB(A) as a 4:2:0 VVC still picture. `stride_px` is the
/// number of channels per pixel (3 for RGB, 4 for RGBA).
pub(crate) fn encode_still(
    rgb: &[u8],
    width: u32,
    height: u32,
    qp: u8,
    bit_depth: u8,
    stride_px: usize,
    lossless: bool,
    chroma: ChromaFormat,
    threads: usize,
    rdoq: bool,
    aq: bool,
    mtt: bool,
    lfnst: bool,
    dep_quant: bool,
    mts: bool,
    dual_tree: bool,
    cclm: bool,
    deblock: bool,
) -> Vec<u8> {
    // 8-bit samples scaled up to the coded bit depth (the "upscale" path).
    let scale_shift = (bit_depth - 8) as u32;
    encode_core(
        rgb,
        width,
        height,
        qp,
        bit_depth,
        stride_px,
        lossless,
        chroma,
        scale_shift,
        threads,
        rdoq,
        aq,
        mtt,
        lfnst,
        dep_quant,
        mts,
        dual_tree,
        cclm,
        deblock,
    )
    .stream
}

/// Encode native RGB(A) samples already at the coded bit depth (no up-scaling),
/// e.g. 10-bit `u16` input in `0..=1023`. `stride_px` is the channel count.
pub(crate) fn encode_still_wide(
    rgb: &[u16],
    width: u32,
    height: u32,
    qp: u8,
    bit_depth: u8,
    stride_px: usize,
    lossless: bool,
    chroma: ChromaFormat,
    threads: usize,
    rdoq: bool,
    aq: bool,
    mtt: bool,
    lfnst: bool,
    dep_quant: bool,
    mts: bool,
    dual_tree: bool,
    cclm: bool,
    deblock: bool,
) -> Vec<u8> {
    encode_core(
        rgb, width, height, qp, bit_depth, stride_px, lossless, chroma, 0, threads, rdoq, aq, mtt,
        lfnst, dep_quant, mts, dual_tree, cclm, deblock,
    )
    .stream
}

pub(crate) fn encode_with_recon(
    rgb: &[u8],
    width: u32,
    height: u32,
    qp: u8,
    bit_depth: u8,
    stride_px: usize,
    lossless: bool,
    chroma: ChromaFormat,
    rdoq: bool,
    aq: bool,
    mtt: bool,
    lfnst: bool,
    dep_quant: bool,
    mts: bool,
    dual_tree: bool,
    cclm: bool,
    deblock: bool,
) -> (Vec<u8>, Vec<u8>) {
    let scale_shift = (bit_depth - 8) as u32;
    let o = encode_core(
        rgb,
        width,
        height,
        qp,
        bit_depth,
        stride_px,
        lossless,
        chroma,
        scale_shift,
        1,
        rdoq,
        aq,
        mtt,
        lfnst,
        dep_quant,
        mts,
        dual_tree,
        cclm,
        deblock,
    );
    let (w, h) = (width as usize, height as usize);
    let has_chroma = !chroma.is_monochrome();
    let (dcw, dch) = if has_chroma {
        (w.div_ceil(chroma.sub_w()), h.div_ceil(chroma.sub_h()))
    } else {
        (0, 0)
    };
    let two_byte = bit_depth > 8;
    let bytes_per = if two_byte { 2 } else { 1 };
    let mut yuv = Vec::with_capacity((w * h + 2 * dcw * dch) * bytes_per);
    let push = |v: i32, out: &mut Vec<u8>| {
        if two_byte {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        } else {
            out.push(v as u8);
        }
    };
    for y in 0..h {
        for x in 0..w {
            push(o.rec_y[y * o.cw + x], &mut yuv);
        }
    }
    if has_chroma {
        for y in 0..dch {
            for x in 0..dcw {
                push(o.rec_cb[y * o.cwc + x], &mut yuv);
            }
        }
        for y in 0..dch {
            for x in 0..dcw {
                push(o.rec_cr[y * o.cwc + x], &mut yuv);
            }
        }
    }
    (o.stream, yuv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::engine::CabacDecoder;
    use crate::intra::{build_mpm, decode_chroma_mode, decode_luma_mode};
    use crate::partition::test_support::decode_partitions;
    use crate::tu::test_support::decode_transform_unit;

    /// Pack a `CoreOutput`'s reconstruction into the planar byte layout the
    /// public decoder returns (Y, then Cb, Cr for non-monochrome), at the coded
    /// dimensions. `two_byte` selects 8-bit vs little-endian 10/12-bit samples.
    fn recon_planar(o: &CoreOutput, has_chroma: bool, two_byte: bool) -> Vec<u8> {
        let push = |v: i32, out: &mut Vec<u8>| {
            if two_byte {
                out.extend_from_slice(&(v as u16).to_le_bytes());
            } else {
                out.push(v as u8);
            }
        };
        let mut yuv = Vec::new();
        for &v in &o.rec_y[..o.cw * o.ch] {
            push(v, &mut yuv);
        }
        if has_chroma {
            for &v in &o.rec_cb[..o.cwc * o.chc] {
                push(v, &mut yuv);
            }
            for &v in &o.rec_cr[..o.cwc * o.chc] {
                push(v, &mut yuv);
            }
        }
        yuv
    }

    #[test]
    fn aq_encoder_decoder_agree_8_and_10_bit() {
        let (w, h) = (256usize, 256usize);
        let mut rgb8 = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                let noisy = (x / 40 + y / 40) % 2 == 1;
                let v = if noisy {
                    (((x * 131) ^ (y * 197)) & 0xff) as u8
                } else {
                    100
                };
                rgb8[o] = v;
                rgb8[o + 1] = v;
                rgb8[o + 2] = v;
            }
        }
        let rgb10: Vec<u16> = rgb8.iter().map(|&b| (b as u16) << 2).collect();
        for &qp in &[9u8, 18, 33] {
            for chroma in [
                ChromaFormat::Yuv420,
                ChromaFormat::Yuv444,
                ChromaFormat::Monochrome,
            ] {
                let has_chroma = !chroma.is_monochrome();
                let o8 = encode_core(
                    &rgb8, w as u32, h as u32, qp, 8, 3, false, chroma, 0, 1, true, true, false,
                    false, false, false, false, false, false,
                );
                let img8 = crate::decode_266(&o8.stream).expect("8-bit AQ decode");
                assert_eq!(
                    img8.planes,
                    recon_planar(&o8, has_chroma, false),
                    "8-bit AQ enc/dec mismatch qp{qp} {chroma:?}"
                );
                let o10 = encode_core(
                    &rgb10, w as u32, h as u32, qp, 10, 3, false, chroma, 0, 1, true, true, false,
                    false, false, false, false, false, false,
                );
                let img10 = crate::decode_266(&o10.stream).expect("10-bit AQ decode");
                assert_eq!(
                    img10.planes,
                    recon_planar(&o10, has_chroma, true),
                    "10-bit AQ enc/dec mismatch qp{qp} {chroma:?}"
                );
            }
        }
    }

    #[test]
    fn aq_changes_stream_and_is_deterministic() {
        // On content with varying activity AQ must change the bitstream (it is
        // doing something), stay decodable, and be byte-deterministic.
        let (w, h) = (256u32, 256u32);
        let mut rgb = vec![0u8; (w * h) as usize * 3];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let o = (y * w as usize + x) * 3;
                let noisy = (x / 128 + y / 128) % 2 == 1;
                let v = if noisy {
                    (((x * 131) ^ (y * 197)) & 0xff) as u8
                } else {
                    100
                };
                rgb[o] = v;
                rgb[o + 1] = v;
                rgb[o + 2] = v;
            }
        }
        for q in [40u8, 70, 90] {
            let base = crate::EncodeConfig::new()
                .with_quality(q)
                .with_chroma(ChromaFormat::Yuv420);
            let off = crate::encode_rgb_266(&rgb, w, h, &base.clone().with_aq(false)).unwrap();
            let on = crate::encode_rgb_266(&rgb, w, h, &base.clone().with_aq(true)).unwrap();
            assert_ne!(off, on, "AQ should change the stream at q{q}");
            crate::decode_266(&on).expect("AQ stream must decode");
            let on2 = crate::encode_rgb_266(&rgb, w, h, &base.with_aq(true)).unwrap();
            assert_eq!(on, on2, "AQ must be deterministic at q{q}");
        }
    }

    /// Strip emulation-prevention bytes (EBSP -> RBSP): remove a 0x03 that
    /// follows a 0x00 0x00 pair preceding a byte <= 0x03.
    fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ebsp.len());
        let mut zeros = 0;
        let mut i = 0;
        while i < ebsp.len() {
            let b = ebsp[i];
            if zeros >= 2 && b == 0x03 && i + 1 < ebsp.len() && ebsp[i + 1] <= 0x03 {
                zeros = 0; // drop the emulation_prevention_three_byte
            } else {
                out.push(b);
                zeros = if b == 0 { zeros + 1 } else { 0 };
            }
            i += 1;
        }
        out
    }

    /// Split an Annex-B byte stream into NAL units (payloads after start codes).
    fn split_nals(stream: &[u8]) -> Vec<Vec<u8>> {
        let mut nals = Vec::new();
        let mut i = 0;
        let mut start: Option<usize> = None;
        while i + 3 <= stream.len() {
            let sc3 = stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1;
            if sc3 {
                if let Some(s) = start {
                    let mut end = i;
                    if end > s && stream[end - 1] == 0 {
                        end -= 1; // trailing zero belongs to next start code
                    }
                    nals.push(stream[s..end].to_vec());
                }
                i += 3;
                start = Some(i);
            } else {
                i += 1;
            }
        }
        if let Some(s) = start {
            nals.push(stream[s..].to_vec());
        }
        nals
    }

    /// Clean-room decoder: reconstruct Y/Cb/Cr planes from CABAC slice data,
    /// mirroring the encoder's per-leaf prediction and reconstruction.
    fn decode_slice(o: &CoreOutput) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let (cw, ch, cwc, chc) = (o.cw, o.ch, o.cwc, o.chc);
        let (qp, bd) = (o.qp, o.bit_depth);
        let max_val = (1i32 << bd) - 1;
        let mut dec = CabacDecoder::new(&o.slice_data);
        let mut ctx = Contexts::new_intra(qp);
        let mut ry = vec![0i32; cw * ch];
        let mut ay = vec![false; cw * ch];
        let mut rcb = vec![0i32; cwc * chc];
        let mut acb = vec![false; cwc * chc];
        let mut rcr = vec![0i32; cwc * chc];
        let mut acr = vec![false; cwc * chc];

        decode_partitions(
            &mut dec,
            &mut ctx,
            cw as u32,
            ch as u32,
            |dec, ctx, grid, x, y, size| {
                let (x, y, n) = (x as usize, y as usize, size as usize);
                let mpm = build_mpm(
                    grid.left_mode(x as u32, y as u32, n as u32),
                    grid.above_mode(x as u32, y as u32, n as u32),
                );
                let mode = decode_luma_mode(dec, ctx, &mpm);
                grid.set_mode(x as u32, y as u32, n as u32, mode);
                let chroma_mode = decode_chroma_mode(dec, ctx, mode, false);
                let tu = decode_transform_unit(dec, ctx, n, n, Some((n / 2, n / 2)));

                let refs_y = gather_refs(&ry, &ay, cw, ch, x, y, n, bd);
                let pred_y = crate::predict::predict(mode, n, n, &refs_y, bd, true);
                reconstruct(
                    &mut ry, &mut ay, cw, x, y, n, &pred_y, &tu.luma, qp, bd, max_val, false,
                    tu.luma_ts, 0, 0, false, 0,
                );

                let (cx, cy, cn) = (x / 2, y / 2, n / 2);
                let refs_cb = gather_refs(&rcb, &acb, cwc, chc, cx, cy, cn, bd);
                let pred_cb = crate::predict::predict(chroma_mode, cn, cn, &refs_cb, bd, false);
                reconstruct(
                    &mut rcb, &mut acb, cwc, cx, cy, cn, &pred_cb, &tu.cb, qp, bd, max_val, false,
                    tu.cb_ts, 0, 0, false, 0,
                );
                let refs_cr = gather_refs(&rcr, &acr, cwc, chc, cx, cy, cn, bd);
                let pred_cr = crate::predict::predict(chroma_mode, cn, cn, &refs_cr, bd, false);
                reconstruct(
                    &mut rcr, &mut acr, cwc, cx, cy, cn, &pred_cr, &tu.cr, qp, bd, max_val, false,
                    tu.cr_ts, 0, 0, false, 0,
                );
            },
        );
        assert_eq!(
            dec.decode_terminate(),
            1,
            "slice data did not terminate cleanly"
        );
        (ry, rcb, rcr)
    }

    #[test]
    fn yuv444_64wide_chroma_roundtrips() {
        // A 64-wide 4:4:4 chroma block (flat region under CTU-128) exercises the
        // 64-point chroma transform + high-frequency zero-out. Regression for an
        // encoder empty-TU panic (invalid transform-skip trial on 64-wide chroma).
        // Verified via the real public decoder, which handles every chroma format
        // (the `decode_slice` test helper is 4:2:0-only).
        let (w, h) = (160usize, 160usize);
        let mut rgb = vec![0u8; w * h * 3];
        let mut s = 0xABCDEF123456u64;
        for y in 0..h {
            for x in 0..w {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let n = (s & 7) as i32;
                let i = (y * w + x) * 3;
                rgb[i] = ((x * 255 / w) as i32 + n).clamp(0, 255) as u8;
                rgb[i + 1] = ((y * 255 / h) as i32 + n).clamp(0, 255) as u8;
                rgb[i + 2] = (128 + n) as u8;
            }
        }
        for &q in &[85u8, 70, 50] {
            let cfg = crate::EncodeConfig::new()
                .with_quality(q)
                .with_chroma(crate::ChromaFormat::Yuv444);
            let (stream, recon) =
                crate::encode_rgb_with_reconstruction(&rgb, w as u32, h as u32, &cfg).unwrap();
            let img = crate::decode_266(&stream).expect("decode failed");
            assert_eq!(
                img.planes, recon,
                "4:4:4 64-wide chroma round-trip mismatch q={q}"
            );
        }
    }

    #[test]
    fn dual_tree_decodes_to_encoder_reconstruction() {
        for &chroma in &[
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            for &(w, h) in &[(64usize, 64usize), (128, 128), (96, 80), (192, 136)] {
                let img = gradient_rgb(w, h);
                for &q in &[80u8, 55] {
                    let cfg = crate::EncodeConfig::new()
                        .with_quality(q)
                        .with_chroma(chroma)
                        .with_dual_tree(true);
                    let (stream, recon) =
                        crate::encode_rgb_with_reconstruction(&img, w as u32, h as u32, &cfg)
                            .unwrap();
                    let dec = crate::decode_266(&stream).expect("dual-tree decode");
                    assert_eq!(
                        dec.planes, recon,
                        "dual round-trip mismatch {chroma:?} {w}x{h} q={q}"
                    );
                }
            }
        }
    }

    #[test]
    fn cclm_decodes_to_encoder_reconstruction() {
        for &chroma in &[
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            for &(w, h) in &[(64usize, 64usize), (128, 128), (192, 136)] {
                let img = gradient_rgb(w, h);
                for &q in &[80u8, 55] {
                    let cfg = crate::EncodeConfig::new()
                        .with_quality(q)
                        .with_chroma(chroma)
                        .with_dual_tree(true)
                        .with_cclm(true);
                    let (stream, recon) =
                        crate::encode_rgb_with_reconstruction(&img, w as u32, h as u32, &cfg)
                            .unwrap();
                    let dec = crate::decode_266(&stream).expect("cclm decode");
                    assert_eq!(
                        dec.planes, recon,
                        "cclm round-trip mismatch {chroma:?} {w}x{h} q={q}"
                    );
                }
            }
        }
    }

    #[test]
    fn slice_data_decodes_to_encoder_reconstruction() {
        // Decoding the emitted CABAC slice data must reproduce, bit-exactly, the
        // encoder's own reconstruction of every Y/Cb/Cr sample.
        let (w, h) = (48usize, 40usize);
        let img = gradient_rgb(w, h);
        for &qp in &[10u8, 27, 45] {
            let o = encode_core(
                &img,
                w as u32,
                h as u32,
                qp,
                8,
                3,
                false,
                ChromaFormat::Yuv420,
                0,
                1,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
            );
            let (ry, rcb, rcr) = decode_slice(&o);
            assert_eq!(ry, o.rec_y, "luma mismatch qp={qp}");
            assert_eq!(rcb, o.rec_cb, "Cb mismatch qp={qp}");
            assert_eq!(rcr, o.rec_cr, "Cr mismatch qp={qp}");
        }
    }

    #[test]
    fn full_annexb_stream_decodes_to_reconstruction() {
        // Parse the real Annex-B output (NAL framing + header + EBSP), recover
        // the slice data, and decode it to the encoder's reconstruction.
        let (w, h) = (40usize, 24usize);
        let img = gradient_rgb(w, h);
        let o = encode_core(
            &img,
            w as u32,
            h as u32,
            26,
            8,
            3,
            false,
            ChromaFormat::Yuv420,
            0,
            1,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        );
        let nals = split_nals(&o.stream);
        assert_eq!(nals.len(), 3, "expected SPS, PPS, IDR");
        let idr = &nals[2];
        // Strip the 2-byte NAL header, undo emulation prevention.
        let rbsp = ebsp_to_rbsp(&idr[2..]);
        // The slice header occupies exactly two bytes in this configuration.
        let slice_data = &rbsp[2..];
        assert_eq!(
            slice_data,
            &o.slice_data[..],
            "recovered slice data mismatch"
        );
        // And it decodes to the same reconstruction.
        let mut o2 = encode_core(
            &img,
            w as u32,
            h as u32,
            26,
            8,
            3,
            false,
            ChromaFormat::Yuv420,
            0,
            1,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        );
        o2.slice_data = slice_data.to_vec();
        let (ry, _, _) = decode_slice(&o2);
        assert_eq!(ry, o.rec_y);
    }

    #[test]
    fn lfnst_round_trips_and_engages() {
        // LFNST end-to-end: smooth low-frequency directional content (its sweet
        // spot) so the encoder actually selects the secondary transform. The
        // clean-room decoder must reproduce the encoder reconstruction
        // bit-exactly, and enabling LFNST must change the stream (proving the
        // index is signalled, not silently dropped). Conformance against VTM is
        // covered by examples/lfnst_probe.rs.
        use crate::fmt::BitDepth;
        let (w, h) = (128usize, 128usize);
        let mut src_y = vec![0i32; w * h];
        for j in 0..h {
            for i in 0..w {
                let (fi, fj) = (i as f32, j as f32);
                let a = 46.0 * (0.13 * fi + 0.05 * fj).sin();
                let b = 30.0 * (0.05 * fi - 0.16 * fj).sin();
                src_y[j * w + i] = (128.0 + a + b).clamp(0.0, 255.0) as i32;
            }
        }
        let hdr = |lfnst: bool| Headers {
            width: w as u32,
            height: h as u32,
            chroma: ChromaFormat::Monochrome,
            bit_depth: BitDepth::Eight,
            qp: 35,
            lossless: false,
            aq: false,
            mtt: false,
            lfnst,
            dep_quant: false,
            mts: false,
            dual_tree: false,
            cclm: false,
            deblock: false,
        };
        let off = encode_planes(hdr(false), src_y.clone(), Vec::new(), Vec::new(), 1, true);
        let on = encode_planes(hdr(true), src_y.clone(), Vec::new(), Vec::new(), 1, true);
        // The decoder reproduces the encoder reconstruction bit-exactly.
        let img = crate::decode_266(&on.stream).expect("LFNST stream must decode");
        let enc_planes: Vec<u8> = on.rec_y.iter().map(|&v| v as u8).collect();
        assert_eq!(
            img.planes, enc_planes,
            "LFNST decoder disagrees with encoder"
        );
        // Enabling LFNST changes the bitstream (the index is actually used).
        assert_ne!(on.stream, off.stream, "LFNST did not affect the stream");
    }

    fn recon_mse(rgb: &[u8], w: usize, h: usize, qp: u8) -> f64 {
        let o = encode_core(
            rgb,
            w as u32,
            h as u32,
            qp,
            8,
            3,
            false,
            ChromaFormat::Yuv420,
            0,
            1,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        );
        let (recon, cw) = (o.rec_y, o.cw);
        // Compare against the encoder's own luma source for the visible region.
        let mut sse = 0i64;
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                let (r, g, b) = (rgb[o] as i32, rgb[o + 1] as i32, rgb[o + 2] as i32);
                let yv = rgb_to_y_q13(r, g, b);
                let d = (yv - recon[y * cw + x]) as i64;
                sse += d * d;
            }
        }
        sse as f64 / (w * h) as f64
    }

    #[test]
    fn mtt_stream_round_trips_through_decoder() {
        // End-to-end MTT validation. `mtt_decide` mixes every split type by
        // position, so a single picture exercises the whole rectangular path;
        // mult-of-8 non-CTU-multiple sizes add implicit boundary splits. Validated
        // across chroma formats. garnetash's decoder must reproduce the encoder
        // reconstruction bit-exactly; streams + recon are dumped for VTM too.
        use crate::fmt::BitDepth;
        let build_plane = |n: usize, salt: u64, kind: u8| -> Vec<i32> {
            let mut st = 0x9E3779B97F4A7C15u64 ^ salt;
            let mut out = vec![0i32; n];
            for (i, v) in out.iter_mut().enumerate() {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let noise = (st >> 40) as i32 & 15;
                *v = match kind {
                    0 => ((i % 251) as i32 + noise).clamp(0, 255),
                    _ => (96 + (i % 64) as i32 / 2 + noise / 4).clamp(0, 255),
                };
            }
            out
        };
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv444,
            ChromaFormat::Yuv422,
            ChromaFormat::Monochrome,
        ] {
            let (sw, sh) = (chroma.sub_w(), chroma.sub_h());
            let has_chroma = !chroma.is_monochrome();
            for &(w, h) in &[(128usize, 128usize), (192, 128), (200, 136)] {
                let (cwc, chc) = if has_chroma { (w / sw, h / sh) } else { (0, 0) };
                let src_y = build_plane(w * h, 1, 0);
                let src_cb = if has_chroma {
                    build_plane(cwc * chc, 2, 1)
                } else {
                    Vec::new()
                };
                let src_cr = if has_chroma {
                    build_plane(cwc * chc, 3, 1)
                } else {
                    Vec::new()
                };
                let qp = 30u8;
                let headers = Headers {
                    width: w as u32,
                    height: h as u32,
                    chroma,
                    bit_depth: BitDepth::Eight,
                    qp,
                    lossless: false,
                    aq: false,
                    mtt: true,
                    lfnst: false,
                    dep_quant: false,
                    mts: false,
                    dual_tree: false,
                    cclm: false,
                    deblock: false,
                };
                let o = encode_planes(
                    headers,
                    src_y.clone(),
                    src_cb.clone(),
                    src_cr.clone(),
                    1,
                    true,
                );
                assert!(
                    o.leaf_count > (w / 32) * (h / 32),
                    "too few MTT leaves {chroma:?} {w}x{h}"
                );
                let img = crate::decode_266(&o.stream).expect("MTT stream must decode");
                let mut enc_planes: Vec<u8> = o.rec_y.iter().map(|&v| v as u8).collect();
                if has_chroma {
                    enc_planes.extend(o.rec_cb.iter().map(|&v| v as u8));
                    enc_planes.extend(o.rec_cr.iter().map(|&v| v as u8));
                }
                assert_eq!(
                    img.planes, enc_planes,
                    "MTT decoder disagrees {chroma:?} {w}x{h}"
                );
                let tag = format!("{:?}_{w}x{h}", chroma);
                let _ = std::fs::write(format!("/tmp/mtt_{tag}.266"), &o.stream);
                let _ = std::fs::write(format!("/tmp/mtt_{tag}.enc.yuv"), &enc_planes);
            }
        }
    }

    #[test]
    fn mtt_vs_qt_rate_distortion() {
        // Directional content (vertical stripes left, horizontal right) plus a
        // flat quadrant, so both MTT (directional splits) and AQ (activity-varying
        // QP) have something to exploit. Reports bytes + luma PSNR for all four
        // QT/AQ/MTT/AQ+MTT combinations so their RD behaviour is visible. Every
        // config must stay conformant (decodes).
        let (w, h) = (256usize, 256usize);
        let mut src_y = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = if y < h / 2 && x < w / 2 {
                    120
                } else if x >= w / 2 && y < h / 2 {
                    if (x / 5) % 2 == 0 { 60 } else { 200 }
                } else if (y / 5) % 2 == 0 {
                    60
                } else {
                    200
                };
                src_y[y * w + x] = v;
            }
        }
        let (cwc, chc) = (w / 2, h / 2);
        let src_cb = vec![128i32; cwc * chc];
        let src_cr = vec![128i32; cwc * chc];
        let psnr = |rec: &[i32]| -> f64 {
            let sse: i64 = rec
                .iter()
                .zip(&src_y)
                .map(|(a, b)| {
                    let d = (a - b) as i64;
                    d * d
                })
                .sum();
            if sse == 0 {
                return 99.0;
            }
            10.0 * (255.0f64 * 255.0 / (sse as f64 / (w * h) as f64)).log10()
        };
        let run = |aq: bool, mtt: bool| {
            let headers = Headers {
                width: w as u32,
                height: h as u32,
                chroma: ChromaFormat::Yuv420,
                bit_depth: crate::fmt::BitDepth::Eight,
                qp: 28,
                lossless: false,
                aq,
                mtt,
                lfnst: false,
                dep_quant: false,
                mts: false,
                dual_tree: false,
                cclm: false,
                deblock: false,
            };
            encode_planes(
                headers,
                src_y.clone(),
                src_cb.clone(),
                src_cr.clone(),
                1,
                true,
            )
        };
        for (name, aq, mtt) in [
            ("QT", false, false),
            ("AQ", true, false),
            ("MTT", false, true),
            ("AQ+MTT", true, true),
        ] {
            let o = run(aq, mtt);
            crate::decode_266(&o.stream).expect("must decode");
            eprintln!(
                "{name:8}: {:5} B   {:.2} dB",
                o.slice_data.len(),
                psnr(&o.rec_y)
            );
        }
        // Conformance + sanity: MTT path must not bloat pathologically vs QT.
        let qt = run(false, false);
        let mt = run(false, true);
        assert!(
            mt.slice_data.len() < qt.slice_data.len() * 2,
            "MTT bloated vs QT"
        );
    }

    fn gradient_rgb(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                v[o] = ((x * 5 + y * 7) & 0xff) as u8;
                v[o + 1] = ((x * 3) & 0xff) as u8;
                v[o + 2] = ((y * 4) & 0xff) as u8;
            }
        }
        v
    }

    #[test]
    fn rdoq_toggle_changes_levels_and_stays_decodable() {
        let (w, h) = (64usize, 64usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                let v = ((x * 37 + y * 53) ^ (x.wrapping_mul(y) >> 1)) as u8;
                rgb[o] = v;
                rgb[o + 1] = v.wrapping_add(40);
                rgb[o + 2] = v.wrapping_mul(3);
            }
        }
        let enc = |rdoq: bool, q: u8, lossless: bool| {
            // Isolate RDOQ: AQ varies QP per block, MTT changes the partition, and
            // the dual tree splits luma/chroma — any of which lets an RD-optimal
            // pass spend extra rate to cut distortion. The "RDOQ never grows the
            // stream" invariant is only meaningful for RDOQ on its own.
            let cfg = crate::EncodeConfig::new()
                .with_quality(q)
                .with_rdoq(rdoq)
                .with_lossless(lossless)
                .with_aq(false)
                .with_mtt(false)
                .with_dual_tree(false);
            crate::encode_rgb(&rgb, w as u32, h as u32, &cfg).unwrap()
        };
        let off = enc(false, 60, false);
        let on = enc(true, 60, false);
        // Both decode.
        assert!(crate::decode(&off).is_ok());
        assert!(crate::decode(&on).is_ok());
        // RDOQ never makes the file larger, and changes the coded levels for at
        // least one quality point (where trailing coefficients exist to trim).
        let mut differed = false;
        for q in [40u8, 55, 75] {
            let o = enc(false, q, false);
            let n = enc(true, q, false);
            assert!(crate::decode(&n).is_ok());
            // RDOQ minimises an RD cost (rate + lambda*distortion), not pure
            // rate, so on rare blocks it may spend a few extra bytes to reduce
            // distortion. Allow a small slack rather than requiring it never grow.
            assert!(
                n.len() <= o.len() + 8,
                "RDOQ grew the stream by >8 bytes at q{q}: {} > {}",
                n.len(),
                o.len()
            );
            differed |= o != n;
        }
        assert!(
            differed,
            "RDOQ altered no quality point on high-frequency content"
        );
        // Lossless ignores the RDOQ flag (separate residual path).
        assert_eq!(enc(false, 60, true), enc(true, 60, true));
    }

    #[test]
    fn luma_reconstruction_faithful_at_high_quality() {
        let (w, h) = (48usize, 32usize);
        let mse = recon_mse(&gradient_rgb(w, h), w, h, 8);
        assert!(mse < 8.0, "high-quality luma MSE too large: {mse}");
    }

    #[test]
    fn quality_degrades_with_qp() {
        let (w, h) = (48usize, 32usize);
        let img = gradient_rgb(w, h);
        assert!(recon_mse(&img, w, h, 42) >= recon_mse(&img, w, h, 8));
    }

    #[test]
    fn flat_image_reconstructs_exactly() {
        let (w, h) = (40usize, 40usize);
        let mut rgb = vec![0u8; w * h * 3];
        for p in rgb.chunks_mut(3) {
            p[0] = 90;
            p[1] = 130;
            p[2] = 60;
        }
        assert_eq!(recon_mse(&rgb, w, h, 32), 0.0);
    }

    #[test]
    fn produces_420_annexb_stream() {
        let (w, h) = (40usize, 24usize);
        let bytes = encode_still(
            &gradient_rgb(w, h),
            w as u32,
            h as u32,
            30,
            8,
            3,
            false,
            ChromaFormat::Yuv420,
            1,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        );
        assert_eq!(&bytes[..4], &[0, 0, 0, 1]);
        let starts = bytes.windows(3).filter(|wnd| wnd == &[0, 0, 1]).count();
        assert!(starts >= 3, "expected >=3 NAL units, found {starts}");
    }
    #[test]
    fn high_detail_splits_more_than_flat() {
        let (w, h) = (64usize, 64usize);
        // Flat image: no voluntary splits -> few, large CUs.
        let mut flat = vec![0u8; w * h * 3];
        for p in flat.chunks_mut(3) {
            p[0] = 100;
            p[1] = 100;
            p[2] = 100;
        }
        let flat_leaves = encode_core(
            &flat,
            w as u32,
            h as u32,
            20,
            8,
            3,
            false,
            ChromaFormat::Yuv420,
            0,
            1,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
        .leaf_count;
        // High-frequency image: many splits.
        let mut noise = vec![0u8; w * h * 3];
        for (i, p) in noise.chunks_mut(3).enumerate() {
            let v = (((i * 97 + (i / w) * 53) % 256) as u8) ^ ((i & 1) as u8 * 200);
            p[0] = v;
            p[1] = v.wrapping_mul(3);
            p[2] = !v;
        }
        let noise_leaves = encode_core(
            &noise,
            w as u32,
            h as u32,
            20,
            8,
            3,
            false,
            ChromaFormat::Yuv420,
            0,
            1,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
        .leaf_count;
        assert!(
            noise_leaves > flat_leaves,
            "detail should split more: flat={flat_leaves} noise={noise_leaves}"
        );
        // Flat 64x64 is a single CTU under CTU-64: one undivided 64×64 leaf.
        assert_eq!(flat_leaves, 1, "flat image should not split");
    }
}
