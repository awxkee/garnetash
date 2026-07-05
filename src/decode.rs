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

//! The garnetash decoder: turn a raw Annex-B VVC intra still stream (SPS + PPS +
//! one IDR slice, as garnetash produces) back into reconstructed YCbCr samples.
//!
//! It parses the parameter sets and slice header, then walks the CTU quadtree
//! through the same per-block machinery the encoder's reconstruction loop uses
//! (`decode_partitions`, `decode_luma_mode` / `decode_chroma_mode`,
//! `decode_transform_unit`, `predict`, `reconstruct_wh`) so that the decoded
//! picture is bit-identical to the encoder's own reconstruction — and therefore
//! to any conformant VVC decoder.
//!
//! Scope: this matches garnetash's own (fixed) stream layout — single-layer,
//! single-slice, intra IDR, quadtree-only, 8-bit, lossy. Lossless (BDPCM)
//! streams are detected and rejected for now.

use crate::bitstream::ebsp_to_rbsp;
use crate::cabac::Contexts;
use crate::cabac::engine::CabacDecoder;
use crate::color::ColorMetadata;
use crate::encode::{gather_refs_wh, reconstruct_wh};
use crate::error::EncodeError;
use crate::fmt::{BitDepth, ChromaFormat};
use crate::headers::{parse_pps_qp, parse_sps, slice_data_offset};
use crate::intra::{
    HOR_IDX, VER_IDX, build_mpm, chroma_422_mode, decode_bdpcm_mode, decode_chroma_mode,
    decode_luma_mode,
};
use crate::metadata::Orientation;

/// Read `residual_lfnst_mode` (mirror of the encoder's `encode_lfnst_idx`).
fn decode_lfnst_idx(
    dec: &mut crate::cabac::engine::CabacDecoder,
    ctx: &mut Contexts,
    sep_tree: bool,
) -> u8 {
    let c0 = if sep_tree { 1 } else { 0 };
    if dec.decode_bin(&mut ctx.lfnst_idx[c0]) == 0 {
        0
    } else if dec.decode_bin(&mut ctx.lfnst_idx[2]) == 0 {
        1
    } else {
        2
    }
}

/// Read `mts_idx` (mirror of `encode_mts_idx`): a context-coded first bin, then
/// a truncated-unary tail over three contexts giving idx 1..=4.
fn decode_mts_idx(dec: &mut crate::cabac::engine::CabacDecoder, ctx: &mut Contexts) -> u8 {
    if dec.decode_bin(&mut ctx.mts_idx[0]) == 0 {
        return 0;
    }
    let mut idx = 1u8;
    for i in 0..3usize {
        if dec.decode_bin(&mut ctx.mts_idx[1 + i]) == 0 {
            break;
        }
        idx += 1;
    }
    idx
}
use crate::partition::test_support::decode_partitions;
use crate::predict::{RefSamples, predict_into};
use crate::tu::test_support::decode_transform_unit_full;

/// Shared reconstruction state for the dual-tree decode path. The two leaf
/// closures (luma-only, chroma-only) borrow this through a `RefCell`; `modes`
/// is the decoded luma-mode grid (4-sample cells) the chroma pass reads for its
/// DM (co-located luma mode at the chroma block centre), mirroring the encoder.
struct DualRec {
    rec_y: Vec<i32>,
    av_y: Vec<bool>,
    rec_cb: Vec<i32>,
    av_cb: Vec<bool>,
    rec_cr: Vec<i32>,
    av_cr: Vec<bool>,
    pred: Vec<i32>,
    pred_scratch: Vec<i32>,
    modes: Vec<u8>,
    mode_cx: usize,
}
impl DualRec {
    fn set_mode(&mut self, x: usize, y: usize, w: usize, h: usize, m: u8) {
        let (mut cy, ey) = (y / 4, (y + h).div_ceil(4));
        while cy < ey {
            let (mut cx, ex) = (x / 4, (x + w).div_ceil(4));
            while cx < ex {
                let i = cy * self.mode_cx + cx;
                if i < self.modes.len() {
                    self.modes[i] = m;
                }
                cx += 1;
            }
            cy += 1;
        }
    }
    fn dm(&self, x: usize, y: usize, w: usize, h: usize) -> u8 {
        let cx = (x + (w >> 1)) / 4;
        let cy = (y + (h >> 1)) / 4;
        self.modes[(cy * self.mode_cx + cx).min(self.modes.len() - 1)]
    }
}

/// A decoded still image: cropped 8-bit planar samples plus the geometry.
#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub chroma: ChromaFormat,
    pub bit_depth: BitDepth,
    /// Cropped planar samples: the full luma plane (`width * height`), then —
    /// unless monochrome — the Cb and Cr planes at chroma resolution
    /// (`ceil(width / SubWidthC) * ceil(height / SubHeightC)` each). This is the
    /// exact layout the encoder's reconstruction uses (I420 / I422 / I444 / Y).
    pub planes: Vec<u8>,
    /// Display orientation recovered from the HEIF `irot`/`imir` properties
    /// (`Orientation::Normal` for raw streams or files without them).
    pub orientation: Orientation,
    /// color metadata recovered from the HEIF `colr` properties: the CICP
    /// description (`nclx`) and/or an embedded ICC profile (`prof`). Empty for
    /// raw streams.
    pub color: ColorMetadata,
    /// The decoded alpha image, when the HEIF carried an alpha auxiliary image
    /// (a monochrome `DecodedImage` whose luma plane is the alpha channel).
    /// `None` for raw streams and for files without alpha.
    pub alpha: Option<Box<DecodedImage>>,
}

/// A read-only view of one plane of a [`DecodedImage`]: a tightly packed,
/// row-major slice of the [`DecodedImage::planes`] buffer plus its geometry.
/// Samples are one byte each at 8-bit and little-endian `u16` pairs at 10/12-bit
/// (`bytes_per_sample` says which); rows are contiguous, so the sample stride
/// equals `width`. Use this instead of recomputing plane offsets/strides by
/// hand — see [`DecodedImage::luma_plane`], [`chroma_planes`](DecodedImage::chroma_planes)
/// and [`alpha_plane`](DecodedImage::alpha_plane).
#[derive(Clone, Copy, Debug)]
pub struct PlaneView<'a> {
    /// The plane's bytes, exactly `width * height * bytes_per_sample` long.
    pub data: &'a [u8],
    /// Samples per row — also the row stride, since the plane is tightly packed.
    pub width: u32,
    /// Number of rows.
    pub height: u32,
    /// `1` for 8-bit samples, `2` for little-endian `u16` (10/12-bit).
    pub bytes_per_sample: usize,
}

impl PlaneView<'_> {
    /// Number of samples in the plane (`width * height`).
    pub fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// The plane's samples as `u16`: little-endian pairs for 10/12-bit planes, or
    /// each byte widened for 8-bit planes. Allocates a fresh `Vec`; handy for
    /// passing to libraries that want `&[u16]` planes.
    pub fn to_u16(&self) -> Vec<u16> {
        if self.bytes_per_sample == 2 {
            self.data
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        } else {
            self.data.iter().map(|&b| b as u16).collect()
        }
    }
}

impl DecodedImage {
    /// Bytes per sample in [`planes`](Self::planes): 2 above 8-bit, else 1.
    fn bytes_per_sample(&self) -> usize {
        if self.bit_depth.bits() > 8 { 2 } else { 1 }
    }

    /// Dimensions of each chroma plane for this image's chroma format
    /// (`ceil(width / SubWidthC) × ceil(height / SubHeightC)`).
    fn chroma_dims(&self) -> (u32, u32) {
        (
            self.width.div_ceil(self.chroma.sub_w() as u32),
            self.height.div_ceil(self.chroma.sub_h() as u32),
        )
    }

    /// View of the luma (Y) plane.
    pub fn luma_plane(&self) -> PlaneView<'_> {
        let bps = self.bytes_per_sample();
        let len = self.width as usize * self.height as usize * bps;
        PlaneView {
            data: &self.planes[..len],
            width: self.width,
            height: self.height,
            bytes_per_sample: bps,
        }
    }

    /// Views of the chroma planes as `(Cb, Cr)`, or `None` for a monochrome
    /// image (which has no chroma).
    pub fn chroma_planes(&self) -> Option<(PlaneView<'_>, PlaneView<'_>)> {
        if self.chroma == ChromaFormat::Monochrome {
            return None;
        }
        let bps = self.bytes_per_sample();
        let y_len = self.width as usize * self.height as usize * bps;
        let (cw, ch) = self.chroma_dims();
        let c_len = cw as usize * ch as usize * bps;
        let cb = PlaneView {
            data: &self.planes[y_len..y_len + c_len],
            width: cw,
            height: ch,
            bytes_per_sample: bps,
        };
        let cr = PlaneView {
            data: &self.planes[y_len + c_len..y_len + 2 * c_len],
            width: cw,
            height: ch,
            bytes_per_sample: bps,
        };
        Some((cb, cr))
    }

    /// View of the alpha plane (the luma of the auxiliary alpha image), or
    /// `None` when the file carried no alpha.
    pub fn alpha_plane(&self) -> Option<PlaneView<'_>> {
        self.alpha.as_ref().map(|a| a.luma_plane())
    }
}

/// Split an Annex-B stream into NAL units, returning `(nal_type, payload)` where
/// `payload` is the bytes after the 2-byte NAL header with emulation prevention
/// removed (i.e. the RBSP).
fn split_nals(stream: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut end = if k + 1 < starts.len() {
            starts[k + 1] - 3
        } else {
            stream.len()
        };
        if end > s && k + 1 < starts.len() && stream[end - 1] == 0 {
            end -= 1;
        }
        if end >= s + 2 {
            let nal_type = (stream[s + 1] >> 3) & 0x1f;
            let rbsp = ebsp_to_rbsp(&stream[s + 2..end]);
            out.push((nal_type, rbsp));
        }
    }
    out
}

/// Decode a raw Annex-B VVC intra still stream into reconstructed YCbCr samples.
///
/// This is hardened for untrusted input: malformed streams return
/// [`EncodeError::Decode`] rather than panicking. Resource-driving header
/// fields (image dimensions) are bounded before any allocation, the bit and
/// CABAC readers saturate at end-of-input, and a `catch_unwind` backstop
/// converts any residual indexing panic from a corrupt bitstream into an error.
/// (Requires the default `panic = "unwind"` strategy.)
pub fn decode_266(annexb: &[u8]) -> Result<DecodedImage, EncodeError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_inner(annexb))).unwrap_or(Err(
        EncodeError::Decode("malformed bitstream (decode aborted)"),
    ))
}

fn decode_inner(annexb: &[u8]) -> Result<DecodedImage, EncodeError> {
    let nals = split_nals(annexb);
    let sps_rbsp = nals
        .iter()
        .find(|(t, _)| *t == 15)
        .map(|(_, d)| d.as_slice())
        .ok_or(EncodeError::Decode("no SPS NAL"))?;
    let pps_rbsp = nals
        .iter()
        .find(|(t, _)| *t == 16)
        .map(|(_, d)| d.as_slice())
        .ok_or(EncodeError::Decode("no PPS NAL"))?;
    // IDR slice: VCL NAL type 7 (IDR_W_RADL) or 8 (IDR_N_LP).
    let slice_rbsp = nals
        .iter()
        .find(|(t, _)| *t == 7 || *t == 8)
        .map(|(_, d)| d.as_slice())
        .ok_or(EncodeError::Decode("no IDR slice NAL"))?;

    let sps = parse_sps(sps_rbsp)?;
    let lossless = sps.lossless;
    let sps_lfnst = sps.lfnst;
    let dep_quant = sps.dep_quant;
    let sps_mts = sps.mts;
    let (qp, cu_qp_delta_enabled, deblock_on) = parse_pps_qp(pps_rbsp, sps.bit_depth)?;
    let mut db_leaves: Vec<(usize, usize, usize, usize)> = Vec::new();
    let (sd_off, cu_qp_delta_subdiv) = slice_data_offset(slice_rbsp, cu_qp_delta_enabled)?;
    let slice_data = slice_rbsp
        .get(sd_off..)
        .ok_or(EncodeError::Decode("slice-data offset past end of NAL"))?;

    let (width, height) = (sps.width as usize, sps.height as usize);
    // Bound the work driven by attacker-controlled header fields: reject zero or
    // out-of-spec dimensions before allocating any plane. The caps match the
    // Level 6.2 limits garnetash declares in the SPS, applied to the *coded*
    // picture (luma padded to a multiple of 8), so peak memory stays bounded
    // (worst case ~3 planes of i32+bool at the coded size) and a hostile SPS
    // cannot trigger an allocation-failure abort. These mirror the encoder's
    // limits, so every picture garnetash can emit, garnetash can decode.
    const MAX_DIMENSION: usize = 16_888; // floor(sqrt(8 · MaxLumaPs)), level 6.2
    const MAX_LUMA_SAMPLES: usize = 35_651_584; // MaxLumaPs, level 6.2
    if width == 0 || height == 0 {
        return Err(EncodeError::Decode("image dimensions out of range"));
    }
    let cw_chk = (width + 7) & !7;
    let ch_chk = (height + 7) & !7;
    if cw_chk > MAX_DIMENSION || ch_chk > MAX_DIMENSION {
        return Err(EncodeError::Decode("image dimensions out of range"));
    }
    if cw_chk * ch_chk > MAX_LUMA_SAMPLES {
        return Err(EncodeError::Decode("image too large to decode"));
    }
    let (sub_w, sub_h) = (sps.chroma.sub_w(), sps.chroma.sub_h());
    let has_chroma = !sps.chroma.is_monochrome();
    let is_422 = sub_w == 2 && sub_h == 1;
    let bd = sps.bit_depth.bits();
    let max_val = (1i32 << bd) - 1;

    // Coded dimensions: luma padded up to a multiple of 8 (Max(8, MinCbSizeY)).
    let cw = (width + 7) & !7;
    let ch = (height + 7) & !7;
    let cwc = if has_chroma { cw / sub_w } else { 0 };
    let chc = if has_chroma { ch / sub_h } else { 0 };

    let mut rec_y = vec![0i32; cw * ch];
    let mut av_y = vec![false; cw * ch];
    let mut rec_cb = vec![0i32; cwc * chc];
    let mut av_cb = vec![false; cwc * chc];
    let mut rec_cr = vec![0i32; cwc * chc];
    let mut av_cr = vec![false; cwc * chc];

    let mut dec = CabacDecoder::new(slice_data);
    // CABAC context init uses SliceQpY = Qp'Y - QpBdOffset (H.266 9.3.2.2).
    let slice_qp = (qp as i32 - 6 * (bd as i32 - 8)).clamp(0, 63) as u8;
    let mut ctx = Contexts::new_intra(slice_qp);

    // Reusable prediction buffers, shared across every block to keep the hot
    // reconstruction loop allocation-free.
    let mut pred_buf: Vec<i32> = Vec::with_capacity(crate::transform::MAX_TB);
    let mut pred_scratch: Vec<i32> = Vec::new();

    // Adaptive-quant QP state (sub-CTU quantization groups). The QG size is
    // derived from the signalled cu_qp_delta subdivision (128 >> (subdiv/2)); the
    // predictor runs in the pre-QpBdOffset SliceQpY domain (which may be negative
    // at high bit depth). chroma QP == luma QP here (identity chroma QP table).
    let qp_bd_offset = 6 * (bd as i32 - 8);
    let base_qpy = qp as i32 - qp_bd_offset; // SliceQpY, unclamped (may be < 0)
    let ctu = crate::partition::CTU_SIZE as usize;
    let qg = if cu_qp_delta_enabled {
        ctu >> (cu_qp_delta_subdiv as usize / 2)
    } else {
        ctu
    };
    let qg = qg.max(1);
    let qg_cols = cw.div_ceil(qg);
    let qg_rows = ch.div_ceil(qg);
    let mut qg_map: Vec<i32> = vec![base_qpy; qg_cols * qg_rows];
    let mut cur_qg: usize = usize::MAX;
    let mut prev_qp: i32 = base_qpy;
    let mut cur_qg_qp_y: i32 = base_qpy;
    let mut qg_dqp_coded = false;
    let _ = qg_rows;

    if sps.dual_tree && has_chroma {
        // ---- Dual-tree decode (separate luma/chroma trees), mirror of the
        // encoder's code_partitions_dual + replay_luma/replay_chroma. The shared
        // 128→64 quadtree is forced (no bin); within each 64×64 region the luma
        // subtree is decoded then the chroma CU (DM = co-located luma mode).
        use crate::partition::MttCfg;
        use crate::partition::test_support::decode_partitions_dual;
        use crate::tu::TreeType;
        use crate::tu::test_support::decode_transform_unit_tree;
        let st = std::cell::RefCell::new(DualRec {
            rec_y: std::mem::take(&mut rec_y),
            av_y: std::mem::take(&mut av_y),
            rec_cb: std::mem::take(&mut rec_cb),
            av_cb: std::mem::take(&mut av_cb),
            rec_cr: std::mem::take(&mut rec_cr),
            av_cr: std::mem::take(&mut av_cr),
            pred: std::mem::take(&mut pred_buf),
            pred_scratch: std::mem::take(&mut pred_scratch),
            modes: vec![0u8; (cw / 4) * (ch / 4)],
            mode_cx: cw / 4,
        });
        let luma_cfg = if sps.mtt {
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
        // Adaptive-quant QG state shared by the luma (resolves) and chroma (reads)
        // dual closures. One QG per 64×64 region: the region's luma codes
        // cu_qp_delta before its chroma CU, so chroma reads the resolved QG QP.
        struct DualQg {
            map: Vec<i32>,
            cur: usize,
            prev: i32,
            qp_y: i32,
            coded: bool,
            /// Per-4×4 luma Qp'Y grid; chroma reads its collocated centre.
            luma_qp: Vec<u8>,
        }
        let qg_state = std::cell::RefCell::new(DualQg {
            map: vec![base_qpy; qg_cols * qg_rows],
            cur: usize::MAX,
            prev: base_qpy,
            qp_y: base_qpy,
            coded: false,
            luma_qp: vec![qp; cw.div_ceil(4) * ch.div_ceil(4)],
        });
        decode_partitions_dual(
            &mut dec,
            &mut ctx,
            cw as u32,
            ch as u32,
            luma_cfg,
            chroma_cfg,
            |d, c, grid, x, y, w, h| {
                let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
                let mut s = st.borrow_mut();
                // Quantization-group entry: derive the QP predictor (same rule as
                // the encoder) and tentatively fill the map.
                if cu_qp_delta_enabled {
                    let (qx, qy) = (x & !(qg - 1), y & !(qg - 1));
                    let qidx = (qy / qg) * qg_cols + qx / qg;
                    let mut q = qg_state.borrow_mut();
                    if qidx != q.cur {
                        q.prev = q.qp_y;
                        q.qp_y =
                            crate::encode::aq_predict_qp(&q.map, qg_cols, qg, ctu, qx, qy, q.prev);
                        let pred = q.qp_y;
                        crate::encode::aq_fill_qg(&mut q.map, qg_cols, qg, x, y, w, pred);
                        q.coded = false;
                        q.cur = qidx;
                    }
                }
                let luma_bdpcm = if lossless {
                    decode_bdpcm_mode(d, c, true)
                } else {
                    0
                };
                let mode = if luma_bdpcm == 0 {
                    let mpm = build_mpm(
                        grid.left_mode_rect(x as u32, y as u32, h as u32),
                        grid.above_mode_rect(x as u32, y as u32, w as u32),
                    );
                    decode_luma_mode(d, c, &mpm)
                } else if luma_bdpcm == 1 {
                    HOR_IDX
                } else {
                    VER_IDX
                };
                grid.set_mode_rect(x as u32, y as u32, w as u32, h as u32, mode);
                s.set_mode(x, y, w, h, mode);
                let read_dqp = cu_qp_delta_enabled && !qg_state.borrow().coded;
                let tu = decode_transform_unit_tree(
                    d,
                    c,
                    w,
                    h,
                    None,
                    TreeType::Luma,
                    dep_quant,
                    read_dqp,
                );
                if let Some(delta) = tu.dqp {
                    let mut q = qg_state.borrow_mut();
                    q.qp_y = if delta == 0 {
                        q.qp_y
                    } else {
                        ((q.qp_y + delta + 64 + 2 * qp_bd_offset) % (64 + qp_bd_offset))
                            - qp_bd_offset
                    };
                    let res = q.qp_y;
                    crate::encode::aq_fill_qg(&mut q.map, qg_cols, qg, x, y, w, res);
                    q.coded = true;
                }
                let leaf_qp = if cu_qp_delta_enabled {
                    (qg_state.borrow().qp_y + qp_bd_offset).clamp(0, 63 + qp_bd_offset) as u8
                } else {
                    qp
                };
                if cu_qp_delta_enabled {
                    let cw4 = cw.div_ceil(4);
                    let mut q = qg_state.borrow_mut();
                    for yy in (y / 4)..((y + h).div_ceil(4)) {
                        for xx in (x / 4)..((x + w).div_ceil(4)) {
                            q.luma_qp[yy * cw4 + xx] = leaf_qp;
                        }
                    }
                }
                let mut lfnst_idx = 0u8;
                if sps_lfnst && !lossless {
                    let lts = tu.luma_ts || luma_bdpcm != 0;
                    if crate::residual::lfnst_present(
                        sps_lfnst,
                        w,
                        h,
                        (tu.luma.as_slice(), lts),
                        None,
                    ) {
                        lfnst_idx = decode_lfnst_idx(d, c, true);
                    }
                }
                let lfnst_mode = if lfnst_idx > 0 {
                    crate::lfnst::lfnst_intra_mode(crate::predict::lfnst_wide_angle(
                        w,
                        h,
                        mode as i32,
                    ))
                } else {
                    0
                };
                let mut mts_idx = 0u8;
                if sps_mts
                    && !lossless
                    && lfnst_idx == 0
                    && !tu.luma_ts
                    && luma_bdpcm == 0
                    && w <= 32
                    && h <= 32
                    && crate::residual::mts_signallable(&tu.luma, w, h)
                {
                    mts_idx = decode_mts_idx(d, c);
                }
                let s = &mut *s;
                let refs_y = gather_refs_wh(&s.rec_y, &s.av_y, cw, ch, x, y, w, h, bd);
                if luma_bdpcm != 0 {
                    let blk = undo_dpcm(&tu.luma, &refs_y, w, h, luma_bdpcm);
                    place_block(&mut s.rec_y, &mut s.av_y, cw, x, y, w, h, &blk);
                } else {
                    s.pred.clear();
                    s.pred.resize(w * h, 0);
                    predict_into(
                        &mut s.pred,
                        &mut s.pred_scratch,
                        None,
                        mode,
                        w,
                        h,
                        &refs_y,
                        bd,
                        true,
                    );
                    reconstruct_wh(
                        &mut s.rec_y,
                        &mut s.av_y,
                        cw,
                        x,
                        y,
                        w,
                        h,
                        &s.pred,
                        &tu.luma,
                        leaf_qp,
                        bd,
                        max_val,
                        lossless,
                        tu.luma_ts,
                        lfnst_idx,
                        lfnst_mode,
                        dep_quant,
                        mts_idx,
                    );
                }
            },
            |d, c, _grid, x, y, w, h| {
                let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
                let mut s = st.borrow_mut();
                let dm = s.dm(x, y, w, h);
                let chroma_bdpcm = if lossless {
                    decode_bdpcm_mode(d, c, false)
                } else {
                    0
                };
                let chroma_mode = if chroma_bdpcm == 0 {
                    decode_chroma_mode(d, c, dm, sps.cclm)
                } else if chroma_bdpcm == 1 {
                    HOR_IDX
                } else {
                    VER_IDX
                };
                let (ccw, cch) = (w / sub_w, h / sub_h);
                // Chroma QP = QP of the luma CU collocated with the chroma CU's
                // centre (H.266 / VTM colLumaCu->qp), read from the per-4×4 grid.
                let cqp = if cu_qp_delta_enabled {
                    let cw4 = cw.div_ceil(4);
                    let lx = ((x / sub_w + ((w / sub_w) >> 1)) * sub_w).min(cw - 1);
                    let ly = ((y / sub_h + ((h / sub_h) >> 1)) * sub_h).min(ch - 1);
                    qg_state.borrow().luma_qp[(ly / 4) * cw4 + lx / 4]
                } else {
                    qp
                };
                let tu = decode_transform_unit_tree(
                    d,
                    c,
                    w,
                    h,
                    Some((ccw, cch)),
                    TreeType::Chroma,
                    dep_quant,
                    false,
                );
                // Chroma carries its own LFNST index in the dual tree; consume it
                // (the encoder applies no chroma LFNST, so it is always 0 here).
                if sps_lfnst && !lossless {
                    let cts = tu.cb_ts || tu.cr_ts || chroma_bdpcm != 0;
                    let chroma = Some((tu.cb.as_slice(), tu.cr.as_slice(), ccw, cch, cts));
                    if crate::residual::lfnst_present(sps_lfnst, ccw, cch, (&[], false), chroma) {
                        let _ = decode_lfnst_idx(d, c, true);
                    }
                }
                let (cx, cy) = (x / sub_w, y / sub_h);
                let s = &mut *s;
                let refs_cb = gather_refs_wh(&s.rec_cb, &s.av_cb, cwc, chc, cx, cy, ccw, cch, bd);
                let refs_cr = gather_refs_wh(&s.rec_cr, &s.av_cr, cwc, chc, cx, cy, ccw, cch, bd);
                if chroma_bdpcm != 0 {
                    let cb = undo_dpcm(&tu.cb, &refs_cb, ccw, cch, chroma_bdpcm);
                    let cr = undo_dpcm(&tu.cr, &refs_cr, ccw, cch, chroma_bdpcm);
                    place_block(&mut s.rec_cb, &mut s.av_cb, cwc, cx, cy, ccw, cch, &cb);
                    place_block(&mut s.rec_cr, &mut s.av_cr, cwc, cx, cy, ccw, cch, &cr);
                } else if crate::intra::is_cclm_mode(chroma_mode) {
                    let lh = s.rec_y.len() / cw;
                    let luma = |xx: isize, yy: isize| {
                        let xx = xx.clamp(0, cw as isize - 1) as usize;
                        let yy = yy.clamp(0, lh as isize - 1) as usize;
                        s.rec_y[yy * cw + xx]
                    };
                    let cba = |xx: isize, yy: isize| {
                        let xx = xx.clamp(0, cwc as isize - 1) as usize;
                        let yy = yy.clamp(0, chc as isize - 1) as usize;
                        s.rec_cb[yy * cwc + xx]
                    };
                    let cra = |xx: isize, yy: isize| {
                        let xx = xx.clamp(0, cwc as isize - 1) as usize;
                        let yy = yy.clamp(0, chc as isize - 1) as usize;
                        s.rec_cr[yy * cwc + xx]
                    };
                    let (above, left) = (cy > 0, cx > 0);
                    let uw = 4 >> if sub_w == 2 { 1 } else { 0 };
                    let uh = 4 >> if sub_h == 2 { 1 } else { 0 };
                    let avai_ar = if above {
                        let mut n = 0;
                        for u in 0..ccw / uw {
                            let col = cx + ccw + u * uw;
                            if col >= cwc || !s.av_cb[(cy - 1) * cwc + col] {
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
                            if row >= chc || !s.av_cb[row * cwc + (cx - 1)] {
                                break;
                            }
                            n += 1;
                        }
                        n
                    } else {
                        0
                    };
                    let first_row = (y & (crate::partition::CTU_SIZE as usize - 1)) == 0;
                    let (pcb, pcr) = crate::cclm::cclm_predict(
                        luma,
                        cba,
                        cra,
                        x,
                        y,
                        cx,
                        cy,
                        ccw,
                        cch,
                        sub_w,
                        sub_h,
                        above,
                        left,
                        first_row,
                        chroma_mode,
                        avai_ar,
                        avai_bl,
                        max_val,
                        bd,
                    );
                    reconstruct_wh(
                        &mut s.rec_cb,
                        &mut s.av_cb,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &pcb,
                        &tu.cb,
                        cqp,
                        bd,
                        max_val,
                        lossless,
                        tu.cb_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                    reconstruct_wh(
                        &mut s.rec_cr,
                        &mut s.av_cr,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &pcr,
                        &tu.cr,
                        cqp,
                        bd,
                        max_val,
                        lossless,
                        tu.cr_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                } else {
                    let pmode = if is_422 {
                        chroma_422_mode(chroma_mode)
                    } else {
                        chroma_mode
                    };
                    s.pred.clear();
                    s.pred.resize(ccw * cch, 0);
                    predict_into(
                        &mut s.pred,
                        &mut s.pred_scratch,
                        None,
                        pmode,
                        ccw,
                        cch,
                        &refs_cb,
                        bd,
                        false,
                    );
                    reconstruct_wh(
                        &mut s.rec_cb,
                        &mut s.av_cb,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &s.pred,
                        &tu.cb,
                        cqp,
                        bd,
                        max_val,
                        lossless,
                        tu.cb_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                    s.pred.clear();
                    s.pred.resize(ccw * cch, 0);
                    predict_into(
                        &mut s.pred,
                        &mut s.pred_scratch,
                        None,
                        pmode,
                        ccw,
                        cch,
                        &refs_cr,
                        bd,
                        false,
                    );
                    reconstruct_wh(
                        &mut s.rec_cr,
                        &mut s.av_cr,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &s.pred,
                        &tu.cr,
                        cqp,
                        bd,
                        max_val,
                        lossless,
                        tu.cr_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                }
            },
        );
        let s = st.into_inner();
        rec_y = s.rec_y;
        rec_cb = s.rec_cb;
        rec_cr = s.rec_cr;
    } else {
        let mut leaf = |dec: &mut CabacDecoder,
                        ctx: &mut Contexts,
                        grid: &mut crate::partition::NeighborGrid,
                        x: usize,
                        y: usize,
                        w: usize,
                        h: usize| {
            // Quantization-group entry: derive the QP predictor (same rule as the
            // encoder) and fill the map tentatively.
            if cu_qp_delta_enabled {
                let (qx, qy) = (x & !(qg - 1), y & !(qg - 1));
                let qidx = (qy / qg) * qg_cols + qx / qg;
                if qidx != cur_qg {
                    prev_qp = cur_qg_qp_y;
                    cur_qg_qp_y =
                        crate::encode::aq_predict_qp(&qg_map, qg_cols, qg, ctu, qx, qy, prev_qp);
                    crate::encode::aq_fill_qg(&mut qg_map, qg_cols, qg, x, y, w, cur_qg_qp_y);
                    qg_dqp_coded = false;
                    cur_qg = qidx;
                }
            }

            // BDPCM flag (lossless only), then the luma intra mode. Under luma
            // BDPCM the mode is inferred from the direction (H/V) rather than coded.
            let luma_bdpcm = if lossless {
                decode_bdpcm_mode(dec, ctx, true)
            } else {
                0
            };
            let mode = if luma_bdpcm == 0 {
                let mpm = build_mpm(
                    grid.left_mode_rect(x as u32, y as u32, h as u32),
                    grid.above_mode_rect(x as u32, y as u32, w as u32),
                );
                decode_luma_mode(dec, ctx, &mpm)
            } else if luma_bdpcm == 1 {
                HOR_IDX
            } else {
                VER_IDX
            };
            grid.set_mode_rect(x as u32, y as u32, w as u32, h as u32, mode);

            // Chroma BDPCM flag + chroma intra mode (inferred under chroma BDPCM).
            let mut chroma_bdpcm = 0u8;
            let mut chroma_mode = 0u8;
            if has_chroma {
                if lossless {
                    chroma_bdpcm = decode_bdpcm_mode(dec, ctx, false);
                }
                chroma_mode = if chroma_bdpcm == 0 {
                    decode_chroma_mode(dec, ctx, mode, sps.cclm)
                } else if chroma_bdpcm == 1 {
                    HOR_IDX
                } else {
                    VER_IDX
                };
            }

            // Transform unit: CBFs, optional cu_qp_delta, residual levels and TS.
            let (ccw, cch) = (w / sub_w, h / sub_h);
            let chroma_dims = if has_chroma { Some((ccw, cch)) } else { None };
            let read_dqp = cu_qp_delta_enabled && !qg_dqp_coded;
            let (tu, dqp) = decode_transform_unit_full(
                dec,
                ctx,
                w,
                h,
                chroma_dims,
                luma_bdpcm != 0,
                chroma_bdpcm != 0,
                dep_quant,
                read_dqp,
            );
            if let Some(delta) = dqp {
                // Resolve QpY for this QG (H.266 cu_qp_delta modulo wrap), store it
                // for neighbour prediction, and stop coding further deltas here.
                cur_qg_qp_y = if delta == 0 {
                    cur_qg_qp_y
                } else {
                    ((cur_qg_qp_y + delta + 64 + 2 * qp_bd_offset) % (64 + qp_bd_offset))
                        - qp_bd_offset
                };
                crate::encode::aq_fill_qg(&mut qg_map, qg_cols, qg, x, y, w, cur_qg_qp_y);
                qg_dqp_coded = true;
            }
            // Per-leaf Qp'Y (= QpY + QpBdOffset). Uniform `qp` when AQ is disabled.
            let leaf_qp = if cu_qp_delta_enabled {
                (cur_qg_qp_y + qp_bd_offset).clamp(0, 63 + qp_bd_offset) as u8
            } else {
                qp
            };

            // residual_lfnst_mode: present (and thus read) iff the just-decoded
            // coefficients meet the VTM presence conditions. Mirror of the encoder.
            let mut lfnst_idx = 0u8;
            if sps_lfnst && !lossless {
                let lts = tu.luma_ts || luma_bdpcm != 0;
                let chroma = if has_chroma {
                    Some((
                        tu.cb.as_slice(),
                        tu.cr.as_slice(),
                        ccw,
                        cch,
                        tu.cb_ts || tu.cr_ts || chroma_bdpcm != 0,
                    ))
                } else {
                    None
                };
                if crate::residual::lfnst_present(
                    sps_lfnst,
                    w,
                    h,
                    (tu.luma.as_slice(), lts),
                    chroma,
                ) {
                    lfnst_idx = decode_lfnst_idx(dec, ctx, false);
                }
            }
            let lfnst_mode = if lfnst_idx > 0 {
                crate::lfnst::lfnst_intra_mode(crate::predict::lfnst_wide_angle(w, h, mode as i32))
            } else {
                0
            };

            // mts_idx: read after residual_lfnst_mode under the same gate the encoder
            // applied (MTS enabled, luma TU ≤32, lfnst_idx == 0, not transform-skip,
            // levels signallable). Selects the per-TU DST-VII/DCT-VIII transform pair.
            let mut mts_idx = 0u8;
            if sps_mts
                && !lossless
                && lfnst_idx == 0
                && !tu.luma_ts
                && luma_bdpcm == 0
                && w <= 32
                && h <= 32
                && crate::residual::mts_signallable(&tu.luma, w, h)
            {
                mts_idx = decode_mts_idx(dec, ctx);
            }

            // Luma reconstruction.
            let refs_y = gather_refs_wh(&rec_y, &av_y, cw, ch, x, y, w, h, bd);
            if luma_bdpcm != 0 {
                let blk = undo_dpcm(&tu.luma, &refs_y, w, h, luma_bdpcm);
                place_block(&mut rec_y, &mut av_y, cw, x, y, w, h, &blk);
            } else {
                pred_buf.clear();
                pred_buf.resize(w * h, 0);
                predict_into(
                    &mut pred_buf,
                    &mut pred_scratch,
                    None,
                    mode,
                    w,
                    h,
                    &refs_y,
                    bd,
                    true,
                );
                reconstruct_wh(
                    &mut rec_y, &mut av_y, cw, x, y, w, h, &pred_buf, &tu.luma, leaf_qp, bd,
                    max_val, lossless, tu.luma_ts, lfnst_idx, lfnst_mode, dep_quant, mts_idx,
                );
            }

            // Chroma reconstruction (at chroma resolution). For 4:2:2 the predictor
            // uses the re-angled mode (H.266 Table 8-3); the coded mode is unchanged.
            if has_chroma {
                let (cx, cy) = (x / sub_w, y / sub_h);
                let refs_cb = gather_refs_wh(&rec_cb, &av_cb, cwc, chc, cx, cy, ccw, cch, bd);
                let refs_cr = gather_refs_wh(&rec_cr, &av_cr, cwc, chc, cx, cy, ccw, cch, bd);
                if chroma_bdpcm != 0 {
                    let cb = undo_dpcm(&tu.cb, &refs_cb, ccw, cch, chroma_bdpcm);
                    let cr = undo_dpcm(&tu.cr, &refs_cr, ccw, cch, chroma_bdpcm);
                    place_block(&mut rec_cb, &mut av_cb, cwc, cx, cy, ccw, cch, &cb);
                    place_block(&mut rec_cr, &mut av_cr, cwc, cx, cy, ccw, cch, &cr);
                } else if crate::intra::is_cclm_mode(chroma_mode) {
                    // CCLM: predict chroma from the just-reconstructed collocated luma
                    // (rec_y was written above) plus the neighbouring chroma/luma.
                    let lh = rec_y.len() / cw;
                    let luma = |xx: isize, yy: isize| {
                        let xx = xx.clamp(0, cw as isize - 1) as usize;
                        let yy = yy.clamp(0, lh as isize - 1) as usize;
                        rec_y[yy * cw + xx]
                    };
                    let cba = |xx: isize, yy: isize| {
                        let xx = xx.clamp(0, cwc as isize - 1) as usize;
                        let yy = yy.clamp(0, chc as isize - 1) as usize;
                        rec_cb[yy * cwc + xx]
                    };
                    let cra = |xx: isize, yy: isize| {
                        let xx = xx.clamp(0, cwc as isize - 1) as usize;
                        let yy = yy.clamp(0, chc as isize - 1) as usize;
                        rec_cr[yy * cwc + xx]
                    };
                    let (above, left) = (cy > 0, cx > 0);
                    let uw = 4 >> if sub_w == 2 { 1 } else { 0 };
                    let uh = 4 >> if sub_h == 2 { 1 } else { 0 };
                    let avai_ar = if above {
                        let mut n = 0;
                        for u in 0..ccw / uw {
                            let col = cx + ccw + u * uw;
                            if col >= cwc || !av_cb[(cy - 1) * cwc + col] {
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
                            if row >= chc || !av_cb[row * cwc + (cx - 1)] {
                                break;
                            }
                            n += 1;
                        }
                        n
                    } else {
                        0
                    };
                    let first_row = (y & (crate::partition::CTU_SIZE as usize - 1)) == 0;
                    let (pcb, pcr) = crate::cclm::cclm_predict(
                        luma,
                        cba,
                        cra,
                        x,
                        y,
                        cx,
                        cy,
                        ccw,
                        cch,
                        sub_w,
                        sub_h,
                        above,
                        left,
                        first_row,
                        chroma_mode,
                        avai_ar,
                        avai_bl,
                        max_val,
                        bd,
                    );
                    reconstruct_wh(
                        &mut rec_cb,
                        &mut av_cb,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &pcb,
                        &tu.cb,
                        leaf_qp,
                        bd,
                        max_val,
                        lossless,
                        tu.cb_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                    reconstruct_wh(
                        &mut rec_cr,
                        &mut av_cr,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &pcr,
                        &tu.cr,
                        leaf_qp,
                        bd,
                        max_val,
                        lossless,
                        tu.cr_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                } else {
                    let pmode = if is_422 {
                        chroma_422_mode(chroma_mode)
                    } else {
                        chroma_mode
                    };
                    pred_buf.clear();
                    pred_buf.resize(ccw * cch, 0);
                    predict_into(
                        &mut pred_buf,
                        &mut pred_scratch,
                        None,
                        pmode,
                        ccw,
                        cch,
                        &refs_cb,
                        bd,
                        false,
                    );
                    reconstruct_wh(
                        &mut rec_cb,
                        &mut av_cb,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &pred_buf,
                        &tu.cb,
                        leaf_qp,
                        bd,
                        max_val,
                        lossless,
                        tu.cb_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                    pred_buf.clear();
                    pred_buf.resize(ccw * cch, 0);
                    predict_into(
                        &mut pred_buf,
                        &mut pred_scratch,
                        None,
                        pmode,
                        ccw,
                        cch,
                        &refs_cr,
                        bd,
                        false,
                    );
                    reconstruct_wh(
                        &mut rec_cr,
                        &mut av_cr,
                        cwc,
                        cx,
                        cy,
                        ccw,
                        cch,
                        &pred_buf,
                        &tu.cr,
                        leaf_qp,
                        bd,
                        max_val,
                        lossless,
                        tu.cr_ts,
                        0,
                        0,
                        dep_quant,
                        0,
                    );
                }
            }
        };

        if sps.mtt {
            crate::partition::test_support::decode_partitions_mtt(
                &mut dec,
                &mut ctx,
                cw as u32,
                ch as u32,
                crate::partition::mtt_cfg(),
                |d, c, g, x, y, w, h| leaf(d, c, g, x as usize, y as usize, w as usize, h as usize),
            );
        } else {
            decode_partitions(
                &mut dec,
                &mut ctx,
                cw as u32,
                ch as u32,
                |d, c, g, x, y, s| {
                    if deblock_on {
                        db_leaves.push((x as usize, y as usize, s as usize, s as usize));
                    }
                    leaf(d, c, g, x as usize, y as usize, s as usize, s as usize)
                },
            );
        }
    }

    if dec.decode_terminate() != 1 {
        return Err(EncodeError::Decode("slice data did not terminate cleanly"));
    }

    if deblock_on {
        // Same filter the encoder applied to its reconstruction. Single-tree,
        // AQ off (a precondition for the encoder enabling the flag), so the grid
        // carries the uniform picture QP and the chroma partition mirrors luma.
        let mut grid = crate::deblock::Grid::new(cw, ch);
        for &(x, y, w, h) in &db_leaves {
            grid.set_cu(x, y, w, h, qp);
        }
        crate::deblock::deblock_luma(&mut rec_y, cw, ch, &grid, bd, ctu);
        if has_chroma {
            let subx = sub_w.trailing_zeros() as usize;
            let suby = sub_h.trailing_zeros() as usize;
            crate::deblock::deblock_chroma(&mut rec_cb, cwc, chc, &grid, subx, suby, bd, ctu);
            crate::deblock::deblock_chroma(&mut rec_cr, cwc, chc, &grid, subx, suby, bd, ctu);
        }
    }

    // Crop to display dimensions, same layout as the encoder's reconstruction.
    let (dcw, dch) = if has_chroma {
        (width.div_ceil(sub_w), height.div_ceil(sub_h))
    } else {
        (0, 0)
    };
    let two_byte = bd > 8;
    let bytes_per = if two_byte { 2 } else { 1 };
    let mut planes = Vec::with_capacity((width * height + 2 * dcw * dch) * bytes_per);
    let push = |v: i32, out: &mut Vec<u8>| {
        if two_byte {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        } else {
            out.push(v as u8);
        }
    };
    for row in rec_y.chunks_exact(cw).take(height) {
        for &v in &row[..width] {
            push(v, &mut planes);
        }
    }
    if has_chroma {
        for row in rec_cb.chunks_exact(cwc).take(dch) {
            for &v in &row[..dcw] {
                push(v, &mut planes);
            }
        }
        for row in rec_cr.chunks_exact(cwc).take(dch) {
            for &v in &row[..dcw] {
                push(v, &mut planes);
            }
        }
    }

    Ok(DecodedImage {
        width: sps.width,
        height: sps.height,
        chroma: sps.chroma,
        bit_depth: sps.bit_depth,
        planes,
        orientation: Orientation::default(),
        color: ColorMetadata::default(),
        alpha: None,
    })
}

/// Invert the BDPCM differential coding (the decode-side counterpart of the
/// encoder's `dpcm_levels_wh`): accumulate the decoded differences along the
/// BDPCM direction, seeded by the block's reconstructed boundary samples. The
/// result is the reconstructed (lossless) block.
fn undo_dpcm(lv: &[i32], refs: &RefSamples, bw: usize, bh: usize, dir: u8) -> Vec<i32> {
    let mut src = vec![0i32; bw * bh];
    if dir == 1 {
        // Horizontal BDPCM: running sum left-to-right, seeded by the left edge.
        for (y, (srow, lrow)) in src
            .chunks_exact_mut(bw)
            .zip(lv.chunks_exact(bw))
            .enumerate()
        {
            let mut prev = refs.left[y + 1];
            for (d, &l) in srow.iter_mut().zip(lrow) {
                prev += l;
                *d = prev;
            }
        }
    } else {
        // Vertical BDPCM: running sum top-to-bottom; `acc` carries the row above
        // (seeded by the top edge), avoiding a read of the row being written.
        let mut acc = [0i32; 64];
        for (a, &t) in acc[..bw].iter_mut().zip(&refs.top[1..1 + bw]) {
            *a = t;
        }
        for (srow, lrow) in src.chunks_exact_mut(bw).zip(lv.chunks_exact(bw)) {
            for ((d, &l), a) in srow.iter_mut().zip(lrow).zip(acc[..bw].iter_mut()) {
                *a += l;
                *d = *a;
            }
        }
    }
    src
}

/// Write a reconstructed block into a plane and mark its samples available.
fn place_block(
    recon: &mut [i32],
    avail: &mut [bool],
    cw: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    blk: &[i32],
) {
    for (yy, blk_row) in blk.chunks_exact(bw).take(bh).enumerate() {
        let off = (y + yy) * cw + x;
        recon[off..off + bw].copy_from_slice(blk_row);
        avail[off..off + bw].fill(true);
    }
}

impl DecodedImage {
    /// Convert the decoded YCbCr samples to packed 8-bit RGB (`width*height*3`,
    /// R,G,B order). This applies the inverse of the encoder's full-range BT.601
    /// matrix and replicates subsampled chroma; it is a best-effort *display*
    /// conversion, not a bit-exact inverse of the original RGB (the forward
    /// RGB→YCbCr step and chroma subsampling are lossy). The canonical decoded
    /// output is [`planes`](Self::planes) (YCbCr).
    pub fn to_rgb(&self) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut rgb = vec![0u8; w * h * 3];
        if matches!(self.chroma, ChromaFormat::Monochrome) {
            for (px, &y) in rgb.chunks_exact_mut(3).zip(&self.planes[..w * h]) {
                px[0] = y;
                px[1] = y;
                px[2] = y;
            }
            return rgb;
        }
        let sub_w = self.chroma.sub_w();
        let sub_h = self.chroma.sub_h();
        let dcw = w.div_ceil(sub_w);
        let dch = h.div_ceil(sub_h);
        let y_plane = &self.planes[..w * h];
        let cb_plane = &self.planes[w * h..w * h + dcw * dch];
        let cr_plane = &self.planes[w * h + dcw * dch..w * h + 2 * dcw * dch];
        for (yy, (rgb_row, y_row)) in rgb
            .chunks_exact_mut(w * 3)
            .zip(y_plane.chunks_exact(w))
            .enumerate()
        {
            let crow = (yy / sub_h) * dcw;
            for (xx, (px, &yv)) in rgb_row.chunks_exact_mut(3).zip(y_row).enumerate() {
                let y = yv as i32;
                let ci = crow + xx / sub_w;
                let cb = cb_plane[ci] as i32 - 128;
                let cr = cr_plane[ci] as i32 - 128;
                // Full-range BT.601 inverse in Q0.13 (matches encode's forward):
                // 1.402, 0.344136, 0.714136, 1.772 scaled by 8192.
                let r = y + ((11485 * cr + 4096) >> 13);
                let g = y - ((2819 * cb + 5850 * cr + 4096) >> 13);
                let b = y + ((14516 * cb + 4096) >> 13);
                px[0] = r.clamp(0, 255) as u8;
                px[1] = g.clamp(0, 255) as u8;
                px[2] = b.clamp(0, 255) as u8;
            }
        }
        rgb
    }
}

/// Decode a raw Annex-B VVC intra still stream into reconstructed YCbCr samples.
/// This is the canonical YCbCr decode entry; [`decode_yuv_266`] is a named alias for
/// it, mirroring [`encode_yuv_266`](crate::encode_yuv_266) on the encode side.
pub fn decode_yuv_266(annexb: &[u8]) -> Result<DecodedImage, EncodeError> {
    decode_266(annexb)
}

fn crop_to_display_extent(
    img: &mut DecodedImage,
    width: u32,
    height: u32,
) -> Result<(), EncodeError> {
    if width == 0 || height == 0 || width > img.width || height > img.height {
        return Err(EncodeError::Decode("HEIF ispe exceeds coded image"));
    }
    if (width, height) == (img.width, img.height) {
        return Ok(());
    }

    let bps = img.bytes_per_sample();
    let (old_w, old_h) = (img.width as usize, img.height as usize);
    let (new_w, new_h) = (width as usize, height as usize);
    let (sub_w, sub_h) = (img.chroma.sub_w(), img.chroma.sub_h());
    let has_chroma = !img.chroma.is_monochrome();
    let (old_cw, old_ch) = if has_chroma {
        (old_w.div_ceil(sub_w), old_h.div_ceil(sub_h))
    } else {
        (0, 0)
    };
    let (new_cw, new_ch) = if has_chroma {
        (new_w.div_ceil(sub_w), new_h.div_ceil(sub_h))
    } else {
        (0, 0)
    };
    let old_y_len = old_w * old_h * bps;
    let old_c_len = old_cw * old_ch * bps;
    let mut planes = Vec::with_capacity((new_w * new_h + 2 * new_cw * new_ch) * bps);
    for row in 0..new_h {
        let start = row * old_w * bps;
        planes.extend_from_slice(&img.planes[start..start + new_w * bps]);
    }
    if has_chroma {
        for offset in [old_y_len, old_y_len + old_c_len] {
            for row in 0..new_ch {
                let start = offset + row * old_cw * bps;
                planes.extend_from_slice(&img.planes[start..start + new_cw * bps]);
            }
        }
    }
    img.width = width;
    img.height = height;
    img.planes = planes;
    Ok(())
}

/// Decode a HEIF file produced by garnetash's `*_to_heif` entry points. The
/// embedded VVC stream is decoded to YCbCr, and the container is mined for side
/// data: display orientation (`irot`/`imir`) and color metadata (CICP `nclx`
/// and/or an embedded ICC profile) populate the returned image, and if the file
/// carries an alpha auxiliary image it is decoded and attached as
/// `DecodedImage::alpha` automatically. Hardened for untrusted input like
/// [`decode_266`].
pub fn decode(heif: &[u8]) -> Result<DecodedImage, EncodeError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut img = decode_266(&crate::isobmff::extract_vvc_stream(heif)?)?;
        if let Some((width, height)) = crate::isobmff::extract_spatial_extents(heif, 0) {
            crop_to_display_extent(&mut img, width, height)?;
        }
        let (orientation, color) = crate::isobmff::extract_metadata(heif);
        img.orientation = orientation;
        img.color = color;
        img.alpha = crate::isobmff::extract_alpha_stream(heif)
            .and_then(|s| decode_266(&s).ok())
            .map(|mut alpha| {
                if let Some((width, height)) = crate::isobmff::extract_spatial_extents(heif, 1) {
                    crop_to_display_extent(&mut alpha, width, height)?;
                }
                Ok(Box::new(alpha))
            })
            .transpose()?;
        Ok(img)
    }))
    .unwrap_or(Err(EncodeError::Decode("malformed HEIF (decode aborted)")))
}

/// Decode a HEIF file, returning the color image and, separately, any alpha
/// auxiliary image. Retained for explicitness; [`decode`] now also attaches
/// the alpha to `DecodedImage::alpha`, so `let img = decode(b)?;` followed
/// by `img.alpha` is the simpler path.
pub fn decode_with_alpha(heif: &[u8]) -> Result<(DecodedImage, Option<DecodedImage>), EncodeError> {
    let mut img = decode(heif)?;
    let alpha = img.alpha.take().map(|b| *b);
    Ok((img, alpha))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncodeConfig, encode_rgb_with_reconstruction};

    fn make(kind: &str, w: usize, h: usize) -> Vec<u8> {
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                let (r, g, b) = match kind {
                    "flat" => (100u8, 100, 100),
                    "gray" => (128, 128, 128),
                    "grad" => (((x * 255) / w) as u8, ((y * 255) / h) as u8, 128),
                    _ => ((x * 7 + y * 13) as u8, (x * 3) as u8, (y * 5) as u8),
                };
                rgb[o] = r;
                rgb[o + 1] = g;
                rgb[o + 2] = b;
            }
        }
        rgb
    }

    #[test]
    fn dep_quant_decoder_reproduces_reconstruction() {
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            for kind in ["grad", "noise", "gray"] {
                for &(w, h) in &[(16usize, 16usize), (32, 32), (64, 48)] {
                    for q in [40u8, 75, 95] {
                        let cfg = EncodeConfig::new()
                            .with_quality(q)
                            .with_chroma(chroma)
                            .with_dep_quant(true);
                        let rgb = make(kind, w, h);
                        let (stream, recon) =
                            encode_rgb_with_reconstruction(&rgb, w as u32, h as u32, &cfg).unwrap();
                        let img = decode_266(&stream).unwrap();
                        assert_eq!(
                            img.planes, recon,
                            "DQ decode mismatch {chroma:?} {} {w}x{h} q{q}",
                            kind
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn decoder_reproduces_encoder_reconstruction() {
        let formats = [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ];
        for chroma in formats {
            for kind in ["flat", "gray", "grad", "noise"] {
                for &(w, h) in &[(16usize, 16usize), (32, 32), (64, 48), (96, 64)] {
                    // Lossy qualities plus a lossless pass.
                    for cfg in [
                        EncodeConfig::new().with_quality(40).with_chroma(chroma),
                        EncodeConfig::new().with_quality(75).with_chroma(chroma),
                        EncodeConfig::new().with_quality(95).with_chroma(chroma),
                        EncodeConfig::new().with_lossless(true).with_chroma(chroma),
                    ] {
                        let rgb = make(&kind, w, h);
                        let (stream, recon) =
                            encode_rgb_with_reconstruction(&rgb, w as u32, h as u32, &cfg).unwrap();
                        let img = decode_266(&stream).unwrap();
                        assert_eq!(img.width, w as u32);
                        assert_eq!(img.height, h as u32);
                        assert_eq!(img.chroma, chroma);
                        assert_eq!(
                            img.planes, recon,
                            "decode mismatch {chroma:?} {} {w}x{h} lossless={}",
                            kind, cfg.lossless
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lossless_decode_is_exact() {
        // A lossless stream must decode to the original source samples exactly.
        let (w, h) = (40usize, 32usize);
        let rgb = make("noise", w, h);
        let cfg = EncodeConfig::new()
            .with_lossless(true)
            .with_chroma(ChromaFormat::Yuv444);
        let (stream, recon) =
            encode_rgb_with_reconstruction(&rgb, w as u32, h as u32, &cfg).unwrap();
        let img = decode_266(&stream).unwrap();
        assert_eq!(img.planes, recon);
    }

    #[test]
    fn alpha_heif_round_trips_master_and_alpha() {
        use crate::{decode_with_alpha, encode_rgba_266, encode_rgba_with_alpha};
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv444,
            ChromaFormat::Yuv422,
        ] {
            let (w, h) = (48u32, 32u32);
            let mut rgba = vec![0u8; (w * h * 4) as usize];
            for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
                px[0] = (i * 5) as u8;
                px[1] = (i * 3) as u8;
                px[2] = (i * 7) as u8;
                px[3] = (i * 11 + 1) as u8;
            }
            let cfg = EncodeConfig::new().with_lossless(true).with_chroma(chroma);
            let heif = encode_rgba_with_alpha(&rgba, w, h, &cfg).unwrap();
            let (master, alpha) = decode_with_alpha(&heif).unwrap();
            // Master equals the standalone color encode.
            let master_ref = decode_266(&encode_rgba_266(&rgba, w, h, &cfg).unwrap()).unwrap();
            assert_eq!(master.planes, master_ref.planes, "master {chroma:?}");
            // Alpha is monochrome and exactly the A channel (lossless).
            let alpha = alpha.expect("alpha present");
            assert_eq!(alpha.chroma, ChromaFormat::Monochrome);
            let a_channel: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
            assert_eq!(alpha.planes, a_channel, "alpha {chroma:?}");
        }
    }

    #[test]
    fn encode_yuv_round_trips_decoded_planes() {
        // Decode an image, then re-encode its YCbCr planes losslessly via
        // encode_yuv_266; decoding that must reproduce the same planes exactly.
        use crate::encode_yuv_266;
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            for bd in [BitDepth::Eight, BitDepth::Ten] {
                let (w, h) = (48u32, 40u32);
                let rgb = make("grad", w as usize, h as usize);
                let cfg = EncodeConfig::new()
                    .with_quality(75)
                    .with_chroma(chroma)
                    .with_bit_depth(bd);
                let img = decode_266(&crate::encode_rgb_266(&rgb, w, h, &cfg).unwrap()).unwrap();
                let llcfg = EncodeConfig::new()
                    .with_lossless(true)
                    .with_chroma(chroma)
                    .with_bit_depth(bd);
                let stream = encode_yuv_266(&img.planes, w, h, &llcfg).unwrap();
                let img2 = decode_266(&stream).unwrap();
                assert_eq!(img2.planes, img.planes, "{chroma:?} {bd:?}");
            }
        }
        // A wrong-length buffer is rejected, not panicked on.
        assert!(encode_yuv_266(&[0u8; 10], 32, 32, &EncodeConfig::new()).is_err());
    }

    #[test]
    fn malformed_input_never_panics() {
        // A spread of adversarial inputs must all return (Ok or Err), never panic.
        use crate::decode;
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // Pure random buffers.
        for _ in 0..2000 {
            let n = (next() % 2048) as usize;
            let buf: Vec<u8> = (0..n).map(|_| (next() & 0xff) as u8).collect();
            let _ = decode_266(&buf);
            let _ = decode(&buf);
        }
        // Mutated valid streams.
        let rgb = make("noise", 32, 24);
        let stream =
            crate::encode_rgb_266(&rgb, 32, 24, &EncodeConfig::new().with_quality(70)).unwrap();
        let heif = crate::encode_rgb(&rgb, 32, 24, &EncodeConfig::new().with_quality(70)).unwrap();
        for _ in 0..2000 {
            let mut s = stream.clone();
            if !s.is_empty() {
                let i = (next() as usize) % s.len();
                s[i] ^= (next() & 0xff) as u8;
                s.truncate(s.len() - (next() as usize % (s.len().max(1))));
            }
            let _ = decode_266(&s);
            let mut h = heif.clone();
            if !h.is_empty() {
                let i = (next() as usize) % h.len();
                h[i] = (next() & 0xff) as u8;
            }
            let _ = decode(&h);
        }
        // Degenerate / boundary inputs.
        for b in [vec![], vec![0], vec![0, 0, 0, 1], vec![0u8; 64]] {
            let _ = decode_266(&b);
            let _ = decode(&b);
        }
    }

    #[test]
    fn ten_bit_decode_matches_reconstruction() {
        // 10-bit (Main-10) round trip through the decoder, lossy and lossless,
        // across every chroma format.
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            let (w, h) = (48usize, 32usize);
            let rgb = make("noise", w, h);
            for cfg in [
                EncodeConfig::new()
                    .with_quality(75)
                    .with_chroma(chroma)
                    .with_bit_depth(BitDepth::Ten),
                EncodeConfig::new()
                    .with_lossless(true)
                    .with_chroma(chroma)
                    .with_bit_depth(BitDepth::Ten),
            ] {
                let (stream, recon) =
                    encode_rgb_with_reconstruction(&rgb, w as u32, h as u32, &cfg).unwrap();
                let img = decode_266(&stream).unwrap();
                assert_eq!(img.bit_depth, BitDepth::Ten);
                assert_eq!(
                    img.planes, recon,
                    "10-bit decode {chroma:?} lossless={}",
                    cfg.lossless
                );
            }
        }
    }

    #[test]
    fn decode_heif_matches_raw_decode() {
        // Decoding via the HEIF container must equal decoding the raw stream.
        use crate::{decode, encode_rgb};
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            let (w, h) = (64usize, 48usize);
            let rgb = make("grad", w, h);
            let cfg = EncodeConfig::new().with_quality(80).with_chroma(chroma);
            let (raw_stream, recon) =
                encode_rgb_with_reconstruction(&rgb, w as u32, h as u32, &cfg).unwrap();
            let heif = encode_rgb(&rgb, w as u32, h as u32, &cfg).unwrap();
            let from_raw = decode_266(&raw_stream).unwrap();
            let from_heif = decode(&heif).unwrap();
            assert_eq!(from_heif.planes, from_raw.planes, "heif vs raw {chroma:?}");
            assert_eq!(from_heif.planes, recon, "heif vs recon {chroma:?}");
        }
    }

    #[test]
    fn to_rgb_has_right_shape() {
        let (w, h) = (16usize, 16usize);
        let rgb = make("flat", w, h);
        let cfg = EncodeConfig::new().with_quality(90);
        let stream = crate::encode_rgb_266(&rgb, w as u32, h as u32, &cfg).unwrap();
        let out = decode_266(&stream).unwrap().to_rgb();
        assert_eq!(out.len(), w * h * 3);
        // A flat gray source stays near-neutral after the lossy round trip.
        assert!((out[0] as i32 - out[1] as i32).abs() <= 4);
    }
}
