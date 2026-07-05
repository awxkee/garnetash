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

#![allow(dead_code)]

use crate::cabac::{CabacEncoder, Contexts};
use crate::headers::{LOG2_CTU_SIZE, LOG2_MIN_CB_SIZE};
use crate::intra::PLANAR_IDX;

const LOG2_MIN_QT: u32 = LOG2_MIN_CB_SIZE;
pub(crate) const CTU_SIZE: u32 = 1 << LOG2_CTU_SIZE;

/// A coded leaf coding unit: top-left luma position and dimensions. With
/// quadtree-only partitioning `w == h`; MTT (binary/ternary) splits allow `w != h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Leaf {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) w: u32,
    pub(crate) h: u32,
}

/// Tracks the covering leaf-CU dimensions and intra mode per 4×4 luma cell, for
/// `split_cu_flag` context derivation and intra MPM-list derivation. Shared so
/// the encoder and verification decoder derive identical contexts and MPMs.
pub(crate) struct NeighborGrid {
    pub(crate) coded_w: u32,
    pub(crate) coded_h: u32,
    cells_x: usize,
    width_of: Vec<u16>,
    height_of: Vec<u16>,
    qtdepth_of: Vec<u8>,
    mode_of: Vec<u8>,
    coded: Vec<bool>,
}

impl NeighborGrid {
    pub(crate) fn new(coded_w: u32, coded_h: u32) -> Self {
        let cells_x = (coded_w as usize) / 4;
        let cells_y = (coded_h as usize) / 4;
        let n = cells_x * cells_y;
        NeighborGrid {
            coded_w,
            coded_h,
            cells_x,
            width_of: vec![0; n],
            height_of: vec![0; n],
            qtdepth_of: vec![0; n],
            mode_of: vec![PLANAR_IDX; n],
            coded: vec![false; n],
        }
    }

    #[inline]
    fn cell(&self, x: u32, y: u32) -> usize {
        (y as usize / 4) * self.cells_x + (x as usize / 4)
    }

    /// Record a finalised leaf CU's size over its covered cells.
    fn fill_size(&mut self, x: u32, y: u32, size: u32) {
        let mut cy = y;
        while cy < y + size && cy < self.coded_h {
            let mut cx = x;
            while cx < x + size && cx < self.coded_w {
                let idx = self.cell(cx, cy);
                self.width_of[idx] = size as u16;
                self.height_of[idx] = size as u16;
                self.coded[idx] = true;
                cx += 4;
            }
            cy += 4;
        }
    }

    /// Record a leaf CU's intra mode over its covered cells.
    pub(crate) fn set_mode(&mut self, x: u32, y: u32, size: u32, mode: u8) {
        let mut cy = y;
        while cy < y + size && cy < self.coded_h {
            let mut cx = x;
            while cx < x + size && cx < self.coded_w {
                let idx = self.cell(cx, cy);
                self.mode_of[idx] = mode;
                cx += 4;
            }
            cy += 4;
        }
    }

    /// `split_cu_flag` context (`DeriveCtx::CtxSplit`, quadtree-only path).
    fn ctx_split(&self, x: u32, y: u32, size: u32) -> usize {
        let mut ctx = 0usize;
        if x > 0 {
            let idx = self.cell(x - 1, y);
            if self.coded[idx] && (self.height_of[idx] as u32) < size {
                ctx += 1;
            }
        }
        if y > 0 {
            let idx = self.cell(x, y - 1);
            if self.coded[idx] && (self.width_of[idx] as u32) < size {
                ctx += 1;
            }
        }
        ctx
    }

    /// Intra mode recorded at luma sample `(x, y)` (the covering 4×4 cell).
    pub(crate) fn mode_at(&self, x: u32, y: u32) -> u8 {
        self.mode_of[self.cell(x.min(self.coded_w - 1), y.min(self.coded_h - 1))]
    }

    /// Dual-tree chroma derived mode (DM): the luma intra mode of the CU covering
    /// the centre of the co-located luma block. Matches VTM `getCoLocatedLumaPU`
    /// (`topLeft.offset(lumaW>>1, lumaH>>1)`); `(x, y, w, h)` is the chroma block
    /// in luma sample coordinates.
    pub(crate) fn chroma_dm(&self, x: u32, y: u32, w: u32, h: u32) -> u8 {
        self.mode_at(x + (w >> 1), y + (h >> 1))
    }

    /// Left-neighbour luma mode for MPM derivation: the CU covering the sample
    /// left of the block's bottom-left corner, available when `x > 0`.
    pub(crate) fn left_mode(&self, x: u32, y: u32, size: u32) -> Option<u8> {
        if x > 0 {
            Some(self.mode_of[self.cell(x - 1, y + size - 1)])
        } else {
            None
        }
    }

    /// Above-neighbour luma mode: the CU covering the sample above the block's
    /// top-right corner, available only when that sample is in the same CTU
    /// (i.e. the block is not at a CTU top boundary).
    pub(crate) fn above_mode(&self, x: u32, y: u32, size: u32) -> Option<u8> {
        if y & (CTU_SIZE - 1) != 0 {
            Some(self.mode_of[self.cell(x + size - 1, y - 1)])
        } else {
            None
        }
    }

    /// Record a finalised rectangular leaf CU's dimensions and quadtree depth.
    fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, qt_depth: u8) {
        let mut cy = y;
        while cy < y + h && cy < self.coded_h {
            let mut cx = x;
            while cx < x + w && cx < self.coded_w {
                let idx = self.cell(cx, cy);
                self.width_of[idx] = w as u16;
                self.height_of[idx] = h as u16;
                self.qtdepth_of[idx] = qt_depth;
                self.coded[idx] = true;
                cx += 4;
            }
            cy += 4;
        }
    }

    /// Record a rectangular leaf CU's intra mode over its covered cells.
    pub(crate) fn set_mode_rect(&mut self, x: u32, y: u32, w: u32, h: u32, mode: u8) {
        let mut cy = y;
        while cy < y + h && cy < self.coded_h {
            let mut cx = x;
            while cx < x + w && cx < self.coded_w {
                let idx = self.cell(cx, cy);
                self.mode_of[idx] = mode;
                cx += 4;
            }
            cy += 4;
        }
    }

    /// Left-neighbour luma mode for a rectangular block: the CU covering the
    /// sample left of the bottom-left corner `(x-1, y+h-1)`, available when `x>0`.
    pub(crate) fn left_mode_rect(&self, x: u32, y: u32, h: u32) -> Option<u8> {
        if x > 0 {
            Some(self.mode_of[self.cell(x - 1, y + h - 1)])
        } else {
            None
        }
    }

    /// Above-neighbour luma mode for a rectangular block: the CU covering
    /// `(x+w-1, y-1)`, available only when that sample is in the same CTU row.
    pub(crate) fn above_mode_rect(&self, x: u32, y: u32, w: u32) -> Option<u8> {
        if y & (CTU_SIZE - 1) != 0 {
            Some(self.mode_of[self.cell(x + w - 1, y - 1)])
        } else {
            None
        }
    }

    /// Neighbour CU dimensions/qtDepth at sample `(nx, ny)`, if coded.
    fn neigh(&self, nx: u32, ny: u32) -> Option<(u32, u32, u32)> {
        let idx = self.cell(nx, ny);
        if self.coded[idx] {
            Some((
                self.width_of[idx] as u32,
                self.height_of[idx] as u32,
                self.qtdepth_of[idx] as u32,
            ))
        } else {
            None
        }
    }

    /// Full QTBTT split-context derivation (`DeriveCtx::CtxSplit`), returning
    /// `(ctxSpl, ctxQt, ctxHv, ctxH12, ctxV12)`. `can` is the six-way
    /// availability `[no, qt, bh, bv, th, tv]`. Single slice/tile: a neighbour is
    /// available iff its covering cell is already coded.
    #[allow(clippy::too_many_arguments)]
    fn mtt_split_ctx(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        qt_depth: u32,
        mt_depth: u32,
        can: &[bool; 6],
    ) -> (usize, usize, usize, usize, usize) {
        let left = if x > 0 { self.neigh(x - 1, y) } else { None };
        let above = if y > 0 { self.neigh(x, y - 1) } else { None };

        // ctxSpl (split_cu_flag)
        let mut ctx_spl = 0usize;
        if let Some((_, hl, _)) = left
            && hl < h
        {
            ctx_spl += 1;
        }
        if let Some((wa, _, _)) = above
            && wa < w
        {
            ctx_spl += 1;
        }
        let mut num_split = 0usize;
        if can[1] {
            num_split += 2;
        }
        for &c in &can[2..6] {
            if c {
                num_split += 1;
            }
        }
        num_split = num_split.saturating_sub(1);
        ctx_spl += 3 * (num_split >> 1);

        // ctxQt (split_qt_flag)
        let mut ctx_qt = 0usize;
        if let Some((_, _, dl)) = left
            && dl > qt_depth
        {
            ctx_qt += 1;
        }
        if let Some((_, _, da)) = above
            && da > qt_depth
        {
            ctx_qt += 1;
        }
        if qt_depth >= 2 {
            ctx_qt += 3;
        }

        // ctxHv (mtt_split_cu_vertical_flag)
        let num_hor = can[2] as usize + can[4] as usize;
        let num_ver = can[3] as usize + can[5] as usize;
        let ctx_hv = if num_ver == num_hor {
            let w_above = above.map(|(wa, _, _)| wa).unwrap_or(1);
            let h_left = left.map(|(_, hl, _)| hl).unwrap_or(1);
            let dep_above = w / w_above;
            let dep_left = h / h_left;
            if dep_above == dep_left || left.is_none() || above.is_none() {
                0
            } else if dep_above < dep_left {
                1
            } else {
                2
            }
        } else if num_ver < num_hor {
            3
        } else {
            4
        };

        // ctxH12 / ctxV12 (mtt_split_cu_binary_flag)
        let ctx_h12 = if mt_depth <= 1 { 1 } else { 0 };
        let ctx_v12 = if mt_depth <= 1 { 3 } else { 2 };

        (ctx_spl, ctx_qt, ctx_hv, ctx_h12, ctx_v12)
    }
}

#[inline]
fn can_quad_split(log2size: u32) -> bool {
    log2size > LOG2_MIN_QT
}

/// The six QTBTT split modes (`None` = leaf). Binary splits halve one dimension;
/// ternary splits cut one dimension 1:2:1. "Vertical" cuts width, "horizontal"
/// cuts height, following H.266 naming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MttSplit {
    None,
    Quad,
    BinH,
    BinV,
    TriH,
    TriV,
}

/// Luma intra MTT constraints, mirroring the SPS-derived limits VTM uses in
/// `canSplit`. All sizes in luma samples; `MAX_TB` is the 64-sample transform cap.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MttCfg {
    pub(crate) max_mtt_depth: u32,
    pub(crate) min_qt_size: u32,
    pub(crate) max_bt_size: u32,
    pub(crate) max_tt_size: u32,
}

const MAX_TB: u32 = 64;
const MIN_CB: u32 = 1 << LOG2_MIN_CB_SIZE; // 4

/// Fixed luma-intra MTT configuration garnetash signals and enforces. Chosen so
/// the quadtree reaches 16×16 before binary/ternary splits take over (max BT/TT
/// size 32, MTT depth 2). The SPS diffs below are derived from these.
pub(crate) const MTT_MIN_QT_LOG2: u32 = 4; // 16
pub(crate) const MTT_MAX_DEPTH: u32 = 2;
pub(crate) const MTT_MAX_BT_LOG2: u32 = 5; // 32
pub(crate) const MTT_MAX_TT_LOG2: u32 = 5; // 32

#[inline]
pub(crate) fn mtt_cfg() -> MttCfg {
    MttCfg {
        max_mtt_depth: MTT_MAX_DEPTH,
        min_qt_size: 1 << MTT_MIN_QT_LOG2,
        max_bt_size: 1 << MTT_MAX_BT_LOG2,
        max_tt_size: 1 << MTT_MAX_TT_LOG2,
    }
}

/// SPS `ue(v)` field values for [`mtt_cfg`]: `(min_qt_diff, max_mtt_depth,
/// max_bt_diff, max_tt_diff)`, all relative as H.266 defines them.
pub(crate) fn mtt_sps_fields() -> (u32, u32, u32, u32) {
    (
        MTT_MIN_QT_LOG2 - LOG2_MIN_CB_SIZE,
        MTT_MAX_DEPTH,
        MTT_MAX_BT_LOG2 - MTT_MIN_QT_LOG2,
        MTT_MAX_TT_LOG2 - MTT_MIN_QT_LOG2,
    )
}

impl MttCfg {
    #[inline]
    fn min_bt_size(&self) -> u32 {
        MIN_CB
    }
    #[inline]
    fn min_tt_size(&self) -> u32 {
        MIN_CB
    }
}

/// Recursion state threaded through the coding tree, matching the depth counters
/// `canSplit`/`CtxSplit` consult.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeCtx {
    pub(crate) qt_depth: u32,
    pub(crate) mt_depth: u32,
    pub(crate) bt_depth: u32,
    pub(crate) implicit_bt_depth: u32,
    pub(crate) last_split: MttSplit,
    pub(crate) part_idx: u32,
}

impl NodeCtx {
    pub(crate) fn root() -> Self {
        NodeCtx {
            qt_depth: 0,
            mt_depth: 0,
            bt_depth: 0,
            implicit_bt_depth: 0,
            last_split: MttSplit::None, // CTU_LEVEL sentinel: QT allowed
            part_idx: 0,
        }
    }
}

/// Picture-boundary implicit split (`getImplicitSplit`, single tree). Returns the
/// forced split when the block crosses the coded picture edge, else `None`.
fn implicit_split(
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    nc: &NodeCtx,
    cfg: &MttCfg,
    coded_w: u32,
    coded_h: u32,
) -> MttSplit {
    // bottomLeft sample (x, y+h-1) and topRight sample (x+w-1, y) inside picture.
    let bl_in = x < coded_w && y + h - 1 < coded_h;
    let tr_in = x + w - 1 < coded_w && y < coded_h;
    if bl_in && tr_in {
        return MttSplit::None;
    }
    let bt_allowed = w <= cfg.max_bt_size
        && h <= cfg.max_bt_size
        && nc.mt_depth < (cfg.max_mtt_depth + nc.implicit_bt_depth);
    let qt_allowed = w > cfg.min_qt_size && h > cfg.min_qt_size && nc.bt_depth == 0;
    if !bl_in && !tr_in && qt_allowed {
        MttSplit::Quad
    } else if !bl_in && bt_allowed && w <= MAX_TB {
        MttSplit::BinH
    } else if !tr_in && bt_allowed && h <= MAX_TB {
        MttSplit::BinV
    } else {
        MttSplit::Quad
    }
}

/// QTBTT split availability (`QTBTPartitioner::canSplit`, single-tree intra
/// luma): `[canNo, canQt, canBh, canBv, canTh, canTv]`.
fn can_split(
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    nc: &NodeCtx,
    cfg: &MttCfg,
    coded_w: u32,
    coded_h: u32,
) -> [bool; 6] {
    let implicit = implicit_split(x, y, w, h, nc, cfg, coded_w, coded_h);

    let max_btd = cfg.max_mtt_depth + nc.implicit_bt_depth;
    let (max_bt, min_bt) = (cfg.max_bt_size, cfg.min_bt_size());
    let (max_tt, min_tt) = (cfg.max_tt_size, cfg.min_tt_size());

    let mut can_no = true;
    let mut can_qt = true;
    let mut can_bh = true;
    let mut can_bv = true;
    let mut can_th = true;
    let mut can_tv = true;
    let mut can_btt = nc.mt_depth < max_btd;

    // No QT once a BT/TT split has happened (last_split sentinel None == CTU_LEVEL).
    let last = nc.last_split;
    if last != MttSplit::None && last != MttSplit::Quad {
        can_qt = false;
    }
    if w <= cfg.min_qt_size {
        can_qt = false;
    }

    if implicit != MttSplit::None {
        can_no = false;
        can_th = false;
        can_tv = false;
        can_bh = implicit == MttSplit::BinH;
        can_bv = implicit == MttSplit::BinV;
        if !can_bh && !can_bv && !can_qt {
            can_qt = true;
        }
        return [can_no, can_qt, can_bh, can_bv, can_th, can_tv];
    }

    // Middle part of a ternary split forbids the parallel binary split:
    // parlSplit = TRIH→HORZ, TRIV→VERT.  canBh = parl≠HORZ, canBv = parl≠VERT.
    if (last == MttSplit::TriH || last == MttSplit::TriV) && nc.part_idx == 1 {
        let parl_is_h = last == MttSplit::TriH;
        can_bh = !parl_is_h;
        can_bv = parl_is_h;
    }

    if can_btt && (w <= min_bt && h <= min_bt) && (w <= min_tt && h <= min_tt) {
        can_btt = false;
    }
    if can_btt && (w > max_bt || h > max_bt) && (w > max_tt || h > max_tt) {
        can_btt = false;
    }
    if !can_btt {
        return [can_no, can_qt, false, false, false, false];
    }

    if w > max_bt || h > max_bt {
        can_bh = false;
        can_bv = false;
    }
    // BT horizontal (cuts height)
    if h <= min_bt {
        can_bh = false;
    }
    if w > MAX_TB && h <= MAX_TB {
        can_bh = false;
    }
    // BT vertical (cuts width)
    if w <= min_bt {
        can_bv = false;
    }
    if w <= MAX_TB && h > MAX_TB {
        can_bv = false;
    }
    // TT horizontal
    if h <= 2 * min_tt || h > max_tt || w > max_tt {
        can_th = false;
    }
    if w > MAX_TB || h > MAX_TB {
        can_th = false;
    }
    // TT vertical
    if w <= 2 * min_tt || w > max_tt || h > max_tt {
        can_tv = false;
    }
    if w > MAX_TB || h > MAX_TB {
        can_tv = false;
    }

    [can_no, can_qt, can_bh, can_bv, can_th, can_tv]
}

/// Code the coding-tree partitioning for the whole picture. `decide(x,y,size)`
/// chooses splits for fully-inside splittable nodes; `leaf(enc,ctx,grid,x,y,size)`
/// codes the coding-unit contents at each leaf (and should record its intra mode
/// via [`NeighborGrid::set_mode`] for MPM derivation). Returns the leaves in
/// coding order. Does not terminate the CABAC stream.
pub(crate) fn code_partitions<D, L>(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    coded_w: u32,
    coded_h: u32,
    mut decide: D,
    mut leaf: L,
) -> Vec<Leaf>
where
    D: FnMut(u32, u32, u32) -> bool,
    L: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32),
{
    let mut grid = NeighborGrid::new(coded_w, coded_h);
    let mut leaves = Vec::new();
    let mut cy = 0;
    while cy < coded_h {
        let mut cx = 0;
        while cx < coded_w {
            code_node(
                enc,
                ctx,
                &mut grid,
                cx,
                cy,
                LOG2_CTU_SIZE,
                &mut decide,
                &mut leaf,
                &mut leaves,
            );
            cx += CTU_SIZE;
        }
        cy += CTU_SIZE;
    }
    leaves
}

#[allow(clippy::too_many_arguments)]
fn code_node<D, L>(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    grid: &mut NeighborGrid,
    x: u32,
    y: u32,
    log2size: u32,
    decide: &mut D,
    leaf: &mut L,
    leaves: &mut Vec<Leaf>,
) where
    D: FnMut(u32, u32, u32) -> bool,
    L: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32),
{
    let size = 1u32 << log2size;
    if x >= grid.coded_w || y >= grid.coded_h {
        return;
    }
    let inside = (x + size <= grid.coded_w) && (y + size <= grid.coded_h);
    let can_qt = can_quad_split(log2size);

    let split = if !inside {
        debug_assert!(can_qt, "boundary block too small to split implicitly");
        true
    } else if can_qt {
        let do_split = decide(x, y, size);
        let c = grid.ctx_split(x, y, size);
        enc.encode_bin(do_split as u8, &mut ctx.split_flag[c]);
        do_split
    } else {
        false
    };

    if split {
        let half = size >> 1;
        let l2 = log2size - 1;
        code_node(enc, ctx, grid, x, y, l2, decide, leaf, leaves);
        code_node(enc, ctx, grid, x + half, y, l2, decide, leaf, leaves);
        code_node(enc, ctx, grid, x, y + half, l2, decide, leaf, leaves);
        code_node(enc, ctx, grid, x + half, y + half, l2, decide, leaf, leaves);
    } else {
        leaf(enc, ctx, grid, x, y, size);
        grid.fill_size(x, y, size);
        leaves.push(Leaf {
            x,
            y,
            w: size,
            h: size,
        });
    }
}

/// Child rectangles of a node under a given split (H.266 geometry: binary halves
/// one dimension, ternary cuts 1:2:1).
fn split_children(x: u32, y: u32, w: u32, h: u32, split: MttSplit) -> Vec<(u32, u32, u32, u32)> {
    match split {
        MttSplit::None => vec![(x, y, w, h)],
        MttSplit::Quad => {
            let (hw, hh) = (w >> 1, h >> 1);
            vec![
                (x, y, hw, hh),
                (x + hw, y, hw, hh),
                (x, y + hh, hw, hh),
                (x + hw, y + hh, hw, hh),
            ]
        }
        MttSplit::BinH => {
            let hh = h >> 1;
            vec![(x, y, w, hh), (x, y + hh, w, hh)]
        }
        MttSplit::BinV => {
            let hw = w >> 1;
            vec![(x, y, hw, h), (x + hw, y, hw, h)]
        }
        MttSplit::TriH => {
            let q = h >> 2;
            vec![(x, y, w, q), (x, y + q, w, h - 2 * q), (x, y + h - q, w, q)]
        }
        MttSplit::TriV => {
            let q = w >> 2;
            vec![(x, y, q, h), (x + q, y, w - 2 * q, h), (x + w - q, y, q, h)]
        }
    }
}

/// Child recursion state after applying `split` (mirrors `splitCurrArea` depth
/// bookkeeping). `part_idx` is the child's index; `implicit` marks a boundary-
/// forced split (its binary children deepen `implicit_bt_depth`).
fn child_ctx(nc: &NodeCtx, split: MttSplit, part_idx: u32, implicit: bool) -> NodeCtx {
    let mut c = *nc;
    c.part_idx = part_idx;
    c.last_split = split;
    match split {
        MttSplit::Quad => {
            c.qt_depth += 1;
            c.mt_depth = 0;
            c.bt_depth = 0;
        }
        MttSplit::BinH | MttSplit::BinV => {
            c.mt_depth += 1;
            c.bt_depth += 1;
            if implicit {
                c.implicit_bt_depth += 1;
            }
        }
        MttSplit::TriH | MttSplit::TriV => {
            c.mt_depth += 1;
            c.bt_depth += 2;
        }
        MttSplit::None => {}
    }
    c
}

/// Emit the `split_cu_mode` syntax for `split` at a node, exactly as
/// `CABACWriter::split_cu_mode` (the bins coded depend on `can`).
fn encode_split_mode(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    split: MttSplit,
    can: &[bool; 6],
    ctxs: (usize, usize, usize, usize, usize),
) {
    let (ctx_spl, ctx_qt, ctx_hv, ctx_h12, ctx_v12) = ctxs;
    let can_split = can[1] || can[2] || can[3] || can[4] || can[5];
    let is_no = split == MttSplit::None;
    if can[0] && can_split {
        enc.encode_bin(!is_no as u8, &mut ctx.split_flag[ctx_spl]);
    }
    if is_no {
        return;
    }
    let can_btt = can[2] || can[3] || can[4] || can[5];
    let is_qt = split == MttSplit::Quad;
    if can[1] && can_btt {
        enc.encode_bin(is_qt as u8, &mut ctx.split_qt_flag[ctx_qt]);
    }
    if is_qt {
        return;
    }
    let can_hor = can[2] || can[4];
    let can_ver = can[3] || can[5];
    let is_ver = split == MttSplit::BinV || split == MttSplit::TriV;
    if can_ver && can_hor {
        enc.encode_bin(is_ver as u8, &mut ctx.mtt_split_vertical[ctx_hv]);
    }
    let can14 = if is_ver { can[5] } else { can[4] };
    let can12 = if is_ver { can[3] } else { can[2] };
    let is12 = if is_ver {
        split == MttSplit::BinV
    } else {
        split == MttSplit::BinH
    };
    if can12 && can14 {
        enc.encode_bin(
            is12 as u8,
            &mut ctx.mtt_split_binary[if is_ver { ctx_v12 } else { ctx_h12 }],
        );
    }
}

/// Code MTT (QTBTT) coding-tree partitioning for the whole picture. `decide`
/// chooses an allowed split per node (must return a mode that is `true` in the
/// supplied `can`); `leaf` codes each leaf CU's contents over its `w×h` area.
pub(crate) fn code_partitions_mtt<D, L>(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    coded_w: u32,
    coded_h: u32,
    cfg: MttCfg,
    mut decide: D,
    mut leaf: L,
) -> Vec<Leaf>
where
    D: FnMut(u32, u32, u32, u32, &[bool; 6]) -> MttSplit,
    L: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
{
    let mut grid = NeighborGrid::new(coded_w, coded_h);
    let mut leaves = Vec::new();
    let mut cy = 0;
    while cy < coded_h {
        let mut cx = 0;
        while cx < coded_w {
            code_node_mtt(
                enc,
                ctx,
                &mut grid,
                cx,
                cy,
                CTU_SIZE,
                CTU_SIZE,
                &NodeCtx::root(),
                &cfg,
                &mut decide,
                &mut leaf,
                &mut leaves,
            );
            cx += CTU_SIZE;
        }
        cy += CTU_SIZE;
    }
    leaves
}

#[allow(clippy::too_many_arguments)]
fn code_node_mtt<D, L>(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    grid: &mut NeighborGrid,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    nc: &NodeCtx,
    cfg: &MttCfg,
    decide: &mut D,
    leaf: &mut L,
    leaves: &mut Vec<Leaf>,
) where
    D: FnMut(u32, u32, u32, u32, &[bool; 6]) -> MttSplit,
    L: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
{
    if x >= grid.coded_w || y >= grid.coded_h {
        return;
    }
    let can = can_split(x, y, w, h, nc, cfg, grid.coded_w, grid.coded_h);
    let imp = implicit_split(x, y, w, h, nc, cfg, grid.coded_w, grid.coded_h);
    let split = decide(x, y, w, h, &can);
    let ctxs = grid.mtt_split_ctx(x, y, w, h, nc.qt_depth, nc.mt_depth, &can);
    encode_split_mode(enc, ctx, split, &can, ctxs);

    if split == MttSplit::None {
        leaf(enc, ctx, grid, x, y, w, h);
        grid.fill_rect(x, y, w, h, nc.qt_depth as u8);
        leaves.push(Leaf { x, y, w, h });
        return;
    }
    let implicit = imp != MttSplit::None;
    for (i, (cx, cy, cw, ch)) in split_children(x, y, w, h, split).into_iter().enumerate() {
        let cn = child_ctx(nc, split, i as u32, implicit);
        code_node_mtt(
            enc, ctx, grid, cx, cy, cw, ch, &cn, cfg, decide, leaf, leaves,
        );
    }
}

/// Code dual-tree (separate luma/chroma) coding-tree partitioning for an intra
/// slice. Matches VTM's interleaving for CTU > 64: the shared quadtree is forced
/// down to 64×64 (no coded bin at the CTU root, since no-split is disallowed at
/// 128), then within each 64×64 region the full luma subtree is coded, followed
/// by the full chroma subtree, each against its own neighbour grid. `chroma_cfg`
/// carries the chroma tree's split limits; `chroma_decide` chooses chroma splits.
/// Returns `(luma_leaves, chroma_leaves)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_partitions_dual<DL, LL, DC, LC>(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    coded_w: u32,
    coded_h: u32,
    cfg: MttCfg,
    chroma_cfg: MttCfg,
    mut luma_decide: DL,
    mut luma_leaf: LL,
    mut chroma_decide: DC,
    mut chroma_leaf: LC,
) -> (Vec<Leaf>, Vec<Leaf>)
where
    DL: FnMut(u32, u32, u32, u32, &[bool; 6]) -> MttSplit,
    LL: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
    DC: FnMut(u32, u32, u32, u32, &[bool; 6]) -> MttSplit,
    LC: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
{
    let mut luma_grid = NeighborGrid::new(coded_w, coded_h);
    let mut chroma_grid = NeighborGrid::new(coded_w, coded_h);
    let mut luma_leaves = Vec::new();
    let mut chroma_leaves = Vec::new();
    let mut cy = 0;
    while cy < coded_h {
        let mut cx = 0;
        while cx < coded_w {
            code_node_dual(
                enc,
                ctx,
                &mut luma_grid,
                &mut chroma_grid,
                cx,
                cy,
                CTU_SIZE,
                CTU_SIZE,
                &NodeCtx::root(),
                &cfg,
                &chroma_cfg,
                &mut luma_decide,
                &mut luma_leaf,
                &mut chroma_decide,
                &mut chroma_leaf,
                &mut luma_leaves,
                &mut chroma_leaves,
            );
            cx += CTU_SIZE;
        }
        cy += CTU_SIZE;
    }
    (luma_leaves, chroma_leaves)
}

#[allow(clippy::too_many_arguments)]
fn code_node_dual<DL, LL, DC, LC>(
    enc: &mut CabacEncoder,
    ctx: &mut Contexts,
    luma_grid: &mut NeighborGrid,
    chroma_grid: &mut NeighborGrid,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    nc: &NodeCtx,
    cfg: &MttCfg,
    chroma_cfg: &MttCfg,
    luma_decide: &mut DL,
    luma_leaf: &mut LL,
    chroma_decide: &mut DC,
    chroma_leaf: &mut LC,
    luma_leaves: &mut Vec<Leaf>,
    chroma_leaves: &mut Vec<Leaf>,
) where
    DL: FnMut(u32, u32, u32, u32, &[bool; 6]) -> MttSplit,
    LL: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
    DC: FnMut(u32, u32, u32, u32, &[bool; 6]) -> MttSplit,
    LC: FnMut(&mut CabacEncoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
{
    if x >= luma_grid.coded_w || y >= luma_grid.coded_h {
        return;
    }
    if w > 64 || h > 64 {
        // Shared quadtree level (CTU > 64). In an intra dual tree the luma/chroma
        // split happens at 64×64, so the 128→64 quadtree is *always forced* and
        // coded with no bin — both at picture boundaries and in the interior
        // (where a 128 dual-tree CU is disallowed). VTM consumes zero bins here;
        // emitting a split_cu_flag (as a normal interior node would) desyncs the
        // bitstream. So we recurse directly without encode_split_mode.
        let split = MttSplit::Quad;
        for (i, (sx, sy, sw, sh)) in split_children(x, y, w, h, split).into_iter().enumerate() {
            let cn = child_ctx(nc, split, i as u32, true);
            code_node_dual(
                enc,
                ctx,
                luma_grid,
                chroma_grid,
                sx,
                sy,
                sw,
                sh,
                &cn,
                cfg,
                chroma_cfg,
                luma_decide,
                luma_leaf,
                chroma_decide,
                chroma_leaf,
                luma_leaves,
                chroma_leaves,
            );
        }
        return;
    }
    // 64×64 region: code the full luma subtree, then the full chroma subtree,
    // each against its own neighbour grid (independent contexts below 64×64).
    code_node_mtt(
        enc,
        ctx,
        luma_grid,
        x,
        y,
        w,
        h,
        nc,
        cfg,
        luma_decide,
        luma_leaf,
        luma_leaves,
    );
    code_node_mtt(
        enc,
        ctx,
        chroma_grid,
        x,
        y,
        w,
        h,
        nc,
        chroma_cfg,
        chroma_decide,
        chroma_leaf,
        chroma_leaves,
    );
}

/// Test-only partitioning decoder, mirroring [`code_partitions`].
pub(crate) mod test_support {
    use super::*;
    use crate::cabac::engine::CabacDecoder;

    pub(crate) fn decode_partitions<L>(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        coded_w: u32,
        coded_h: u32,
        mut dleaf: L,
    ) -> Vec<Leaf>
    where
        L: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32),
    {
        let mut grid = NeighborGrid::new(coded_w, coded_h);
        let mut leaves = Vec::new();
        let mut cy = 0;
        while cy < coded_h {
            let mut cx = 0;
            while cx < coded_w {
                decode_node(
                    dec,
                    ctx,
                    &mut grid,
                    cx,
                    cy,
                    LOG2_CTU_SIZE,
                    &mut dleaf,
                    &mut leaves,
                );
                cx += CTU_SIZE;
            }
            cy += CTU_SIZE;
        }
        leaves
    }

    fn decode_node<L>(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        grid: &mut NeighborGrid,
        x: u32,
        y: u32,
        log2size: u32,
        dleaf: &mut L,
        leaves: &mut Vec<Leaf>,
    ) where
        L: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32),
    {
        let size = 1u32 << log2size;
        if x >= grid.coded_w || y >= grid.coded_h {
            return;
        }
        let inside = (x + size <= grid.coded_w) && (y + size <= grid.coded_h);
        let can_qt = can_quad_split(log2size);
        let split = if !inside {
            true
        } else if can_qt {
            let c = grid.ctx_split(x, y, size);
            dec.decode_bin(&mut ctx.split_flag[c]) != 0
        } else {
            false
        };
        if split {
            let half = size >> 1;
            let l2 = log2size - 1;
            decode_node(dec, ctx, grid, x, y, l2, dleaf, leaves);
            decode_node(dec, ctx, grid, x + half, y, l2, dleaf, leaves);
            decode_node(dec, ctx, grid, x, y + half, l2, dleaf, leaves);
            decode_node(dec, ctx, grid, x + half, y + half, l2, dleaf, leaves);
        } else {
            dleaf(dec, ctx, grid, x, y, size);
            grid.fill_size(x, y, size);
            leaves.push(Leaf {
                x,
                y,
                w: size,
                h: size,
            });
        }
    }

    /// Decode `split_cu_mode`, inverting [`encode_split_mode`]. Bins are read only
    /// where the encoder coded them; forced cases derive the value from `can`.
    fn decode_split_mode(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        can: &[bool; 6],
        ctxs: (usize, usize, usize, usize, usize),
    ) -> MttSplit {
        let (ctx_spl, ctx_qt, ctx_hv, ctx_h12, ctx_v12) = ctxs;
        let can_split = can[1] || can[2] || can[3] || can[4] || can[5];
        let do_split = if can[0] && can_split {
            dec.decode_bin(&mut ctx.split_flag[ctx_spl]) != 0
        } else {
            can_split
        };
        if !do_split {
            return MttSplit::None;
        }
        let can_btt = can[2] || can[3] || can[4] || can[5];
        let is_qt = if can[1] && can_btt {
            dec.decode_bin(&mut ctx.split_qt_flag[ctx_qt]) != 0
        } else {
            can[1]
        };
        if is_qt {
            return MttSplit::Quad;
        }
        let can_hor = can[2] || can[4];
        let can_ver = can[3] || can[5];
        let is_ver = if can_ver && can_hor {
            dec.decode_bin(&mut ctx.mtt_split_vertical[ctx_hv]) != 0
        } else {
            can_ver
        };
        let can14 = if is_ver { can[5] } else { can[4] };
        let can12 = if is_ver { can[3] } else { can[2] };
        let is12 = if can12 && can14 {
            dec.decode_bin(&mut ctx.mtt_split_binary[if is_ver { ctx_v12 } else { ctx_h12 }]) != 0
        } else {
            can12
        };
        match (is_ver, is12) {
            (true, true) => MttSplit::BinV,
            (true, false) => MttSplit::TriV,
            (false, true) => MttSplit::BinH,
            (false, false) => MttSplit::TriH,
        }
    }

    /// MTT (QTBTT) partitioning decoder, mirroring [`super::code_partitions_mtt`].
    pub(crate) fn decode_partitions_mtt<L>(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        coded_w: u32,
        coded_h: u32,
        cfg: MttCfg,
        mut dleaf: L,
    ) -> Vec<Leaf>
    where
        L: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
    {
        let mut grid = NeighborGrid::new(coded_w, coded_h);
        let mut leaves = Vec::new();
        let mut cy = 0;
        while cy < coded_h {
            let mut cx = 0;
            while cx < coded_w {
                decode_node_mtt(
                    dec,
                    ctx,
                    &mut grid,
                    cx,
                    cy,
                    CTU_SIZE,
                    CTU_SIZE,
                    &NodeCtx::root(),
                    &cfg,
                    &mut dleaf,
                    &mut leaves,
                );
                cx += CTU_SIZE;
            }
            cy += CTU_SIZE;
        }
        leaves
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_node_mtt<L>(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        grid: &mut NeighborGrid,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        nc: &NodeCtx,
        cfg: &MttCfg,
        dleaf: &mut L,
        leaves: &mut Vec<Leaf>,
    ) where
        L: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
    {
        if x >= grid.coded_w || y >= grid.coded_h {
            return;
        }
        let can = can_split(x, y, w, h, nc, cfg, grid.coded_w, grid.coded_h);
        let imp = implicit_split(x, y, w, h, nc, cfg, grid.coded_w, grid.coded_h);
        let ctxs = grid.mtt_split_ctx(x, y, w, h, nc.qt_depth, nc.mt_depth, &can);
        let split = decode_split_mode(dec, ctx, &can, ctxs);

        if split == MttSplit::None {
            dleaf(dec, ctx, grid, x, y, w, h);
            grid.fill_rect(x, y, w, h, nc.qt_depth as u8);
            leaves.push(Leaf { x, y, w, h });
            return;
        }
        let implicit = imp != MttSplit::None;
        for (i, (cx, cy, cw, ch)) in split_children(x, y, w, h, split).into_iter().enumerate() {
            let cn = child_ctx(nc, split, i as u32, implicit);
            decode_node_mtt(dec, ctx, grid, cx, cy, cw, ch, &cn, cfg, dleaf, leaves);
        }
    }

    /// Decoder mirror of [`super::code_partitions_dual`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_partitions_dual<LL, LC>(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        coded_w: u32,
        coded_h: u32,
        cfg: MttCfg,
        chroma_cfg: MttCfg,
        mut luma_leaf: LL,
        mut chroma_leaf: LC,
    ) -> (Vec<Leaf>, Vec<Leaf>)
    where
        LL: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
        LC: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
    {
        let mut luma_grid = NeighborGrid::new(coded_w, coded_h);
        let mut chroma_grid = NeighborGrid::new(coded_w, coded_h);
        let mut luma_leaves = Vec::new();
        let mut chroma_leaves = Vec::new();
        let mut cy = 0;
        while cy < coded_h {
            let mut cx = 0;
            while cx < coded_w {
                decode_node_dual(
                    dec,
                    ctx,
                    &mut luma_grid,
                    &mut chroma_grid,
                    cx,
                    cy,
                    CTU_SIZE,
                    CTU_SIZE,
                    &NodeCtx::root(),
                    &cfg,
                    &chroma_cfg,
                    &mut luma_leaf,
                    &mut chroma_leaf,
                    &mut luma_leaves,
                    &mut chroma_leaves,
                );
                cx += CTU_SIZE;
            }
            cy += CTU_SIZE;
        }
        (luma_leaves, chroma_leaves)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_node_dual<LL, LC>(
        dec: &mut CabacDecoder,
        ctx: &mut Contexts,
        luma_grid: &mut NeighborGrid,
        chroma_grid: &mut NeighborGrid,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        nc: &NodeCtx,
        cfg: &MttCfg,
        chroma_cfg: &MttCfg,
        luma_leaf: &mut LL,
        chroma_leaf: &mut LC,
        luma_leaves: &mut Vec<Leaf>,
        chroma_leaves: &mut Vec<Leaf>,
    ) where
        LL: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
        LC: FnMut(&mut CabacDecoder, &mut Contexts, &mut NeighborGrid, u32, u32, u32, u32),
    {
        if x >= luma_grid.coded_w || y >= luma_grid.coded_h {
            return;
        }
        if w > 64 || h > 64 {
            // Forced 128→64 quadtree (no coded bin), mirroring code_node_dual.
            let split = MttSplit::Quad;
            for (i, (sx, sy, sw, sh)) in split_children(x, y, w, h, split).into_iter().enumerate() {
                let cn = child_ctx(nc, split, i as u32, true);
                decode_node_dual(
                    dec,
                    ctx,
                    luma_grid,
                    chroma_grid,
                    sx,
                    sy,
                    sw,
                    sh,
                    &cn,
                    cfg,
                    chroma_cfg,
                    luma_leaf,
                    chroma_leaf,
                    luma_leaves,
                    chroma_leaves,
                );
            }
            return;
        }
        decode_node_mtt(
            dec,
            ctx,
            luma_grid,
            x,
            y,
            w,
            h,
            nc,
            cfg,
            luma_leaf,
            luma_leaves,
        );
        decode_node_mtt(
            dec,
            ctx,
            chroma_grid,
            x,
            y,
            w,
            h,
            nc,
            chroma_cfg,
            chroma_leaf,
            chroma_leaves,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::cabac::engine::CabacDecoder;
    use crate::intra::{build_mpm, decode_luma_mode, encode_luma_mode};

    fn noop_leaf(
        _: &mut CabacEncoder,
        _: &mut Contexts,
        _: &mut NeighborGrid,
        _: u32,
        _: u32,
        _: u32,
    ) {
    }

    fn split_roundtrip(
        coded_w: u32,
        coded_h: u32,
        qp: u8,
        mut decide: impl FnMut(u32, u32, u32) -> bool,
    ) -> Vec<Leaf> {
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(qp);
        let enc_leaves = code_partitions(
            &mut enc,
            &mut ectx,
            coded_w,
            coded_h,
            &mut decide,
            noop_leaf,
        );
        enc.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(qp);
        let dec_leaves =
            decode_partitions(&mut dec, &mut dctx, coded_w, coded_h, |_, _, _, _, _, _| {});
        assert_eq!(dec.decode_terminate(), 1);
        assert_eq!(enc_leaves, dec_leaves);
        enc_leaves
    }

    #[test]
    fn leaves_tile_the_picture_exactly() {
        for (w, h) in [(32, 32), (64, 64), (96, 64), (32, 96), (128, 128), (40, 40)] {
            let leaves = split_roundtrip(w, h, 32, |_, _, _| false);
            let area: u64 = leaves.iter().map(|l| (l.w as u64) * (l.h as u64)).sum();
            assert_eq!(area, (w as u64) * (h as u64), "{w}x{h}");
            assert!(leaves.iter().all(|l| l.x + l.w <= w && l.y + l.h <= h));
        }
    }

    #[test]
    fn split_policies_round_trip() {
        split_roundtrip(64, 64, 32, |_, _, s| s > 8);
        split_roundtrip(32, 32, 28, |_, _, s| s > 4);
        for qp in [10u8, 32, 51] {
            split_roundtrip(128, 96, qp, |x, y, s| s > 4 && ((x + y) / s) % 3 != 0);
        }
    }

    /// Full partition + luma intra-mode round trip: the heart of this increment.
    fn intra_roundtrip(
        coded_w: u32,
        coded_h: u32,
        qp: u8,
        mut decide: impl FnMut(u32, u32, u32) -> bool,
        mut pick_mode: impl FnMut(u32, u32, u32) -> u8,
    ) -> Vec<(Leaf, u8)> {
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(qp);
        let mut enc_modes: Vec<(Leaf, u8)> = Vec::new();
        {
            let enc_leaf = |e: &mut CabacEncoder,
                            c: &mut Contexts,
                            g: &mut NeighborGrid,
                            x: u32,
                            y: u32,
                            s: u32| {
                let mpm = build_mpm(g.left_mode(x, y, s), g.above_mode(x, y, s));
                let mode = pick_mode(x, y, s);
                encode_luma_mode(e, c, &mpm, mode);
                g.set_mode(x, y, s, mode);
                enc_modes.push((Leaf { x, y, w: s, h: s }, mode));
            };
            code_partitions(&mut enc, &mut ectx, coded_w, coded_h, &mut decide, enc_leaf);
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(qp);
        let mut dec_modes: Vec<(Leaf, u8)> = Vec::new();
        {
            let dec_leaf = |d: &mut CabacDecoder,
                            c: &mut Contexts,
                            g: &mut NeighborGrid,
                            x: u32,
                            y: u32,
                            s: u32| {
                let mpm = build_mpm(g.left_mode(x, y, s), g.above_mode(x, y, s));
                let mode = decode_luma_mode(d, c, &mpm);
                g.set_mode(x, y, s, mode);
                dec_modes.push((Leaf { x, y, w: s, h: s }, mode));
            };
            decode_partitions(&mut dec, &mut dctx, coded_w, coded_h, dec_leaf);
        }
        assert_eq!(dec.decode_terminate(), 1, "terminate mismatch");
        assert_eq!(enc_modes, dec_modes, "partition+mode mismatch");
        enc_modes
    }

    #[test]
    fn intra_modes_round_trip_uniform() {
        // Every mode value, all CUs the same mode (exercises MPM hits/misses).
        for mode in 0u8..67 {
            intra_roundtrip(64, 64, 32, |_, _, s| s > 16, |_, _, _| mode);
        }
    }

    #[test]
    fn intra_modes_round_trip_varied() {
        // Position-dependent modes + non-uniform splits stress MPM derivation
        // across many neighbour combinations.
        for qp in [10u8, 32, 51] {
            let got = intra_roundtrip(
                128,
                96,
                qp,
                |x, y, s| s > 4 && ((x ^ y) / s) % 2 == 0,
                |x, y, s| (((x / s) * 7 + (y / s) * 13 + s) % 67) as u8,
            );
            assert!(!got.is_empty());
        }
    }

    #[test]
    fn intra_modes_round_trip_boundary() {
        intra_roundtrip(40, 72, 30, |_, _, s| s > 8, |x, y, _| ((x + y) % 67) as u8);
    }

    // ── MTT (QTBTT) partition syntax ───────────────────────────────────────
    fn allowed(s: MttSplit, can: &[bool; 6]) -> bool {
        match s {
            MttSplit::None => can[0],
            MttSplit::Quad => can[1],
            MttSplit::BinH => can[2],
            MttSplit::BinV => can[3],
            MttSplit::TriH => can[4],
            MttSplit::TriV => can[5],
        }
    }

    /// Deterministic decision exercising QT + BT + TT: prefers a position-keyed
    /// order, always returns a split that is allowed in `can`, and stops at small
    /// blocks. Never returns `None` at a forced (boundary) node.
    fn mtt_decide(x: u32, y: u32, w: u32, h: u32, can: &[bool; 6]) -> MttSplit {
        let small = w <= 8 && h <= 8;
        if small && can[0] {
            return MttSplit::None;
        }
        let pat = ((x / 16 + y / 16) % 4) as usize;
        let order = match pat {
            0 => [
                MttSplit::Quad,
                MttSplit::BinH,
                MttSplit::BinV,
                MttSplit::TriH,
                MttSplit::TriV,
            ],
            1 => [
                MttSplit::BinV,
                MttSplit::BinH,
                MttSplit::Quad,
                MttSplit::TriV,
                MttSplit::TriH,
            ],
            2 => [
                MttSplit::TriH,
                MttSplit::BinV,
                MttSplit::Quad,
                MttSplit::BinH,
                MttSplit::TriV,
            ],
            _ => [
                MttSplit::BinH,
                MttSplit::TriV,
                MttSplit::Quad,
                MttSplit::BinV,
                MttSplit::TriH,
            ],
        };
        for s in order {
            if allowed(s, can) {
                return s;
            }
        }
        if can[0] {
            MttSplit::None
        } else {
            // Forced node: take any legal split.
            for s in [
                MttSplit::Quad,
                MttSplit::BinH,
                MttSplit::BinV,
                MttSplit::TriH,
                MttSplit::TriV,
            ] {
                if allowed(s, can) {
                    return s;
                }
            }
            MttSplit::None
        }
    }

    fn mtt_roundtrip(coded_w: u32, coded_h: u32, qp: u8, cfg: MttCfg) -> Vec<Leaf> {
        let mut enc = CabacEncoder::new();
        let mut ectx = Contexts::new_intra(qp);
        let enc_leaves = code_partitions_mtt(
            &mut enc,
            &mut ectx,
            coded_w,
            coded_h,
            cfg,
            |x, y, w, h, can| mtt_decide(x, y, w, h, can),
            |_, _, _, _, _, _, _| {},
        );
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx = Contexts::new_intra(qp);
        let dec_leaves = decode_partitions_mtt(
            &mut dec,
            &mut dctx,
            coded_w,
            coded_h,
            cfg,
            |_, _, _, _, _, _, _| {},
        );
        assert_eq!(dec.decode_terminate(), 1, "terminate mismatch");
        assert_eq!(
            enc_leaves, dec_leaves,
            "MTT partition enc/dec mismatch {coded_w}x{coded_h}"
        );
        enc_leaves
    }

    #[test]
    fn mtt_partition_round_trips_and_tiles() {
        let cfg = MttCfg {
            max_mtt_depth: 3,
            min_qt_size: 16,
            max_bt_size: 64,
            max_tt_size: 64,
        };
        // CTU-multiple sizes isolate the interior QTBTT syntax (no implicit splits).
        for (w, h) in [(128, 128), (256, 128), (128, 256)] {
            let leaves = mtt_roundtrip(w, h, 32, cfg);
            let area: u64 = leaves.iter().map(|l| (l.w as u64) * (l.h as u64)).sum();
            assert_eq!(area, (w as u64) * (h as u64), "tiling {w}x{h}");
            // At least one genuinely non-square leaf must appear (BT/TT happened).
            assert!(
                leaves.iter().any(|l| l.w != l.h),
                "no rectangular leaves at {w}x{h}"
            );
        }
    }

    #[test]
    fn mtt_partition_round_trips_at_qps() {
        let cfg = MttCfg {
            max_mtt_depth: 2,
            min_qt_size: 16,
            max_bt_size: 64,
            max_tt_size: 32,
        };
        for qp in [10u8, 32, 51] {
            mtt_roundtrip(128, 128, qp, cfg);
        }
    }

    #[test]
    fn mtt_partition_round_trips_with_boundary() {
        // Non-CTU-multiple dimensions exercise the implicit boundary splits.
        let cfg = MttCfg {
            max_mtt_depth: 3,
            min_qt_size: 16,
            max_bt_size: 64,
            max_tt_size: 64,
        };
        for (w, h) in [(200, 136), (192, 128), (160, 160)] {
            let leaves = mtt_roundtrip(w, h, 30, cfg);
            let area: u64 = leaves.iter().map(|l| (l.w as u64) * (l.h as u64)).sum();
            assert_eq!(area, (w as u64) * (h as u64), "boundary tiling {w}x{h}");
        }
    }

    // ── Dual-tree (separate luma/chroma) partition structure ───────────────
    /// Chroma decision for the minimal milestone: never split below 64×64 (one
    /// chroma CU per 64×64 region), but honour forced boundary splits.
    fn chroma_no_split(_x: u32, _y: u32, _w: u32, _h: u32, can: &[bool; 6]) -> MttSplit {
        if can[0] {
            return MttSplit::None;
        }
        for s in [
            MttSplit::Quad,
            MttSplit::BinH,
            MttSplit::BinV,
            MttSplit::TriH,
            MttSplit::TriV,
        ] {
            if allowed(s, can) {
                return s;
            }
        }
        MttSplit::None
    }

    #[test]
    fn chroma_dm_reads_colocated_luma_centre() {
        // Build a luma-mode grid with four quadrant modes over a 64×64 region,
        // then confirm chroma_dm reads the luma mode at each chroma block's
        // centre (VTM getCoLocatedLumaPU: topLeft.offset(w>>1, h>>1)).
        let mut g = NeighborGrid::new(64, 64);
        g.set_mode_rect(0, 0, 32, 32, 10);
        g.set_mode_rect(32, 0, 32, 32, 20);
        g.set_mode_rect(0, 32, 32, 32, 30);
        g.set_mode_rect(32, 32, 32, 32, 40);
        // A 64×64 chroma block (luma coords): centre (32,32) → quadrant with 40.
        assert_eq!(g.chroma_dm(0, 0, 64, 64), 40);
        // Chroma sub-blocks landing in each quadrant.
        assert_eq!(g.chroma_dm(0, 0, 16, 16), 10); // centre (8,8)
        assert_eq!(g.chroma_dm(32, 0, 16, 16), 20); // centre (40,8)
        assert_eq!(g.chroma_dm(0, 32, 16, 16), 30); // centre (8,40)
        assert_eq!(g.chroma_dm(48, 48, 16, 16), 40); // centre (56,56)
    }

    #[test]
    fn dual_tree_partition_round_trips() {
        // The luma tree partitions freely; the chroma tree stays one CU per
        // 64×64. Encoder and decoder must reconstruct identical luma AND chroma
        // leaf sequences from the interleaved (shared-QT-to-64, then luma-then-
        // chroma) bitstream, using two independent neighbour grids.
        let cfg = MttCfg {
            max_mtt_depth: 3,
            min_qt_size: 16,
            max_bt_size: 64,
            max_tt_size: 64,
        };
        let cc = MttCfg {
            max_mtt_depth: 0,
            min_qt_size: 64,
            max_bt_size: 64,
            max_tt_size: 64,
        };
        for (w, h) in [(128u32, 128u32), (256, 128), (128, 256)] {
            let mut enc = CabacEncoder::new();
            let mut ectx = Contexts::new_intra(32);
            let (ll_e, cl_e) = code_partitions_dual(
                &mut enc,
                &mut ectx,
                w,
                h,
                cfg,
                cc,
                |x, y, bw, bh, can| mtt_decide(x, y, bw, bh, can),
                |_, _, _, _, _, _, _| {},
                |x, y, bw, bh, can| chroma_no_split(x, y, bw, bh, can),
                |_, _, _, _, _, _, _| {},
            );
            enc.encode_terminate(1);
            let bytes = enc.finish();

            let mut dec = CabacDecoder::new(&bytes);
            let mut dctx = Contexts::new_intra(32);
            let (ll_d, cl_d) = decode_partitions_dual(
                &mut dec,
                &mut dctx,
                w,
                h,
                cfg,
                cc,
                |_, _, _, _, _, _, _| {},
                |_, _, _, _, _, _, _| {},
            );
            assert_eq!(dec.decode_terminate(), 1, "terminate mismatch {w}x{h}");
            assert_eq!(ll_e, ll_d, "dual luma partition enc/dec mismatch {w}x{h}");
            assert_eq!(cl_e, cl_d, "dual chroma partition enc/dec mismatch {w}x{h}");
            // Both trees tile the whole picture.
            let la: u64 = ll_e.iter().map(|l| (l.w as u64) * (l.h as u64)).sum();
            let ca: u64 = cl_e.iter().map(|l| (l.w as u64) * (l.h as u64)).sum();
            assert_eq!(la, (w as u64) * (h as u64), "luma tiling {w}x{h}");
            assert_eq!(ca, (w as u64) * (h as u64), "chroma tiling {w}x{h}");
            // Chroma is one CU per 64×64; luma split further (rectangles present).
            assert!(
                cl_e.iter().all(|l| l.w == 64 && l.h == 64),
                "chroma not 64×64 {w}x{h}"
            );
            assert!(ll_e.iter().any(|l| l.w != l.h), "luma not split {w}x{h}");
            // The two trees genuinely differ (independent partitioning).
            assert!(ll_e.len() > cl_e.len(), "trees not independent {w}x{h}");
        }
    }
}
