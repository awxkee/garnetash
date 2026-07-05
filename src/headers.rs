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

use crate::bitstream::{BitReader, BitWriter, NalUnitType, annexb, write_nal};
use crate::error::EncodeError;
use crate::fmt::{BitDepth, ChromaFormat};

/// Log2 of the coding-tree-unit size. VVC permits 5, 6, or 7 (32/64/128);
/// garnetash v1 uses 32.
pub(crate) const LOG2_CTU_SIZE: u32 = 7;
/// Log2 of the minimum luma coding block size (4×4).
pub(crate) const LOG2_MIN_CB_SIZE: u32 = 2;
/// `general_level_idc`: Level 6.2 (`major*16 + minor*3` = 6*16 + 2*3 = 102),
/// generous enough to cover all supported picture sizes for now.
pub(crate) const LEVEL_IDC: u8 = 102;

/// Everything needed to emit the parameter sets for one picture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Headers {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma: ChromaFormat,
    pub(crate) bit_depth: BitDepth,
    /// Picture QP signalled in the PPS (`pps_init_qp_minus26 = qp - 26`).
    pub(crate) qp: u8,
    /// When set, the SPS enables transform-skip (for the lossless TSRC path)
    /// and the slice header enables TS residual coding. Lossy streams leave it
    /// clear so their bitstreams are byte-for-byte unchanged.
    pub(crate) lossless: bool,
    /// Perceptual adaptive quantization: enables `pps_cu_qp_delta_enabled_flag`
    /// and the picture-header QG subdivision so per-block `cu_qp_delta` may be
    /// signalled. Off leaves the bitstream byte-for-byte unchanged.
    pub(crate) aq: bool,
    /// Multi-type-tree partitioning: sets a non-zero `sps_max_mtt_hierarchy_depth`
    /// so binary/ternary splits (rectangular CUs) may be signalled. Off leaves the
    /// bitstream byte-for-byte unchanged (quadtree-only).
    pub(crate) mtt: bool,
    /// Enable LFNST (low-frequency non-separable transform) for intra luma blocks.
    pub(crate) lfnst: bool,
    /// Enable dependent quantization (VVC `sps_dep_quant_enabled_flag` +
    /// `sh_dep_quant_used_flag`). When set, transformed (non-transform-skip)
    /// blocks use the 4-state trellis quantizer. Off leaves the bitstream
    /// byte-for-byte unchanged.
    pub(crate) dep_quant: bool,
    /// Enable explicit MTS (multiple transform selection) for intra luma blocks
    /// (`sps_mts_enabled_flag` + `sps_explicit_mts_intra_enabled_flag`). When set,
    /// luma TUs up to 32×32 (with `lfnst_idx == 0`) may signal a per-TU
    /// DST-VII/DCT-VIII transform pair via `mts_idx`. Off leaves the bitstream
    /// byte-for-byte unchanged.
    pub(crate) mts: bool,
    /// Enable separate luma/chroma coding trees in the intra slice
    /// (VVC `sps_qtbtt_dual_tree_intra_flag`). Requires the dual-tree coding
    /// path; ignored for monochrome.
    pub(crate) dual_tree: bool,
    /// Enable CCLM cross-component chroma prediction (`sps_cclm_enabled_flag`).
    pub(crate) cclm: bool,
    /// Enable the in-loop deblocking filter (clears
    /// `pps_deblocking_filter_disabled_flag`). Off leaves the bitstream
    /// byte-for-byte unchanged.
    pub(crate) deblock: bool,
}

impl Headers {
    /// Coded luma width, padded up to a multiple of 8 (= Max(8, MinCbSizeY))
    /// as required by VVC; the excess is cropped by the conformance window.
    pub(crate) fn coded_width(&self) -> u32 {
        (self.width + 7) & !7
    }

    pub(crate) fn coded_height(&self) -> u32 {
        (self.height + 7) & !7
    }

    /// `general_profile_idc` (H.266 Annex A): Main 10 for 4:2:0/monochrome,
    /// Main 10 4:4:4 for 4:2:2/4:4:4; Main 12 variants at 12-bit.
    pub(crate) fn profile_idc(&self) -> u8 {
        let is_444_family = matches!(self.chroma, ChromaFormat::Yuv422 | ChromaFormat::Yuv444);
        match (self.bit_depth, is_444_family) {
            (BitDepth::Twelve, true) => 34, // MAIN_12_444
            (BitDepth::Twelve, false) => 2, // MAIN_12
            (_, true) => 33,                // MAIN_10_444
            (_, false) => 1,                // MAIN_10
        }
    }

    /// `gci_three_minus_max_chroma_format_constraint_idc` source value.
    fn max_chroma_constraint_idc(&self) -> u32 {
        self.chroma.idc()
    }

    // ── profile_tier_level (§7.3.3.1) ──────────────────────────────────────

    fn write_ptl(&self, w: &mut BitWriter) {
        // profileTierPresentFlag = true, maxNumSubLayersMinus1 = 0.
        w.put_bits(self.profile_idc() as u32, 7); // general_profile_idc
        w.put_bit(0); // general_tier_flag = main tier
        w.put_bits(LEVEL_IDC as u32, 8); // general_level_idc
        w.put_bit(1); // ptl_frame_only_constraint_flag
        w.put_bit(0); // ptl_multilayer_enabled_flag
        // general_constraints_info(): no constraints signalled.
        w.put_bit(0); // gci_present_flag
        while !w.is_byte_aligned() {
            w.put_bit(0); // gci_alignment_zero_bit
        }
        // No sub-layers: the sub_layer_level_present loop is empty.
        while !w.is_byte_aligned() {
            w.put_bit(0); // ptl_reserved_zero_bit
        }
        w.put_bits(0, 8); // ptl_num_sub_profiles
    }

    // ── SPS (§7.3.2.3) ─────────────────────────────────────────────────────

    /// Build the SPS RBSP (without NAL framing).
    /// Build the byte-aligned VUI payload. It tags the stream as BT.601
    /// full-range 4:2:0 YCbCr — exactly what `encode_core` produces — so a
    /// decoder reconstructs the input RGB without guessing a color matrix.
    fn vui_payload(&self) -> Vec<u8> {
        let mut v = BitWriter::new();
        v.put_bit(1); // vui_progressive_source_flag
        v.put_bit(0); // vui_interlaced_source_flag
        v.put_bit(0); // vui_non_packed_constraint_flag
        v.put_bit(0); // vui_non_projected_constraint_flag
        v.put_bit(0); // vui_aspect_ratio_info_present_flag
        v.put_bit(0); // vui_overscan_info_present_flag
        v.put_bit(1); // vui_color_description_present_flag
        v.put_bits(1, 8); //  vui_color_primaries          = BT.709 (sRGB gamut)
        v.put_bits(13, 8); // vui_transfer_characteristics  = IEC 61966-2-1 (sRGB)
        v.put_bits(6, 8); //  vui_matrix_coeffs             = SMPTE 170M (BT.601)
        v.put_bit(1); // vui_full_range_flag
        v.put_bit(0); // vui_chroma_loc_info_present_flag
        // vui_payload trailing: a stop bit then zero-pad to a byte boundary.
        if !v.is_byte_aligned() {
            v.put_bit(1); // vui_payload_bit_equal_to_one
            v.byte_align(); // vui_payload_bit_equal_to_zero ...
        }
        v.into_bytes()
    }

    pub(crate) fn write_sps_rbsp(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_bits(0, 4); // sps_seq_parameter_set_id
        w.put_bits(0, 4); // sps_video_parameter_set_id
        w.put_bits(0, 3); // sps_max_sub_layers_minus1
        w.put_bits(self.chroma.idc(), 2); // sps_chroma_format_idc
        w.put_bits(LOG2_CTU_SIZE - 5, 2); // sps_log2_ctu_size_minus5
        w.put_bit(1); // sps_ptl_dpb_hrd_params_present_flag
        self.write_ptl(&mut w);

        w.put_bit(0); // sps_gdr_enabled_flag
        w.put_bit(0); // sps_ref_pic_resampling_enabled_flag

        w.put_ue(self.coded_width()); // sps_pic_width_max_in_luma_samples
        w.put_ue(self.coded_height()); // sps_pic_height_max_in_luma_samples

        // Conformance window crops the padding back as far as the chroma format
        // permits. Subsampled odd dimensions retain one replicated edge sample;
        // the HEIF ispe property carries the exact display extent.
        let crop_r = (self.coded_width() - self.width) / self.chroma.sub_w() as u32;
        let crop_b = (self.coded_height() - self.height) / self.chroma.sub_h() as u32;
        if crop_r != 0 || crop_b != 0 {
            w.put_bit(1); // sps_conformance_window_flag
            w.put_ue(0); // sps_conf_win_left_offset
            w.put_ue(crop_r); // sps_conf_win_right_offset
            w.put_ue(0); // sps_conf_win_top_offset
            w.put_ue(crop_b); // sps_conf_win_bottom_offset
        } else {
            w.put_bit(0); // sps_conformance_window_flag
        }

        w.put_bit(0); // sps_subpic_info_present_flag
        w.put_ue(self.bit_depth.minus8() as u32); // sps_bitdepth_minus8
        w.put_bit(0); // sps_entropy_coding_sync_enabled_flag
        w.put_bit(0); // sps_entry_point_offsets_present_flag
        w.put_bits(0, 4); // sps_log2_max_pic_order_cnt_lsb_minus4 (POC = 4 bits)
        w.put_bit(0); // sps_poc_msb_cycle_flag
        w.put_bits(0, 2); // sps_num_extra_ph_bytes
        w.put_bits(0, 2); // sps_num_extra_sh_bytes

        // dpb_parameters (single sub-layer): DPB of 1, no reorder, no latency.
        w.put_ue(0); // dpb_max_dec_pic_buffering_minus1
        w.put_ue(0); // dpb_max_num_reorder_pics
        w.put_ue(0); // dpb_max_latency_increase_plus1

        w.put_ue(LOG2_MIN_CB_SIZE - 2); // sps_log2_min_luma_coding_block_size_minus2
        w.put_bit(0); // sps_partition_constraints_override_enabled_flag

        // Intra slice luma partitioning. Quadtree-only (no MTT) leaves these all
        // zero; MTT signals a non-zero hierarchy depth plus the BT/TT size diffs.
        if self.mtt {
            let (min_qt_diff, max_depth, max_bt_diff, max_tt_diff) =
                crate::partition::mtt_sps_fields();
            w.put_ue(min_qt_diff); // sps_log2_diff_min_qt_min_cb_intra_slice_luma
            w.put_ue(max_depth); // sps_max_mtt_hierarchy_depth_intra_slice_luma
            w.put_ue(max_bt_diff); // sps_log2_diff_max_bt_min_qt_intra_slice_luma
            w.put_ue(max_tt_diff); // sps_log2_diff_max_tt_min_qt_intra_slice_luma
        } else {
            w.put_ue(0); // sps_log2_diff_min_qt_min_cb_intra_slice_luma
            w.put_ue(0); // sps_max_mtt_hierarchy_depth_intra_slice_luma  (=> no MTT)
            // MaxMTTDepth == 0 -> no max_bt/max_tt diff fields.
        }

        if !self.chroma.is_monochrome() {
            w.put_bit(self.dual_tree as u32); // sps_qtbtt_dual_tree_intra_flag
            if self.dual_tree {
                // Chroma intra partition constraints (present iff dual tree). v1:
                // chroma min QT = 64 (one CU per 64×64 region) and no chroma MTT.
                // MinQtSizeC = 1 << (4 + log2MinCb=2) = 64.
                w.put_ue(4); // sps_log2_diff_min_qt_min_cb_intra_slice_chroma
                w.put_ue(0); // sps_max_mtt_hierarchy_depth_intra_slice_chroma (=> no chroma MTT)
            }
        }
        // Inter slice partitioning (present even for intra-only sequences).
        w.put_ue(0); // sps_log2_diff_min_qt_min_cb_inter_slice
        w.put_ue(0); // sps_max_mtt_hierarchy_depth_inter_slice

        // sps_max_luma_transform_size_64_flag is present only when CtbSizeY>32.
        // We always allow the 64-point transform (max TB = 64) when present, so
        // a 64×64 CU is coded as a single transform unit.
        if (1u32 << LOG2_CTU_SIZE) > 32 {
            w.put_bit(1); // sps_max_luma_transform_size_64_flag (=> max TB 64)
        }
        // Transform-skip is enabled for both lossless (mandatory) and lossy: on
        // the lossy path the encoder may choose transform-skip per block for
        // screen-content-like residuals (sharp edges code far better than DCT).
        w.put_bit(1); // sps_transform_skip_enabled_flag
        // Transform-skip up to 32x32 (log2 = 5) so every leaf CU qualifies.
        w.put_ue(3); // sps_log2_transform_skip_max_size_minus2  (=> 32)
        // BDPCM is a lossless-only tool here.
        w.put_bit(self.lossless as u32); // sps_bdpcm_enabled_flag
        w.put_bit(self.mts as u32); // sps_mts_enabled_flag
        if self.mts {
            w.put_bit(1); // sps_explicit_mts_intra_enabled_flag
            w.put_bit(0); // sps_explicit_mts_inter_enabled_flag (intra-only encoder)
        }
        w.put_bit(self.lfnst as u32); // sps_lfnst_enabled_flag

        if !self.chroma.is_monochrome() {
            w.put_bit(0); // sps_joint_cbcr_enabled_flag
            w.put_bit(1); // sps_same_qp_table_for_chroma_flag
            // One identity chroma QP mapping table.
            w.put_se(0); // sps_qp_table_start_minus26
            w.put_ue(0); // sps_num_points_in_qp_table_minus1
            w.put_ue(0); // sps_delta_qp_in_val_minus1[0]   (deltaIn = 1)
            w.put_ue(1); // sps_delta_qp_diff_val[0]         (deltaOut = 1 ^ 0)
        }

        w.put_bit(0); // sps_sao_enabled_flag
        w.put_bit(0); // sps_alf_enabled_flag
        w.put_bit(0); // sps_lmcs_enable_flag
        w.put_bit(0); // sps_weighted_pred_flag
        w.put_bit(0); // sps_weighted_bipred_flag
        w.put_bit(0); // sps_long_term_ref_pics_flag
        w.put_bit(0); // sps_idr_rpl_present_flag
        w.put_bit(0); // sps_rpl1_same_as_rpl0_flag
        w.put_ue(0); // sps_num_ref_pic_lists[0]
        w.put_ue(0); // sps_num_ref_pic_lists[1]

        w.put_bit(0); // sps_ref_wraparound_enabled_flag
        w.put_bit(0); // sps_temporal_mvp_enabled_flag
        w.put_bit(0); // sps_amvr_enabled_flag
        w.put_bit(0); // sps_bdof_enabled_flag
        w.put_bit(0); // sps_smvd_enabled_flag
        w.put_bit(0); // sps_dmvr_enabled_flag
        w.put_bit(0); // sps_mmvd_enabled_flag
        w.put_ue(0); // sps_six_minus_max_num_merge_cand  (=> 6 merge cands)
        w.put_bit(0); // sps_sbt_enabled_flag
        w.put_bit(0); // sps_affine_enabled_flag
        w.put_bit(0); // sps_bcw_enabled_flag
        w.put_bit(0); // sps_ciip_enabled_flag
        // sps_gpm: only when MaxNumMergeCand >= 2 (it is, =6).
        w.put_bit(0); // sps_gpm_enabled_flag
        w.put_ue(0); // sps_log2_parallel_merge_level_minus2

        w.put_bit(0); // sps_isp_enabled_flag
        w.put_bit(0); // sps_mrl_enabled_flag
        w.put_bit(0); // sps_mip_enabled_flag
        if !self.chroma.is_monochrome() {
            w.put_bit(self.cclm as u32); // sps_cclm_enabled_flag
        }
        if matches!(self.chroma, ChromaFormat::Yuv420) {
            w.put_bit(0); // sps_chroma_horizontal_collocated_flag
            w.put_bit(0); // sps_chroma_vertical_collocated_flag
        }
        w.put_bit(0); // sps_palette_enabled_flag
        // sps_act_enabled_flag is present only for 4:4:4 AND when the max
        // transform size is *not* 64 (Log2MaxTbSize != 6, per H.266 §7.3.2.4
        // and vvdec HLSyntaxReader). garnetash always uses a 128 CTU with the
        // 64-point max transform (Log2MaxTbSize == 6), so the flag is never
        // present; writing it unconditionally desynchronised the 4:4:4 SPS.
        if matches!(self.chroma, ChromaFormat::Yuv444) && (1u32 << LOG2_CTU_SIZE) <= 32 {
            w.put_bit(0); // sps_act_enabled_flag
        }
        // sps_min_qp_prime_ts_minus4 is present whenever transform-skip is
        // enabled (always, now). Minimum Qp' for transform-skip blocks is 4,
        // matching the lossless QP, so the value is 0.
        w.put_ue(0); // sps_min_qp_prime_ts_minus4
        w.put_bit(0); // sps_ibc_enabled_flag
        w.put_bit(0); // sps_ladf_enabled_flag
        w.put_bit(0); // sps_explicit_scaling_list_enabled_flag
        w.put_bit(self.dep_quant as u32); // sps_dep_quant_enabled_flag
        w.put_bit(0); // sps_sign_data_hiding_enabled_flag
        w.put_bit(0); // sps_virtual_boundaries_enabled_flag
        w.put_bit(0); // sps_timing_hrd_params_present_flag (under ptl/dpb/hrd)
        w.put_bit(0); // sps_field_seq_flag
        // VUI: signal BT.601 full-range color so RGB round-trips correctly.
        let vui = self.vui_payload();
        w.put_bit(1); // sps_vui_parameters_present_flag
        w.put_ue((vui.len() - 1) as u32); // sps_vui_payload_size_minus1
        while !w.is_byte_aligned() {
            w.put_bit(0); // sps_vui_alignment_zero_bit
        }
        for &b in &vui {
            w.put_bits(b as u32, 8);
        }
        w.put_bit(0); // sps_extension_present_flag

        w.rbsp_trailing_bits();
        w.into_bytes()
    }

    // ── PPS (§7.3.2.4) ─────────────────────────────────────────────────────

    /// Build the PPS RBSP (without NAL framing).
    pub(crate) fn write_pps_rbsp(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_bits(0, 6); // pps_pic_parameter_set_id
        w.put_bits(0, 4); // pps_seq_parameter_set_id
        w.put_bit(0); // pps_mixed_nalu_types_in_pic_flag
        w.put_ue(self.coded_width()); // pps_pic_width_in_luma_samples
        w.put_ue(self.coded_height()); // pps_pic_height_in_luma_samples
        w.put_bit(0); // pps_conformance_window_flag (inherits SPS)
        w.put_bit(0); // pps_scaling_window_explicit_signalling_flag
        w.put_bit(0); // pps_output_flag_present_flag
        w.put_bit(1); // pps_no_pic_partition_flag (one tile, one slice)
        w.put_bit(0); // pps_subpic_id_mapping_present_flag
        // No partitioning -> tile/slice block skipped.
        w.put_bit(0); // pps_cabac_init_present_flag
        w.put_ue(0); // pps_num_ref_idx_default_active_minus1[0]
        w.put_ue(0); // pps_num_ref_idx_default_active_minus1[1]
        w.put_bit(0); // pps_rpl1_idx_present_flag
        w.put_bit(0); // pps_weighted_pred_flag
        w.put_bit(0); // pps_weighted_bipred_flag
        w.put_bit(0); // pps_ref_wraparound_enabled_flag
        // vvdec derives Qp'Y = SliceQpY + QpBdOffset, with QpBdOffset =
        // 6·(BitDepth−8). Signal SliceQpY pre-offset so the effective luma QP is
        // exactly `self.qp` (identity transform-skip at the lossless QP). No-op
        // at 8-bit (offset 0).
        let qp_bd_offset = 6 * (self.bit_depth.bits() as i32 - 8);
        w.put_se(self.qp as i32 - qp_bd_offset - 26); // pps_init_qp_minus26
        w.put_bit(self.aq as u32); // pps_cu_qp_delta_enabled_flag
        w.put_bit(0); // pps_chroma_tool_offsets_present_flag
        w.put_bit(1); // pps_deblocking_filter_control_present_flag
        w.put_bit(0); // pps_deblocking_filter_override_enabled_flag
        w.put_bit(!self.deblock as u32); // pps_deblocking_filter_disabled_flag
        if self.deblock {
            // chroma_tool_offsets_present is 0, so only the luma beta/tc offsets
            // are present; both signalled as 0 (slice uses these, no override).
            w.put_se(0); // pps_beta_offset_div2
            w.put_se(0); // pps_tc_offset_div2
        }
        w.put_bit(0); // pps_picture_header_extension_present_flag
        w.put_bit(0); // pps_slice_header_extension_present_flag
        w.put_bit(0); // pps_extension_flag
        w.rbsp_trailing_bits();
        w.into_bytes()
    }

    /// Assemble the leading non-VCL NAL units (SPS + PPS) as an Annex-B stream.
    /// The picture header and coded slice are appended by the slice layer.
    pub(crate) fn write_parameter_set_nals(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&annexb(&write_nal(
            NalUnitType::Sps,
            &self.write_sps_rbsp(),
            1,
        )));
        out.extend_from_slice(&annexb(&write_nal(
            NalUnitType::Pps,
            &self.write_pps_rbsp(),
            1,
        )));
        out
    }

    // ── Picture header (§7.3.2.8, picture_header_structure) ─────────────────

    /// Write the picture-header structure (no RBSP trailing bits). For an intra
    /// IDR with every advanced/in-loop tool disabled in the SPS/PPS, this is the
    /// minimal six-element form.
    fn write_picture_header(&self, w: &mut BitWriter) {
        w.put_bit(1); // ph_gdr_or_irap_pic_flag (IRAP)
        w.put_bit(0); // ph_non_ref_pic_flag
        w.put_bit(0); // ph_gdr_pic_flag (IRAP, not GDR)
        w.put_bit(0); // ph_inter_slice_allowed_flag (=> intra only)
        // ph_intra_slice_allowed_flag inferred 1.
        w.put_ue(0); // ph_pic_parameter_set_id
        w.put_bits(0, 4); // ph_pic_order_cnt_lsb (POC = 0, 4 bits)
        // Intra-slice QG subdivision: present iff cu_qp_delta is enabled. 0 puts
        // one quantization group per CTU (the simplest QP-predictor chain).
        if self.aq {
            // MTT quantization groups are derived by subdivision level, not a
            // fixed pixel grid; subdiv 0 places one QG per CTU, which the simple
            // fixed-grid predictor handles correctly for both QT and MTT.
            let subdiv = if self.dual_tree {
                2
            } else if self.mtt {
                0
            } else {
                crate::encode::AQ_CU_QP_DELTA_SUBDIV
            };
            w.put_ue(subdiv); // ph_cu_qp_delta_subdiv_intra_slice
        }
        // No ALF/LMCS/scaling/virtual-boundary/RPL/partition-override signalling:
        // all gated off by the SPS/PPS configuration.
    }

    // ── Slice header (§7.3.7.1) ────────────────────────────────────────────

    /// Write the slice header for the single intra IDR slice, with the picture
    /// header embedded, ending on a byte boundary (ready for `slice_data`).
    fn write_slice_header(&self, w: &mut BitWriter) {
        w.put_bit(1); // sh_picture_header_in_slice_header_flag
        self.write_picture_header(w);
        // subpic info absent; one tile / one slice -> no slice address;
        // intra-only -> sh_slice_type inferred I.
        w.put_bit(0); // sh_no_output_of_prior_pics_flag (IDR NAL)
        // ALF/LMCS/scaling/RPL/ref-idx/cabac-init/TMVP/weighted-pred: skipped.
        w.put_se(0); // sh_qp_delta (SliceQp == pps_init_qp)
        // chroma QP offsets / SAO / deblock-override skipped by configuration.
        // Dependent quantization: sps flag == self.dep_quant, so the slice flag
        // is present iff dep_quant is enabled, and signals it is used.
        if self.dep_quant {
            w.put_bit(1); // sh_dep_quant_used_flag
        }
        // sign_data_hiding disabled. sh_ts_residual_coding_disabled_flag is
        // present iff transform-skip is enabled AND dep-quant is *not* used (per
        // VVC); transform-skip is always enabled, so it is present exactly when
        // dep_quant is off. 0 selects the dedicated transform-skip residual coder.
        if !self.dep_quant {
            w.put_bit(0); // sh_ts_residual_coding_disabled_flag (=> use TSRC)
        }
        // No entry points (single substream). Align for slice_data.
        w.put_bit(1); // alignment_bit_equal_to_one
        w.byte_align(); // alignment_bit_equal_to_zero ...
    }

    /// Build the complete IDR VCL NAL unit: slice header (byte-aligned) followed
    /// by the CABAC-coded `slice_data` payload, NAL-framed as `IDR_N_LP`.
    ///
    /// `slice_data` is the byte-aligned output of the slice-data CABAC coder
    /// (CTU coding), which terminates the RBSP itself. That layer is the next
    /// pipeline stage; this function is the assembly seam.
    pub(crate) fn write_idr_slice_nal(&self, slice_data: &[u8]) -> Vec<u8> {
        let mut w = BitWriter::new();
        self.write_slice_header(&mut w);
        let mut rbsp = w.into_bytes();
        rbsp.extend_from_slice(slice_data);
        write_nal(NalUnitType::IdrNLp, &rbsp, 1)
    }

    /// Assemble a full still-picture Annex-B stream: SPS, PPS, then the IDR
    /// slice carrying `slice_data`.
    pub(crate) fn write_still_picture(&self, slice_data: &[u8]) -> Vec<u8> {
        let mut out = self.write_parameter_set_nals();
        out.extend_from_slice(&annexb(&self.write_idr_slice_nal(slice_data)));
        out
    }
}

// ── Decoder-side parsing ───────────────────────────────────────────────────
//
// These read back exactly the fields the writers above emit, in the same order.
// They are intentionally matched to garnetash's own (fixed) parameter-set and
// slice-header layout rather than being a general VVC parser.

/// The picture parameters a decoder needs, recovered from the SPS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParsedSps {
    pub(crate) chroma: ChromaFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bit_depth: BitDepth,
    /// `sps_bdpcm_enabled_flag`, which garnetash sets iff the stream is lossless.
    pub(crate) lossless: bool,
    /// True when `sps_max_mtt_hierarchy_depth_intra_slice_luma > 0`, i.e. the
    /// coding tree may contain binary/ternary splits (rectangular CUs).
    pub(crate) mtt: bool,
    pub(crate) lfnst: bool,
    pub(crate) dep_quant: bool,
    pub(crate) mts: bool,
    pub(crate) dual_tree: bool,
    pub(crate) cclm: bool,
}

fn chroma_from_idc(idc: u32) -> Result<ChromaFormat, EncodeError> {
    match idc {
        0 => Ok(ChromaFormat::Monochrome),
        1 => Ok(ChromaFormat::Yuv420),
        2 => Ok(ChromaFormat::Yuv422),
        3 => Ok(ChromaFormat::Yuv444),
        _ => Err(EncodeError::Decode("invalid chroma_format_idc")),
    }
}

fn bit_depth_from_minus8(m: u32) -> Result<BitDepth, EncodeError> {
    match m {
        0 => Ok(BitDepth::Eight),
        2 => Ok(BitDepth::Ten),
        4 => Ok(BitDepth::Twelve),
        _ => Err(EncodeError::Decode("unsupported bit depth")),
    }
}

/// Parse the SPS RBSP up to and including `sps_bdpcm_enabled_flag`, recovering
/// the picture geometry, chroma format, bit depth and lossless flag.
pub(crate) fn parse_sps(rbsp: &[u8]) -> Result<ParsedSps, EncodeError> {
    let mut r = BitReader::new(rbsp);
    r.read_bits(4); // sps_seq_parameter_set_id
    r.read_bits(4); // sps_video_parameter_set_id
    r.read_bits(3); // sps_max_sub_layers_minus1
    let chroma = chroma_from_idc(r.read_bits(2))?;
    let log2_ctu_minus5 = r.read_bits(2);
    r.read_bit(); // sps_ptl_dpb_hrd_params_present_flag (=1)
    // profile_tier_level
    r.read_bits(7); // general_profile_idc
    r.read_bit(); // general_tier_flag
    r.read_bits(8); // general_level_idc
    r.read_bit(); // ptl_frame_only_constraint_flag
    r.read_bit(); // ptl_multilayer_enabled_flag
    r.read_bit(); // gci_present_flag (=0)
    r.byte_align(); // gci_alignment + ptl_reserved
    r.read_bits(8); // ptl_num_sub_profiles (=0)

    r.read_bit(); // sps_gdr_enabled_flag
    r.read_bit(); // sps_ref_pic_resampling_enabled_flag
    let coded_w = r.read_ue();
    let coded_h = r.read_ue();
    let (mut crop_r, mut crop_b) = (0u32, 0u32);
    if r.read_bit() == 1 {
        r.read_ue(); // left
        crop_r = r.read_ue(); // right
        r.read_ue(); // top
        crop_b = r.read_ue(); // bottom
    }
    r.read_bit(); // sps_subpic_info_present_flag
    let bit_depth = bit_depth_from_minus8(r.read_ue())?;
    r.read_bit(); // entropy_coding_sync
    r.read_bit(); // entry_point_offsets_present
    r.read_bits(4); // log2_max_pic_order_cnt_lsb_minus4
    r.read_bit(); // poc_msb_cycle_flag
    r.read_bits(2); // num_extra_ph_bytes
    r.read_bits(2); // num_extra_sh_bytes
    r.read_ue(); // dpb_max_dec_pic_buffering_minus1
    r.read_ue(); // dpb_max_num_reorder_pics
    r.read_ue(); // dpb_max_latency_increase_plus1
    r.read_ue(); // log2_min_cb_minus2
    r.read_bit(); // partition_constraints_override
    r.read_ue(); // diff_min_qt_min_cb_intra_luma
    let max_mtt_depth_intra = r.read_ue(); // max_mtt_depth_intra_luma
    let mtt = max_mtt_depth_intra > 0;
    if mtt {
        r.read_ue(); // sps_log2_diff_max_bt_min_qt_intra_slice_luma
        r.read_ue(); // sps_log2_diff_max_tt_min_qt_intra_slice_luma
    }
    let mut dual_tree = false;
    if !chroma.is_monochrome() {
        dual_tree = r.read_bit() == 1; // sps_qtbtt_dual_tree_intra_flag
        if dual_tree {
            r.read_ue(); // sps_log2_diff_min_qt_min_cb_intra_slice_chroma
            let chroma_mtt_depth = r.read_ue(); // sps_max_mtt_hierarchy_depth_intra_slice_chroma
            if chroma_mtt_depth > 0 {
                r.read_ue(); // sps_log2_diff_max_bt_min_qt_intra_slice_chroma
                r.read_ue(); // sps_log2_diff_max_tt_min_qt_intra_slice_chroma
            }
        }
    }
    r.read_ue(); // diff_min_qt_min_cb_inter
    r.read_ue(); // max_mtt_depth_inter
    if (1u32 << (log2_ctu_minus5 + 5)) > 32 {
        r.read_bit(); // sps_max_luma_transform_size_64_flag
    }
    r.read_bit(); // sps_transform_skip_enabled_flag (=1)
    r.read_ue(); // sps_log2_transform_skip_max_size_minus2
    let lossless = r.read_bit() == 1; // sps_bdpcm_enabled_flag
    let mts = r.read_bit() == 1; // sps_mts_enabled_flag
    if mts {
        r.read_bit(); // sps_explicit_mts_intra_enabled_flag
        r.read_bit(); // sps_explicit_mts_inter_enabled_flag
    }
    let lfnst = r.read_bit() == 1; // sps_lfnst_enabled_flag
    // Read through the remaining SPS syntax to sps_dep_quant_enabled_flag,
    // mirroring the writer exactly (all tool flags are 0; the conditional bits
    // depend only on the chroma format). sps_act_enabled_flag is never present
    // (128 CTU => 64-point max transform).
    if !chroma.is_monochrome() {
        r.read_bit(); // sps_joint_cbcr_enabled_flag
        r.read_bit(); // sps_same_qp_table_for_chroma_flag (=1 => one table)
        r.read_se(); // sps_qp_table_start_minus26
        r.read_ue(); // sps_num_points_in_qp_table_minus1 (=0 => 1 point)
        r.read_ue(); // sps_delta_qp_in_val_minus1[0]
        r.read_ue(); // sps_delta_qp_diff_val[0]
    }
    r.read_bit(); // sps_sao_enabled_flag
    r.read_bit(); // sps_alf_enabled_flag
    r.read_bit(); // sps_lmcs_enable_flag
    r.read_bit(); // sps_weighted_pred_flag
    r.read_bit(); // sps_weighted_bipred_flag
    r.read_bit(); // sps_long_term_ref_pics_flag
    r.read_bit(); // sps_idr_rpl_present_flag
    r.read_bit(); // sps_rpl1_same_as_rpl0_flag
    r.read_ue(); // sps_num_ref_pic_lists[0]
    r.read_ue(); // sps_num_ref_pic_lists[1]
    r.read_bit(); // sps_ref_wraparound_enabled_flag
    r.read_bit(); // sps_temporal_mvp_enabled_flag
    r.read_bit(); // sps_amvr_enabled_flag
    r.read_bit(); // sps_bdof_enabled_flag
    r.read_bit(); // sps_smvd_enabled_flag
    r.read_bit(); // sps_dmvr_enabled_flag
    r.read_bit(); // sps_mmvd_enabled_flag
    r.read_ue(); // sps_six_minus_max_num_merge_cand
    r.read_bit(); // sps_sbt_enabled_flag
    r.read_bit(); // sps_affine_enabled_flag
    r.read_bit(); // sps_bcw_enabled_flag
    r.read_bit(); // sps_ciip_enabled_flag
    r.read_bit(); // sps_gpm_enabled_flag
    r.read_ue(); // sps_log2_parallel_merge_level_minus2
    r.read_bit(); // sps_isp_enabled_flag
    r.read_bit(); // sps_mrl_enabled_flag
    r.read_bit(); // sps_mip_enabled_flag
    let mut cclm = false;
    if !chroma.is_monochrome() {
        cclm = r.read_bit() == 1; // sps_cclm_enabled_flag
    }
    if matches!(chroma, ChromaFormat::Yuv420) {
        r.read_bit(); // sps_chroma_horizontal_collocated_flag
        r.read_bit(); // sps_chroma_vertical_collocated_flag
    }
    r.read_bit(); // sps_palette_enabled_flag
    r.read_ue(); // sps_min_qp_prime_ts_minus4
    r.read_bit(); // sps_ibc_enabled_flag
    r.read_bit(); // sps_ladf_enabled_flag
    r.read_bit(); // sps_explicit_scaling_list_enabled_flag
    let dep_quant = r.read_bit() == 1; // sps_dep_quant_enabled_flag

    let sub_w = chroma.sub_w() as u32;
    let sub_h = chroma.sub_h() as u32;
    let width = coded_w
        .checked_sub(crop_r * sub_w)
        .ok_or(EncodeError::Decode("bad crop"))?;
    let height = coded_h
        .checked_sub(crop_b * sub_h)
        .ok_or(EncodeError::Decode("bad crop"))?;

    Ok(ParsedSps {
        chroma,
        width,
        height,
        bit_depth,
        lossless,
        mtt,
        lfnst,
        dep_quant,
        mts,
        dual_tree,
        cclm,
    })
}

/// Parse the PPS RBSP up to `pps_cu_qp_delta_enabled_flag`, returning the
/// picture QP and whether per-block `cu_qp_delta` is enabled (adaptive quant).
pub(crate) fn parse_pps_qp(
    rbsp: &[u8],
    bit_depth: BitDepth,
) -> Result<(u8, bool, bool), EncodeError> {
    let mut r = BitReader::new(rbsp);
    r.read_bits(6); // pps_pic_parameter_set_id
    r.read_bits(4); // pps_seq_parameter_set_id
    r.read_bit(); // pps_mixed_nalu_types_in_pic_flag
    r.read_ue(); // pps_pic_width_in_luma_samples
    r.read_ue(); // pps_pic_height_in_luma_samples
    r.read_bit(); // pps_conformance_window_flag
    r.read_bit(); // pps_scaling_window_explicit_signalling_flag
    r.read_bit(); // pps_output_flag_present_flag
    r.read_bit(); // pps_no_pic_partition_flag (=1)
    r.read_bit(); // pps_subpic_id_mapping_present_flag
    r.read_bit(); // pps_cabac_init_present_flag
    r.read_ue(); // pps_num_ref_idx_default_active_minus1[0]
    r.read_ue(); // pps_num_ref_idx_default_active_minus1[1]
    r.read_bit(); // pps_rpl1_idx_present_flag
    r.read_bit(); // pps_weighted_pred_flag
    r.read_bit(); // pps_weighted_bipred_flag
    r.read_bit(); // pps_ref_wraparound_enabled_flag
    let init_qp_minus26 = r.read_se();
    let qp_bd_offset = 6 * (bit_depth.bits() as i32 - 8);
    let qp = init_qp_minus26 + 26 + qp_bd_offset;
    if !(0..=63).contains(&qp) {
        return Err(EncodeError::Decode("QP out of range"));
    }
    let cu_qp_delta_enabled = r.read_bit() == 1; // pps_cu_qp_delta_enabled_flag
    r.read_bit(); // pps_chroma_tool_offsets_present_flag
    r.read_bit(); // pps_deblocking_filter_control_present_flag (=1)
    r.read_bit(); // pps_deblocking_filter_override_enabled_flag (=0)
    let deblock = r.read_bit() == 0; // !pps_deblocking_filter_disabled_flag
    Ok((qp as u8, cu_qp_delta_enabled, deblock))
}

/// Parse the slice header (picture-header-in-slice-header form) and return the
/// byte offset within `rbsp` at which the byte-aligned `slice_data` begins, plus
/// the intra-slice `cu_qp_delta` QG subdivision (0 when AQ is disabled).
pub(crate) fn slice_data_offset(
    rbsp: &[u8],
    cu_qp_delta_enabled: bool,
) -> Result<(usize, u32), EncodeError> {
    let mut r = BitReader::new(rbsp);
    r.read_bit(); // sh_picture_header_in_slice_header_flag (=1)
    // picture_header_structure
    r.read_bit(); // ph_gdr_or_irap_pic_flag
    r.read_bit(); // ph_non_ref_pic_flag
    r.read_bit(); // ph_gdr_pic_flag
    r.read_bit(); // ph_inter_slice_allowed_flag (=0 -> intra only)
    r.read_ue(); // ph_pic_parameter_set_id
    r.read_bits(4); // ph_pic_order_cnt_lsb
    let cu_qp_delta_subdiv = if cu_qp_delta_enabled {
        r.read_ue() // ph_cu_qp_delta_subdiv_intra_slice
    } else {
        0
    };
    // slice header body
    r.read_bit(); // sh_no_output_of_prior_pics_flag
    r.read_se(); // sh_qp_delta
    r.read_bit(); // sh_ts_residual_coding_disabled_flag
    r.read_bit(); // alignment_bit_equal_to_one
    r.byte_align();
    let off = r.bit_pos() / 8;
    if off > rbsp.len() {
        return Err(EncodeError::Decode("slice header overruns RBSP"));
    }
    Ok((off, cu_qp_delta_subdiv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::{BitReader, NalUnitType, rbsp_to_ebsp, write_nal};

    fn check_lands_on_stop_bit(rbsp: &[u8], consumed_bits: usize) {
        // The next bit after all syntax elements must be rbsp_stop_one_bit (1),
        // followed by zero padding to the byte boundary.
        let mut r = BitReader::new(rbsp);
        for _ in 0..consumed_bits {
            r.read_bit();
        }
        assert_eq!(r.read_bit(), 1, "expected rbsp_stop_one_bit");
        // Remaining bits in the final byte must be zero, and we must be at the
        // exact end of the RBSP.
        let end = rbsp.len() * 8;
        while r.bit_pos() < end {
            assert_eq!(r.read_bit(), 0, "alignment padding must be zero");
        }
        assert_eq!(r.bit_pos(), end, "RBSP not fully consumed");
    }

    /// Fully parse the SPS the same way a decoder would, returning the number of
    /// bits consumed (excluding trailing bits) and checking key field values.
    fn parse_sps(rbsp: &[u8], h: &Headers) -> usize {
        let mut r = BitReader::new(rbsp);
        assert_eq!(r.read_bits(4), 0); // sps_seq_parameter_set_id
        assert_eq!(r.read_bits(4), 0); // sps_video_parameter_set_id
        assert_eq!(r.read_bits(3), 0); // sps_max_sub_layers_minus1
        assert_eq!(r.read_bits(2), h.chroma.idc()); // chroma_format_idc
        assert_eq!(r.read_bits(2), LOG2_CTU_SIZE - 5); // log2_ctu_size_minus5
        assert_eq!(r.read_bit(), 1); // ptl_dpb_hrd_params_present_flag
        // profile_tier_level
        assert_eq!(r.read_bits(7), h.profile_idc() as u32);
        assert_eq!(r.read_bit(), 0); // tier
        assert_eq!(r.read_bits(8), LEVEL_IDC as u32);
        assert_eq!(r.read_bit(), 1); // frame_only
        assert_eq!(r.read_bit(), 0); // multilayer
        assert_eq!(r.read_bit(), 0); // gci_present_flag
        r.byte_align(); // gci_alignment + ptl_reserved align
        assert_eq!(r.read_bits(8), 0); // ptl_num_sub_profiles

        assert_eq!(r.read_bit(), 0); // gdr
        assert_eq!(r.read_bit(), 0); // rpr
        assert_eq!(r.read_ue(), h.coded_width());
        assert_eq!(r.read_ue(), h.coded_height());
        let crop_r = (h.coded_width() - h.width) / h.chroma.sub_w() as u32;
        let crop_b = (h.coded_height() - h.height) / h.chroma.sub_h() as u32;
        if crop_r != 0 || crop_b != 0 {
            assert_eq!(r.read_bit(), 1);
            assert_eq!(r.read_ue(), 0);
            assert_eq!(r.read_ue(), crop_r);
            assert_eq!(r.read_ue(), 0);
            assert_eq!(r.read_ue(), crop_b);
        } else {
            assert_eq!(r.read_bit(), 0);
        }
        assert_eq!(r.read_bit(), 0); // subpic_info_present
        assert_eq!(r.read_ue(), h.bit_depth.minus8() as u32);
        assert_eq!(r.read_bit(), 0); // entropy_coding_sync
        assert_eq!(r.read_bit(), 0); // entry_point_offsets
        assert_eq!(r.read_bits(4), 0); // log2_max_poc_lsb_minus4
        assert_eq!(r.read_bit(), 0); // poc_msb_cycle_flag
        assert_eq!(r.read_bits(2), 0); // num_extra_ph_bytes
        assert_eq!(r.read_bits(2), 0); // num_extra_sh_bytes
        assert_eq!(r.read_ue(), 0); // dpb_max_dec_pic_buffering_minus1
        assert_eq!(r.read_ue(), 0); // dpb_max_num_reorder_pics
        assert_eq!(r.read_ue(), 0); // dpb_max_latency_increase_plus1
        assert_eq!(r.read_ue(), 0); // log2_min_cb_minus2
        assert_eq!(r.read_bit(), 0); // partition_constraints_override
        assert_eq!(r.read_ue(), 0); // diff_min_qt_min_cb_intra_luma
        assert_eq!(r.read_ue(), 0); // max_mtt_depth_intra_luma
        if !h.chroma.is_monochrome() {
            assert_eq!(r.read_bit(), 0); // dual_tree_intra
        }
        assert_eq!(r.read_ue(), 0); // diff_min_qt_min_cb_inter
        assert_eq!(r.read_ue(), 0); // max_mtt_depth_inter
        if (1u32 << LOG2_CTU_SIZE) > 32 {
            assert_eq!(r.read_bit(), 1); // max_luma_transform_size_64_flag
        }
        assert_eq!(r.read_bit(), 1); // transform_skip (always enabled)
        assert_eq!(r.read_ue(), 3); // log2_transform_skip_max_size_minus2 (=> 32)
        assert_eq!(r.read_bit(), h.lossless as u32); // bdpcm (lossless only)
        assert_eq!(r.read_bit(), 0); // mts
        assert_eq!(r.read_bit(), 0); // lfnst
        if !h.chroma.is_monochrome() {
            assert_eq!(r.read_bit(), 0); // joint_cbcr
            assert_eq!(r.read_bit(), 1); // same_qp_table_for_chroma
            assert_eq!(r.read_se(), 0); // qp_table_start_minus26
            assert_eq!(r.read_ue(), 0); // num_points_minus1
            assert_eq!(r.read_ue(), 0); // delta_qp_in_val_minus1
            assert_eq!(r.read_ue(), 1); // delta_qp_diff_val
        }
        assert_eq!(r.read_bit(), 0); // sao
        assert_eq!(r.read_bit(), 0); // alf
        assert_eq!(r.read_bit(), 0); // lmcs
        assert_eq!(r.read_bit(), 0); // weighted_pred
        assert_eq!(r.read_bit(), 0); // weighted_bipred
        assert_eq!(r.read_bit(), 0); // long_term_ref
        assert_eq!(r.read_bit(), 0); // idr_rpl_present
        assert_eq!(r.read_bit(), 0); // rpl1_same_as_rpl0
        assert_eq!(r.read_ue(), 0); // num_ref_pic_lists[0]
        assert_eq!(r.read_ue(), 0); // num_ref_pic_lists[1]
        assert_eq!(r.read_bit(), 0); // ref_wraparound
        assert_eq!(r.read_bit(), 0); // temporal_mvp
        assert_eq!(r.read_bit(), 0); // amvr
        assert_eq!(r.read_bit(), 0); // bdof
        assert_eq!(r.read_bit(), 0); // smvd
        assert_eq!(r.read_bit(), 0); // dmvr
        assert_eq!(r.read_bit(), 0); // mmvd
        assert_eq!(r.read_ue(), 0); // six_minus_max_merge_cand
        assert_eq!(r.read_bit(), 0); // sbt
        assert_eq!(r.read_bit(), 0); // affine
        assert_eq!(r.read_bit(), 0); // bcw
        assert_eq!(r.read_bit(), 0); // ciip
        assert_eq!(r.read_bit(), 0); // gpm
        assert_eq!(r.read_ue(), 0); // log2_parallel_merge_level_minus2
        assert_eq!(r.read_bit(), 0); // isp
        assert_eq!(r.read_bit(), 0); // mrl
        assert_eq!(r.read_bit(), 0); // mip
        if !h.chroma.is_monochrome() {
            assert_eq!(r.read_bit(), 0); // cclm
        }
        if matches!(h.chroma, ChromaFormat::Yuv420) {
            assert_eq!(r.read_bit(), 0); // chroma_horizontal_collocated
            assert_eq!(r.read_bit(), 0); // chroma_vertical_collocated
        }
        assert_eq!(r.read_bit(), 0); // palette
        if matches!(h.chroma, ChromaFormat::Yuv444) && (1u32 << LOG2_CTU_SIZE) <= 32 {
            assert_eq!(r.read_bit(), 0); // act_enabled
        }
        assert_eq!(r.read_ue(), 0); // min_qp_prime_ts_minus4 (TS always enabled)
        assert_eq!(r.read_bit(), 0); // ibc
        assert_eq!(r.read_bit(), 0); // ladf
        assert_eq!(r.read_bit(), 0); // explicit_scaling_list
        assert_eq!(r.read_bit(), 0); // dep_quant
        assert_eq!(r.read_bit(), 0); // sign_data_hiding
        assert_eq!(r.read_bit(), 0); // virtual_boundaries
        assert_eq!(r.read_bit(), 0); // timing_hrd
        assert_eq!(r.read_bit(), 0); // field_seq
        assert_eq!(r.read_bit(), 1); // vui_present
        let vui_size = r.read_ue() + 1; // sps_vui_payload_size_minus1 + 1
        while r.bit_pos() & 7 != 0 {
            assert_eq!(r.read_bit(), 0); // sps_vui_alignment_zero_bit
        }
        // Skip the byte-aligned VUI payload and confirm it matches what we wrote.
        let vui_start = r.bit_pos() / 8;
        let expected = h.vui_payload();
        assert_eq!(
            vui_size as usize,
            expected.len(),
            "VUI payload size mismatch"
        );
        assert_eq!(
            &rbsp[vui_start..vui_start + expected.len()],
            &expected[..],
            "VUI bytes mismatch"
        );
        for _ in 0..vui_size * 8 {
            r.read_bit();
        }
        assert_eq!(r.read_bit(), 0); // extension_present
        r.bit_pos()
    }

    fn parse_pps(rbsp: &[u8], h: &Headers) -> usize {
        let mut r = BitReader::new(rbsp);
        assert_eq!(r.read_bits(6), 0); // pps_id
        assert_eq!(r.read_bits(4), 0); // sps_id
        assert_eq!(r.read_bit(), 0); // mixed_nalu
        assert_eq!(r.read_ue(), h.coded_width());
        assert_eq!(r.read_ue(), h.coded_height());
        assert_eq!(r.read_bit(), 0); // conformance_window_flag
        assert_eq!(r.read_bit(), 0); // scaling_window
        assert_eq!(r.read_bit(), 0); // output_flag_present
        assert_eq!(r.read_bit(), 1); // no_pic_partition
        assert_eq!(r.read_bit(), 0); // subpic_id_mapping
        assert_eq!(r.read_bit(), 0); // cabac_init_present
        assert_eq!(r.read_ue(), 0); // num_ref_idx_default[0]
        assert_eq!(r.read_ue(), 0); // num_ref_idx_default[1]
        assert_eq!(r.read_bit(), 0); // rpl1_idx_present
        assert_eq!(r.read_bit(), 0); // weighted_pred
        assert_eq!(r.read_bit(), 0); // weighted_bipred
        assert_eq!(r.read_bit(), 0); // ref_wraparound
        assert_eq!(
            r.read_se(),
            h.qp as i32 - 6 * (h.bit_depth.bits() as i32 - 8) - 26
        ); // init_qp_minus26
        assert_eq!(r.read_bit(), 0); // cu_qp_delta_enabled
        assert_eq!(r.read_bit(), 0); // chroma_tool_offsets_present
        assert_eq!(r.read_bit(), 1); // deblocking_filter_control_present
        assert_eq!(r.read_bit(), 0); // deblocking_filter_override_enabled
        assert_eq!(r.read_bit(), 1); // deblocking_filter_disabled
        assert_eq!(r.read_bit(), 0); // picture_header_extension_present
        assert_eq!(r.read_bit(), 0); // slice_header_extension_present
        assert_eq!(r.read_bit(), 0); // extension_flag
        r.bit_pos()
    }

    fn cases() -> Vec<Headers> {
        let mut v = Vec::new();
        for &chroma in &[
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            for &(w, h) in &[(64u32, 64u32), (640, 480), (97, 33), (1920, 1080)] {
                v.push(Headers {
                    width: w,
                    height: h,
                    chroma,
                    bit_depth: BitDepth::Eight,
                    qp: 32,
                    lossless: false,
                    aq: false,
                    mtt: false,
                    lfnst: false,
                    dep_quant: false,
                    mts: false,
                    dual_tree: false,
                    cclm: false,
                    deblock: false,
                });
            }
        }
        v.push(Headers {
            width: 256,
            height: 256,
            chroma: ChromaFormat::Yuv444,
            bit_depth: BitDepth::Ten,
            qp: 18,
            lossless: false,
            aq: false,
            mtt: false,
            lfnst: false,
            dep_quant: false,
            mts: false,
            dual_tree: false,
            cclm: false,
            deblock: false,
        });
        v
    }

    #[test]
    fn sps_round_trips_and_lands_on_stop_bit() {
        for h in cases() {
            let rbsp = h.write_sps_rbsp();
            let consumed = parse_sps(&rbsp, &h);
            check_lands_on_stop_bit(&rbsp, consumed);
        }
    }

    #[test]
    fn sps_cclm_flag_round_trips() {
        for cclm in [false, true] {
            let h = Headers {
                width: 128,
                height: 128,
                chroma: ChromaFormat::Yuv420,
                bit_depth: BitDepth::Eight,
                qp: 30,
                lossless: false,
                aq: false,
                mtt: false,
                lfnst: false,
                dep_quant: false,
                mts: false,
                dual_tree: true,
                cclm,
                deblock: false,
            };
            let parsed = super::parse_sps(&h.write_sps_rbsp()).unwrap();
            assert_eq!(parsed.cclm, cclm, "cclm flag mismatch");
        }
    }

    #[test]
    fn sps_dual_tree_flag_round_trips() {
        // The sps_qtbtt_dual_tree_intra_flag must survive write+parse for both
        // states (and is absent/false for monochrome).
        for &dt in &[false, true] {
            let h = Headers {
                width: 128,
                height: 128,
                chroma: ChromaFormat::Yuv420,
                bit_depth: BitDepth::Eight,
                qp: 32,
                lossless: false,
                aq: false,
                mtt: false,
                lfnst: false,
                dep_quant: false,
                mts: false,
                dual_tree: dt,
                cclm: false,
                deblock: false,
            };
            let rbsp = h.write_sps_rbsp();
            let parsed = super::parse_sps(&rbsp).unwrap();
            assert_eq!(parsed.dual_tree, dt, "dual_tree flag mismatch");
        }
        // Monochrome: the flag is not present and parses as false.
        let hm = Headers {
            width: 128,
            height: 128,
            chroma: ChromaFormat::Monochrome,
            bit_depth: BitDepth::Eight,
            qp: 32,
            lossless: false,
            aq: false,
            mtt: false,
            lfnst: false,
            dep_quant: false,
            mts: false,
            dual_tree: false,
            cclm: false,
            deblock: false,
        };
        assert!(!super::parse_sps(&hm.write_sps_rbsp()).unwrap().dual_tree);
    }

    #[test]
    fn pps_round_trips_and_lands_on_stop_bit() {
        for h in cases() {
            let rbsp = h.write_pps_rbsp();
            let consumed = parse_pps(&rbsp, &h);
            check_lands_on_stop_bit(&rbsp, consumed);
        }
    }

    #[test]
    fn conformance_window_set_when_padding_needed() {
        let h = Headers {
            width: 97,
            height: 33,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            qp: 30,
            lossless: false,
            aq: false,
            mtt: false,
            lfnst: false,
            dep_quant: false,
            mts: false,
            dual_tree: false,
            cclm: false,
            deblock: false,
        };
        assert_eq!(h.coded_width(), 104);
        assert_eq!(h.coded_height(), 40);
        // round-trip already asserts the window fields; just confirm padding math.
        assert_eq!(
            (h.coded_width() - h.width) / h.chroma.sub_w() as u32,
            (104 - 97) / 2
        );
    }

    #[test]
    fn subsampled_odd_dimensions_retain_one_edge_sample() {
        let h = Headers {
            width: 1777,
            height: 777,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            qp: 30,
            lossless: false,
            aq: false,
            mtt: false,
            lfnst: false,
            dep_quant: false,
            mts: false,
            dual_tree: false,
            cclm: false,
            deblock: false,
        };
        assert_eq!((h.coded_width(), h.coded_height()), (1784, 784));
        let parsed = super::parse_sps(&h.write_sps_rbsp()).unwrap();
        assert_eq!((parsed.width, parsed.height), (1778, 778));
    }

    #[test]
    fn nal_framing_has_start_code_and_header() {
        let h = Headers {
            width: 64,
            height: 64,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            qp: 32,
            lossless: false,
            aq: false,
            mtt: false,
            lfnst: false,
            dep_quant: false,
            mts: false,
            dual_tree: false,
            cclm: false,
            deblock: false,
        };
        let nal = write_nal(NalUnitType::Sps, &h.write_sps_rbsp(), 1);
        // SPS nuh_unit_type = 15.
        let ebsp_first = rbsp_to_ebsp(&h.write_sps_rbsp())[0];
        assert_eq!(nal[2], ebsp_first); // payload starts after 2-byte header
        let stream = h.write_parameter_set_nals();
        assert_eq!(&stream[0..4], &[0, 0, 0, 1]); // Annex-B start code
    }
}

#[cfg(test)]
mod slice_tests {
    use super::*;
    use crate::bitstream::BitReader;

    /// Parse picture header + slice header the way a decoder would, returning the
    /// reader positioned after byte alignment, and asserting field values.
    fn parse_slice_header(rbsp: &[u8], h: &Headers) -> usize {
        let mut r = BitReader::new(rbsp);
        assert_eq!(r.read_bit(), 1); // sh_picture_header_in_slice_header_flag
        // picture_header_structure
        assert_eq!(r.read_bit(), 1); // ph_gdr_or_irap_pic_flag
        assert_eq!(r.read_bit(), 0); // ph_non_ref_pic_flag
        assert_eq!(r.read_bit(), 0); // ph_gdr_pic_flag
        assert_eq!(r.read_bit(), 0); // ph_inter_slice_allowed_flag
        assert_eq!(r.read_ue(), 0); // ph_pic_parameter_set_id
        assert_eq!(r.read_bits(4), 0); // ph_pic_order_cnt_lsb
        // slice header continues
        assert_eq!(r.read_bit(), 0); // sh_no_output_of_prior_pics_flag
        assert_eq!(r.read_se(), 0); // sh_qp_delta
        assert_eq!(r.read_bit(), 0); // sh_ts_residual_coding_disabled_flag
        // byte alignment: a 1 bit then zeros to the byte boundary.
        assert_eq!(r.read_bit(), 1); // alignment_bit_equal_to_one
        let _ = h;
        while r.bit_pos() & 7 != 0 {
            assert_eq!(r.read_bit(), 0); // alignment_bit_equal_to_zero
        }
        r.bit_pos()
    }

    fn case() -> Headers {
        Headers {
            width: 320,
            height: 240,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            qp: 32,
            lossless: false,
            aq: false,
            mtt: false,
            lfnst: false,
            dep_quant: false,
            mts: false,
            dual_tree: false,
            cclm: false,
            deblock: false,
        }
    }

    #[test]
    fn slice_header_round_trips_and_byte_aligns() {
        for &chroma in &[
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv444,
        ] {
            let h = Headers { chroma, ..case() };
            // Build just the slice header (empty slice_data) and check alignment.
            let mut w = crate::bitstream::BitWriter::new();
            h.write_slice_header(&mut w);
            let bytes = w.into_bytes();
            let consumed = parse_slice_header(&bytes, &h);
            assert_eq!(consumed % 8, 0, "slice header must end byte-aligned");
            assert_eq!(consumed, bytes.len() * 8, "no trailing bytes expected");
        }
    }

    #[test]
    fn idr_nal_has_correct_type_and_carries_payload() {
        let h = case();
        let payload = [0xAA, 0xBB, 0xCC];
        let nal = h.write_idr_slice_nal(&payload);
        // nuh_unit_type for IDR_N_LP = 8: byte1 = (8<<3)|tid+1 = 0x41.
        assert_eq!(nal[1], 0x41);
        // The payload must appear at the tail (no emulation bytes triggered here).
        assert_eq!(&nal[nal.len() - 3..], &payload);
    }

    #[test]
    fn full_still_picture_stream_structure() {
        let h = case();
        let payload = [0x12, 0x34, 0x56, 0x78];
        let stream = h.write_still_picture(&payload);
        // Three Annex-B start codes: SPS, PPS, IDR slice.
        let starts = stream.windows(4).filter(|w| *w == [0, 0, 0, 1]).count();
        assert_eq!(starts, 3, "expected SPS + PPS + slice start codes");
        assert_eq!(&stream[0..4], &[0, 0, 0, 1]);
    }
}
