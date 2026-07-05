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
#![allow(clippy::too_many_arguments)]
#![deny(unreachable_pub)]

mod bitstream;
mod cabac;

#[allow(dead_code)]
mod cclm;
mod color;
mod deblock;
mod decode;
mod depquant;
mod encode;
mod error;
mod fmt;
mod headers;
mod intra;
mod isobmff;
mod lfnst;
mod lfnst_tables;
mod metadata;
mod partition;
mod predict;
mod residual;
mod residual_ts;
mod transform;
mod tu;
mod yuv;

pub use color::{Cicp, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
pub use decode::{DecodedImage, PlaneView, decode, decode_266, decode_with_alpha, decode_yuv_266};
pub use error::EncodeError;
pub use fmt::{BitDepth, ChromaFormat};
pub use metadata::{ContentLightLevel, ImageMetadata, Orientation};
pub use yuv::Yuv;

const MIN_DIM: u32 = 1;
// VVC Level 6.2 picture limits (ITU-T H.266 Table A.1 / §A.4.2). garnetash
// declares Level 6.2 (`general_level_idc` = 102) in every SPS, so the encoder
// must not emit — and the decoder need not accept — a picture larger than this
// level permits. The limits apply to the *coded* luma picture (display size
// padded up to a multiple of 8):
//   MaxLumaPs                = 35 651 584 samples   (≈ 8192×4352)
//   max width / height       = floor(sqrt(8 · MaxLumaPs)) = 16 888
const MAX_LUMA_SAMPLES: u64 = 35_651_584;
const MAX_DIM: u32 = 16_888;

/// VVC's largest coding tree unit is 128×128 luma samples (vs HEVC's 64×64).
#[allow(dead_code)]
pub(crate) const MAX_CTU_SIZE: u32 = 128;

/// Encoder configuration shared by all entry points.
///
/// Build with [`EncodeConfig::new`] and the `with_*` builder methods.
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    /// Visual quality 1..=100 (higher = better, larger file). M
    pub quality: u8,
    /// Mathematically lossless encoding.
    pub lossless: bool,
    /// Chroma subsampling format.
    pub chroma: ChromaFormat,
    /// Sample bit depth.
    pub bit_depth: BitDepth,
    pub color: ColorMetadata,
    pub metadata: ImageMetadata,
    pub threads: usize,
    /// Rate-distortion optimized quantization
    pub rdoq: bool,
    /// Perceptual adaptive quantization.
    pub aq: bool,
    /// Multi-type-tree (QTBTT) partitioning
    pub mtt: bool,
    /// Enable LFNST (low-frequency non-separable secondary transform)
    pub lfnst: bool,
    /// Enable dependent quantization (VVC `sps_dep_quant_enabled_flag`)
    pub dep_quant: bool,
    /// Enable explicit MTS (DST-VII/DCT-VIII per-TU selection) for intra luma.
    pub mts: bool,
    /// Enable separate luma/chroma intra coding trees (VVC dual tree).
    pub dual_tree: bool,
    /// Enable CCLM cross-component chroma prediction (requires the dual tree).
    pub cclm: bool,
    /// Enable the in-loop deblocking filter (8-bit, single-tree, AQ off).
    pub deblock: bool,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig {
            quality: 90,
            lossless: false,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            color: ColorMetadata::default(),
            metadata: ImageMetadata::default(),
            threads: 1,
            rdoq: true,
            aq: true,
            mtt: true,
            lfnst: true,
            dep_quant: true,
            mts: true,
            dual_tree: true,
            cclm: true,
            deblock: true,
        }
    }
}

impl EncodeConfig {
    /// Default settings: q = 90, 4:2:0, 8-bit.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Enable mathematically lossless encoding. In this mode
    /// [`quality`](Self::quality) is ignored.
    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Select the preferred coded chroma format. For an exact raw VVC display
    /// extent, odd dimensions promote 4:2:0 to 4:2:2/4:4:4 as needed because
    /// subsampled conformance-window offsets cannot crop a single luma sample.
    pub fn with_chroma(mut self, chroma: ChromaFormat) -> Self {
        self.chroma = chroma;
        self
    }

    pub fn with_bit_depth(mut self, bit_depth: BitDepth) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Toggle rate-distortion optimized quantization (see [`rdoq`](Self::rdoq)).
    /// Enabled by default; disable for byte-for-byte comparison against plain
    /// scalar quantization, or if a perceptual A/B favours it off.
    pub fn with_rdoq(mut self, rdoq: bool) -> Self {
        self.rdoq = rdoq;
        self
    }

    /// Toggle perceptual adaptive quantization (see [`aq`](Self::aq)). Off by
    /// default; enable to spend bits where the eye notices (flat regions) and
    /// save them where masking hides distortion (busy regions).
    pub fn with_aq(mut self, aq: bool) -> Self {
        self.aq = aq;
        self
    }

    /// Enable multi-type-tree (binary/ternary) partitioning. See [`Self::mtt`].
    pub fn with_mtt(mut self, mtt: bool) -> Self {
        self.mtt = mtt;
        self
    }

    /// Enable LFNST. See [`Self::lfnst`].
    pub fn with_lfnst(mut self, lfnst: bool) -> Self {
        self.lfnst = lfnst;
        self
    }

    /// Enable dependent quantization. See [`Self::dep_quant`].
    pub fn with_dep_quant(mut self, dep_quant: bool) -> Self {
        self.dep_quant = dep_quant;
        self
    }

    /// Enable explicit MTS. See [`Self::mts`].
    pub fn with_mts(mut self, mts: bool) -> Self {
        self.mts = mts;
        self
    }

    /// Enable the intra dual tree. See [`Self::dual_tree`].
    pub fn with_dual_tree(mut self, dual_tree: bool) -> Self {
        self.dual_tree = dual_tree;
        self
    }

    /// Enable CCLM. See [`Self::cclm`].
    pub fn with_cclm(mut self, cclm: bool) -> Self {
        self.cclm = cclm;
        self
    }

    /// Enable the deblocking filter. See [`Self::deblock`].
    pub fn with_deblocking(mut self, deblock: bool) -> Self {
        self.deblock = deblock;
        self
    }

    /// Set the number of encoder worker threads (`0`/`1` = single-threaded). The
    /// coded bitstream is identical regardless of this value.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Set the full color description (`colr` property) for HEIF output.
    pub fn with_color(mut self, color: ColorMetadata) -> Self {
        self.color = color;
        self
    }

    /// Set the CICP color code points for HEIF output (an `nclx` `colr` box).
    pub fn with_cicp(mut self, cicp: Cicp) -> Self {
        self.color.cicp = Some(cicp);
        self
    }

    /// Embed an ICC profile (`prof` `colr` box) in HEIF output.
    pub fn with_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.color.icc = Some(icc);
        self
    }

    /// Set the full image-metadata bundle (orientation / CLLI / Exif) for HEIF.
    pub fn with_metadata(mut self, metadata: ImageMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the display orientation (`irot` / `imir`) for HEIF output.
    pub fn with_orientation(mut self, o: Orientation) -> Self {
        self.metadata.orientation = o;
        self
    }

    /// Set the HDR content light level (`clli`) for HEIF output.
    pub fn with_content_light_level(mut self, cll: ContentLightLevel) -> Self {
        self.metadata.content_light_level = Some(cll);
        self
    }

    /// Attach a raw Exif payload (a separate `Exif` item) to HEIF output.
    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.metadata.exif = Some(exif);
        self
    }

    /// Map `quality` (1..=100) to a luma QP. VVC QP range is 0..=63 for 8-bit
    /// (0..=63 + QpBdOffset internally). Higher quality -> lower QP.
    #[allow(dead_code)]
    pub(crate) fn luma_qp(&self) -> u8 {
        // Linear map: q=100 -> QP 4, q=1 -> QP 51. Clamped into VVC's 8-bit range.
        let q = self.quality.clamp(1, 100) as i32;
        let qp = 51 - ((q - 1) * 47) / 99;
        qp.clamp(0, 63) as u8
    }

    fn validate(&self) -> Result<(), EncodeError> {
        validate_quality(self.quality)?;
        Ok(())
    }
}

// ── validation helpers ──────────────────────────────────────────────────────

fn validate_quality(q: u8) -> Result<(), EncodeError> {
    if (1..=100).contains(&q) {
        Ok(())
    } else {
        Err(EncodeError::InvalidQuality(q))
    }
}

fn validate_dims(width: u32, height: u32) -> Result<(), EncodeError> {
    if width < MIN_DIM || height < MIN_DIM {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    // Level limits constrain the coded picture (luma padded to a multiple of 8),
    // not the display size. Reject anything beyond the declared Level 6.2 so the
    // encoder never produces a stream a conformant decoder would refuse.
    let cw = (width as u64 + 7) & !7;
    let ch = (height as u64 + 7) & !7;
    if cw > MAX_DIM as u64 || ch > MAX_DIM as u64 || cw * ch > MAX_LUMA_SAMPLES {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    Ok(())
}

fn validate_buf_u8(
    buf: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<(), EncodeError> {
    let expected = width as usize * height as usize * channels;
    if buf.len() == expected {
        Ok(())
    } else {
        Err(EncodeError::BufferSize {
            expected,
            found: buf.len(),
        })
    }
}

use crate::residual_ts::LOSSLESS_QP;

pub fn encode_rgb_266(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgb, width, height, 3)?;
    let bd = cfg.bit_depth.bits();
    let qp = if cfg.lossless {
        LOSSLESS_QP
    } else {
        cfg.luma_qp()
    };
    Ok(encode::encode_still(
        rgb,
        width,
        height,
        qp,
        bd,
        3,
        cfg.lossless,
        cfg.chroma,
        cfg.threads,
        cfg.rdoq,
        cfg.aq,
        cfg.mtt,
        cfg.lfnst,
        cfg.dep_quant,
        cfg.mts,
        cfg.dual_tree,
        cfg.cclm,
        cfg.deblock,
    ))
}

/// Encode a packed 8-bit RGBA image to a VVC still picture. Alpha is discarded.
///
/// `rgba` must hold exactly `width * height * 4` bytes in R, G, B, A order.
///
/// Uses the configured chroma format (see [`encode_rgb_266`]); odd dimensions
/// may promote subsampled chroma so the raw VVC display extent remains exact.
pub fn encode_rgba_266(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgba, width, height, 4)?;
    let bd = cfg.bit_depth.bits();
    let qp = if cfg.lossless {
        LOSSLESS_QP
    } else {
        cfg.luma_qp()
    };
    Ok(encode::encode_still(
        rgba,
        width,
        height,
        qp,
        bd,
        4,
        cfg.lossless,
        cfg.chroma,
        cfg.threads,
        cfg.rdoq,
        cfg.aq,
        cfg.mtt,
        cfg.lfnst,
        cfg.dep_quant,
        cfg.mts,
        cfg.dual_tree,
        cfg.cclm,
        cfg.deblock,
    ))
}

/// Encode RGB and also return the encoder's reconstruction as cropped planar
/// I420 (Y, Cb, Cr). A conformant VVC decoder must reproduce the second buffer
/// exactly; intended for validating garnetash against a reference decoder.
#[doc(hidden)]
pub fn encode_rgb_with_reconstruction(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<(Vec<u8>, Vec<u8>), EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgb, width, height, 3)?;
    let bd = cfg.bit_depth.bits();
    let qp = if cfg.lossless {
        LOSSLESS_QP
    } else {
        cfg.luma_qp()
    };
    Ok(encode::encode_with_recon(
        rgb,
        width,
        height,
        qp,
        bd,
        3,
        cfg.lossless,
        cfg.chroma,
        cfg.rdoq,
        cfg.aq,
        cfg.mtt,
        cfg.lfnst,
        cfg.dep_quant,
        cfg.mts,
        cfg.dual_tree,
        cfg.cclm,
        cfg.deblock,
    ))
}

/// Encode RGB and wrap the result in a HEIF container (ISO/IEC 23008-12), the
/// file form image viewers recognise. The primary item is a single VVC still
/// image (`vvc1` / `vvcC`); see [`encode_rgb_266`] for the encoding parameters.
///
/// `rgb` must hold exactly `width * height * 3` bytes in R, G, B order. Returns
/// the complete `.heif` file bytes.
pub fn encode_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let annexb = encode_rgb_266(rgb, width, height, cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode 8-bit RGBA (alpha discarded) and wrap the result in a HEIF container.
/// See [`encode_rgb`] and [`encode_rgba_266`].
pub fn encode_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let annexb = encode_rgba_266(rgba, width, height, cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode a packed planar YCbCr image directly to a VVC still
pub fn encode_yuv_266(
    planes: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    let bd = cfg.bit_depth.bits();
    let qp = if cfg.lossless {
        LOSSLESS_QP
    } else {
        cfg.luma_qp()
    };
    encode::encode_still_yuv(
        planes,
        width,
        height,
        qp,
        bd,
        cfg.lossless,
        cfg.chroma,
        cfg.threads,
        cfg.rdoq,
        cfg.aq,
        cfg.mtt,
        cfg.lfnst,
        cfg.dep_quant,
        cfg.mts,
        cfg.dual_tree,
        cfg.cclm,
        cfg.deblock,
    )
}

/// Encode packed planar YCbCr (see [`encode_yuv_266`]) and wrap it in a HEIF file.
pub fn encode_yuv(
    planes: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let annexb = encode_yuv_266(planes, width, height, cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode 8-bit RGBA to a HEIF file
pub fn encode_rgba_with_alpha(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgba, width, height, 4)?;
    if !matches!(cfg.bit_depth, BitDepth::Eight) {
        return Err(EncodeError::Unsupported(
            "alpha-from-RGBA requires an 8-bit configuration",
        ));
    }
    // color image: RGB (alpha discarded by the 4-stride RGB path).
    let master = encode_rgba_266(rgba, width, height, cfg)?;
    // Alpha image: the A channel as a monochrome plane (Y = alpha, exact).
    let alpha_plane: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
    let mono_cfg = cfg.clone().with_chroma(ChromaFormat::Monochrome);
    let alpha = encode_yuv_266(&alpha_plane, width, height, &mono_cfg)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
    )
}

#[inline]
fn force_bd(cfg: &EncodeConfig, bd: BitDepth) -> EncodeConfig {
    cfg.clone().with_bit_depth(bd)
}

fn validate_buf_u16(
    buf: &[u16],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<(), EncodeError> {
    if buf.len() == width as usize * height as usize * channels {
        Ok(())
    } else {
        Err(EncodeError::Unsupported(
            "buffer length does not match width × height × channels",
        ))
    }
}

/// Samples must fit the coded bit depth: `0..=(2^bits − 1)`.
fn validate_range_wide(buf: &[u16], bd: BitDepth) -> Result<(), EncodeError> {
    let max = (1u16 << bd.bits()) - 1;
    if buf.iter().all(|&v| v <= max) {
        Ok(())
    } else {
        Err(EncodeError::Unsupported(
            "sample out of range for the coded bit depth",
        ))
    }
}

/// Encode native high-bit-depth RGB(A) (`u16`, `stride_px` channels) to a raw
/// VVC stream at the config's bit depth (no up-scaling; samples are already at
/// the coded depth).
fn encode_wide_stream(
    rgb: &[u16],
    width: u32,
    height: u32,
    stride_px: usize,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgb, width, height, stride_px)?;
    validate_range_wide(rgb, cfg.bit_depth)?;
    let qp = if cfg.lossless {
        LOSSLESS_QP
    } else {
        cfg.luma_qp()
    };
    Ok(encode::encode_still_wide(
        rgb,
        width,
        height,
        qp,
        cfg.bit_depth.bits(),
        stride_px,
        cfg.lossless,
        cfg.chroma,
        cfg.threads,
        cfg.rdoq,
        cfg.aq,
        cfg.mtt,
        cfg.lfnst,
        cfg.dep_quant,
        cfg.mts,
        cfg.dual_tree,
        cfg.cclm,
        cfg.deblock,
    ))
}

/// Encode a `u16` alpha plane (`width × height`) as a monochrome VVC stream at
/// the config's bit depth (`Y == alpha`, exact when lossless).
fn encode_alpha_wide_stream(
    alpha: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_range_wide(alpha, cfg.bit_depth)?;
    let le: Vec<u8> = alpha.iter().flat_map(|&v| v.to_le_bytes()).collect();
    encode_yuv_266(
        &le,
        width,
        height,
        &cfg.clone().with_chroma(ChromaFormat::Monochrome),
    )
}

/// Encode packed 8-bit RGB to a raw VVC stream (see [`encode_rgb_266`]).
pub fn encode_rgb8_266(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_rgb_266(rgb, width, height, &force_bd(cfg, BitDepth::Eight))
}

/// Encode packed 10-bit RGB (`u16`, `0..=1023`) to a raw VVC stream.
pub fn encode_rgb10_266(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_wide_stream(rgb, width, height, 3, &force_bd(cfg, BitDepth::Ten))
}

/// Encode packed 8-bit RGBA (alpha discarded) to a raw VVC stream.
pub fn encode_rgba8_266(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_rgba_266(rgba, width, height, &force_bd(cfg, BitDepth::Eight))
}

/// Encode packed 10-bit RGBA (`u16`, alpha discarded) to a raw VVC stream.
pub fn encode_rgba10_266(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_wide_stream(rgba, width, height, 4, &force_bd(cfg, BitDepth::Ten))
}

/// Encode packed 8-bit RGBA to a HEIF file, preserving alpha (see
/// [`encode_rgba_with_alpha`]).
pub fn encode_rgba8_with_alpha(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_rgba_with_alpha(rgba, width, height, &force_bd(cfg, BitDepth::Eight))
}

/// Encode packed 10-bit RGBA (`u16`) to a HEIF file, preserving alpha as a
/// 10-bit monochrome auxiliary image.
pub fn encode_rgba10_with_alpha(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Ten);
    let master = encode_wide_stream(rgba, width, height, 4, &cfg)?;
    let alpha_plane: Vec<u16> = rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
    let alpha = encode_alpha_wide_stream(&alpha_plane, width, height, &cfg)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
    )
}

/// Encode an 8-bit grayscale image to a raw monochrome VVC stream.
pub fn encode_gray8_266(
    gray: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_gray_266(gray, width, height, &force_bd(cfg, BitDepth::Eight))
}

/// Encode a 10-bit grayscale image (`u16`, `0..=1023`) to a raw monochrome stream.
pub fn encode_gray10_266(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u16(gray, width, height, 1)?;
    validate_range_wide(gray, BitDepth::Ten)?;
    let le: Vec<u8> = gray.iter().flat_map(|&v| v.to_le_bytes()).collect();
    encode_yuv_266(
        &le,
        width,
        height,
        &force_bd(
            &cfg.clone().with_chroma(ChromaFormat::Monochrome),
            BitDepth::Ten,
        ),
    )
}

/// Encode an 8-bit interleaved grayscale+alpha image (Y, A bytes) to a raw
/// monochrome stream. **Alpha is discarded** (only the luma is coded).
pub fn encode_gray_alpha8_266(
    ya: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u8(ya, width, height, 2)?;
    let gray: Vec<u8> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
    encode_gray_266(&gray, width, height, &force_bd(cfg, BitDepth::Eight))
}

/// Encode a 10-bit interleaved grayscale+alpha image (`u16` Y, A pairs) to a raw
/// monochrome stream. Alpha is discarded.
pub fn encode_gray_alpha10_266(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u16(ya, width, height, 2)?;
    let gray: Vec<u16> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
    encode_gray10_266(&gray, width, height, cfg)
}

/// Encode packed planar 8-bit YCbCr to a raw VVC stream (see [`encode_yuv_266`]).
pub fn encode_yuv8_266(
    planes: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_yuv_266(planes, width, height, &force_bd(cfg, BitDepth::Eight))
}

/// Encode planar 10-bit YCbCr (`u16` samples in plane order Y, Cb, Cr) to a raw
/// VVC stream.
pub fn encode_yuv10_266(
    planes: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_range_wide(planes, BitDepth::Ten)?;
    let le: Vec<u8> = planes.iter().flat_map(|&v| v.to_le_bytes()).collect();
    encode_yuv_266(&le, width, height, &force_bd(cfg, BitDepth::Ten))
}

/// Encode planar 8-bit YCbCr plus a separate 8-bit alpha plane (`width × height`)
/// to a HEIF file, preserving alpha as a monochrome auxiliary image.
pub fn encode_yuva8_with_alpha(
    planes: &[u8],
    alpha: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Eight);
    let master = encode_yuv_266(planes, width, height, &cfg)?;
    let alpha_stream = encode_yuv_266(
        alpha,
        width,
        height,
        &cfg.clone().with_chroma(ChromaFormat::Monochrome),
    )?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha_stream,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
    )
}

/// Encode planar 10-bit YCbCr (`u16`) plus a separate 10-bit alpha plane to a
/// HEIF file, preserving alpha as a 10-bit monochrome auxiliary image.
pub fn encode_yuva10_with_alpha(
    planes: &[u16],
    alpha: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Ten);
    validate_range_wide(planes, BitDepth::Ten)?;
    validate_range_wide(alpha, BitDepth::Ten)?;
    let master = encode_yuv10_266(planes, width, height, &cfg)?;
    let alpha_stream = encode_alpha_wide_stream(alpha, width, height, &cfg)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha_stream,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
    )
}

/// Encode packed 12-bit RGB (`u16`, `0..=4095`) to a raw VVC stream.
pub fn encode_rgb12_266(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_wide_stream(rgb, width, height, 3, &force_bd(cfg, BitDepth::Twelve))
}

/// Encode packed 12-bit RGBA (`u16`, alpha discarded) to a raw VVC stream.
pub fn encode_rgba12_266(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_wide_stream(rgba, width, height, 4, &force_bd(cfg, BitDepth::Twelve))
}

/// Encode packed 12-bit RGBA (`u16`) to a HEIF file, preserving alpha as a
/// 12-bit monochrome auxiliary image.
pub fn encode_rgba12_with_alpha(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Twelve);
    let master = encode_wide_stream(rgba, width, height, 4, &cfg)?;
    let alpha_plane: Vec<u16> = rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
    let alpha = encode_alpha_wide_stream(&alpha_plane, width, height, &cfg)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
    )
}

/// Encode a 12-bit grayscale image (`u16`, `0..=4095`) to a raw monochrome stream.
pub fn encode_gray12_266(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u16(gray, width, height, 1)?;
    validate_range_wide(gray, BitDepth::Twelve)?;
    let le: Vec<u8> = gray.iter().flat_map(|&v| v.to_le_bytes()).collect();
    encode_yuv_266(
        &le,
        width,
        height,
        &force_bd(
            &cfg.clone().with_chroma(ChromaFormat::Monochrome),
            BitDepth::Twelve,
        ),
    )
}

/// Encode a 12-bit interleaved grayscale+alpha image (`u16` Y, A pairs) to a raw
/// monochrome stream. Alpha is discarded.
pub fn encode_gray_alpha12_266(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u16(ya, width, height, 2)?;
    let gray: Vec<u16> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
    encode_gray12_266(&gray, width, height, cfg)
}

/// Encode planar 12-bit YCbCr (`u16` samples in plane order Y, Cb, Cr) to a raw
/// VVC stream.
pub fn encode_yuv12_266(
    planes: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_range_wide(planes, BitDepth::Twelve)?;
    let le: Vec<u8> = planes.iter().flat_map(|&v| v.to_le_bytes()).collect();
    encode_yuv_266(&le, width, height, &force_bd(cfg, BitDepth::Twelve))
}

/// Encode planar 12-bit YCbCr (`u16`) plus a separate 12-bit alpha plane to a
/// HEIF file, preserving alpha as a 12-bit monochrome auxiliary image.
pub fn encode_yuva12_with_alpha(
    planes: &[u16],
    alpha: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Twelve);
    validate_range_wide(planes, BitDepth::Twelve)?;
    let master = encode_yuv12_266(planes, width, height, &cfg)?;
    let alpha_stream = encode_alpha_wide_stream(alpha, width, height, &cfg)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha_stream,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
    )
}

/// Encode a single-channel 8-bit grayscale image as a monochrome (4:0:0) VVC
/// still picture. `gray` must hold exactly `width * height` bytes.
///
/// The luma plane is taken directly: because the RGB→Y matrix coefficients sum
/// to one, a pixel with `R = G = B = v` yields `Y = v` exactly, so the stored
/// luma equals the input. The chroma format in `cfg` is overridden to
/// monochrome.
pub fn encode_gray_266(
    gray: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(gray, width, height, 1)?;
    let mut rgb = vec![0u8; gray.len() * 3];
    for (i, &v) in gray.iter().enumerate() {
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }
    let mono = cfg.clone().with_chroma(ChromaFormat::Monochrome);
    encode_rgb_266(&rgb, width, height, &mono)
}

/// Encode grayscale and wrap it in a HEIF container. See [`encode_gray_266`].
pub fn encode_gray(
    gray: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let mono = cfg.clone().with_chroma(ChromaFormat::Monochrome);
    let annexb = encode_gray_266(gray, width, height, cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        mono.bit_depth,
        ChromaFormat::Monochrome,
        &mono.color,
        &mono.metadata,
    )
}

/// Wrap an already-encoded raw Annex-B VVC still stream (SPS + PPS + IDR slice,
pub fn wrap_vvc_in_heif(
    annexb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    isobmff::wrap_vvc_still(
        annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode packed 8-bit RGB to a HEIF file (see [`encode_rgb8_266`]).
pub fn encode_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Eight);
    let annexb = encode_rgb8_266(rgb, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode packed 10-bit RGB (`u16`, `0..=1023`) to a HEIF file.
pub fn encode_rgb10(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Ten);
    let annexb = encode_rgb10_266(rgb, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode packed 12-bit RGB (`u16`, `0..=4095`) to a HEIF file.
pub fn encode_rgb12(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Twelve);
    let annexb = encode_rgb12_266(rgb, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode packed 8-bit RGBA (alpha discarded) to a HEIF file.
pub fn encode_rgba8(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Eight);
    let annexb = encode_rgba8_266(rgba, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode packed 10-bit RGBA (`u16`, alpha discarded) to a HEIF file.
pub fn encode_rgba10(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Ten);
    let annexb = encode_rgba10_266(rgba, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode packed 12-bit RGBA (`u16`, alpha discarded) to a HEIF file.
pub fn encode_rgba12(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Twelve);
    let annexb = encode_rgba12_266(rgba, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode an 8-bit grayscale image to a monochrome HEIF file (see [`encode_gray8_266`]).
pub fn encode_gray8(
    gray: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Eight);
    let annexb = encode_gray8_266(gray, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        ChromaFormat::Monochrome,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode a 10-bit grayscale image (`u16`, `0..=1023`) to a monochrome HEIF file.
pub fn encode_gray10(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Ten);
    let annexb = encode_gray10_266(gray, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        ChromaFormat::Monochrome,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode a 12-bit grayscale image (`u16`, `0..=4095`) to a monochrome HEIF file.
pub fn encode_gray12(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Twelve);
    let annexb = encode_gray12_266(gray, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        ChromaFormat::Monochrome,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode planar 8-bit YCbCr to a HEIF file (see [`encode_yuv8_266`]).
pub fn encode_yuv8(
    planes: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Eight);
    let annexb = encode_yuv8_266(planes, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode planar 10-bit YCbCr (`u16`, plane order Y, Cb, Cr) to a HEIF file.
pub fn encode_yuv10(
    planes: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Ten);
    let annexb = encode_yuv10_266(planes, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode planar 12-bit YCbCr (`u16`, plane order Y, Cb, Cr) to a HEIF file.
pub fn encode_yuv12(
    planes: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = force_bd(cfg, BitDepth::Twelve);
    let annexb = encode_yuv12_266(planes, width, height, &cfg)?;
    isobmff::wrap_vvc_still(
        &annexb,
        width,
        height,
        cfg.bit_depth,
        cfg.chroma,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode an 8-bit interleaved grayscale+alpha image (`Y, A` bytes) to a HEIF
/// file, preserving alpha as a monochrome auxiliary image. `ya` must hold
/// exactly `width * height * 2` bytes.
pub fn encode_gray_alpha8(
    ya: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u8(ya, width, height, 2)?;
    let cfg = force_bd(cfg, BitDepth::Eight);
    let gray: Vec<u8> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
    let alpha: Vec<u8> = ya.as_chunks::<2>().0.iter().map(|p| p[1]).collect();
    let mono = cfg.clone().with_chroma(ChromaFormat::Monochrome);
    let master = encode_gray_266(&gray, width, height, &cfg)?;
    let alpha_stream = encode_yuv_266(&alpha, width, height, &mono)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha_stream,
        width,
        height,
        cfg.bit_depth,
        ChromaFormat::Monochrome,
        &cfg.color,
    )
}

/// Encode a 10-bit interleaved grayscale+alpha image (`u16` `Y, A` pairs) to a
/// HEIF file, preserving alpha as a 10-bit monochrome auxiliary image.
pub fn encode_gray_alpha10(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u16(ya, width, height, 2)?;
    let cfg = force_bd(cfg, BitDepth::Ten);
    let gray: Vec<u16> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
    let alpha: Vec<u16> = ya.as_chunks::<2>().0.iter().map(|p| p[1]).collect();
    let master = encode_gray10_266(&gray, width, height, &cfg)?;
    let alpha_stream = encode_alpha_wide_stream(&alpha, width, height, &cfg)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha_stream,
        width,
        height,
        cfg.bit_depth,
        ChromaFormat::Monochrome,
        &cfg.color,
    )
}

/// Encode a 12-bit interleaved grayscale+alpha image (`u16` `Y, A` pairs) to a
/// HEIF file, preserving alpha as a 12-bit monochrome auxiliary image.
pub fn encode_gray_alpha12(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_buf_u16(ya, width, height, 2)?;
    let cfg = force_bd(cfg, BitDepth::Twelve);
    let gray: Vec<u16> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
    let alpha: Vec<u16> = ya.as_chunks::<2>().0.iter().map(|p| p[1]).collect();
    let master = encode_gray12_266(&gray, width, height, &cfg)?;
    let alpha_stream = encode_alpha_wide_stream(&alpha, width, height, &cfg)?;
    isobmff::wrap_vvc_still_with_alpha(
        &master,
        &alpha_stream,
        width,
        height,
        cfg.bit_depth,
        ChromaFormat::Monochrome,
        &cfg.color,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_12bit_helpers_round_trip() {
        let (w, h) = (16u32, 16u32);
        // Native 12-bit RGB decodes as Twelve at the requested format.
        let rgb: Vec<u16> = (0..w * h * 3).map(|i| ((i * 13) % 4096) as u16).collect();
        let img = decode_266(
            &encode_rgb12_266(
                &rgb,
                w,
                h,
                &EncodeConfig::new()
                    .with_quality(85)
                    .with_chroma(ChromaFormat::Yuv444),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(img.bit_depth, BitDepth::Twelve);
        assert_eq!(img.chroma, ChromaFormat::Yuv444);
        // Out-of-range (> 4095) input is rejected.
        let mut bad = vec![100u16; (w * h * 3) as usize];
        bad[2] = 4096;
        assert!(encode_rgb12_266(&bad, w, h, &EncodeConfig::new()).is_err());
        // gray_alpha12 discards alpha == gray12 of the Y channel.
        let ya: Vec<u16> = (0..w * h * 2).map(|i| ((i * 5) % 4096) as u16).collect();
        let y: Vec<u16> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
        let cfg = EncodeConfig::new().with_quality(80);
        assert_eq!(
            encode_gray_alpha12_266(&ya, w, h, &cfg).unwrap(),
            encode_gray12_266(&y, w, h, &cfg).unwrap()
        );
        // 12-bit alpha auto-embeds into DecodedImage::alpha.
        let rgba: Vec<u16> = (0..w * h * 4).map(|i| ((i * 9) % 4096) as u16).collect();
        let heif = encode_rgba12_with_alpha(
            &rgba,
            w,
            h,
            &EncodeConfig::new()
                .with_lossless(true)
                .with_chroma(ChromaFormat::Yuv444),
        )
        .unwrap();
        let d = decode(&heif).unwrap();
        let a_exp: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[3].to_le_bytes())
            .collect();
        assert_eq!(d.bit_depth, BitDepth::Twelve);
        assert_eq!(d.alpha.as_ref().expect("alpha auto-embedded").planes, a_exp);
    }

    #[test]
    fn twelve_bit_is_supported_and_round_trips() {
        let (w, h) = (16u32, 16u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv444,
            ChromaFormat::Yuv422,
            ChromaFormat::Monochrome,
        ] {
            // Lossless 12-bit must round-trip exactly through encode → decode.
            let cfg = EncodeConfig::new()
                .with_bit_depth(BitDepth::Twelve)
                .with_lossless(true)
                .with_chroma(chroma);
            let img = decode_266(&encode_rgb_266(&rgb, w, h, &cfg).unwrap()).unwrap();
            assert_eq!(img.bit_depth, BitDepth::Twelve);
            assert_eq!(img.chroma, chroma);
            // Re-encoding the decoded 12-bit planes losslessly reproduces them.
            let planes16: Vec<u16> = img
                .planes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            assert!(planes16.iter().all(|&v| v <= 4095));
            let yuv = encode_yuv_266(
                &img.planes,
                w,
                h,
                &EncodeConfig::new()
                    .with_bit_depth(BitDepth::Twelve)
                    .with_lossless(true)
                    .with_chroma(chroma),
            )
            .unwrap();
            assert_eq!(decode_266(&yuv).unwrap().planes, img.planes);
        }
    }

    #[test]
    fn named_8bit_aliases_match_base() {
        let (w, h) = (16u32, 16u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 5 % 256) as u8).collect();
        let gray: Vec<u8> = (0..w * h).map(|i| (i * 3 % 256) as u8).collect();
        let cfg = EncodeConfig::new().with_quality(70);
        assert_eq!(
            encode_rgb8_266(&rgb, w, h, &cfg).unwrap(),
            encode_rgb_266(&rgb, w, h, &cfg).unwrap()
        );
        assert_eq!(
            encode_rgba8_266(&rgba, w, h, &cfg).unwrap(),
            encode_rgba_266(&rgba, w, h, &cfg).unwrap()
        );
        assert_eq!(
            encode_gray8_266(&gray, w, h, &cfg).unwrap(),
            encode_gray_266(&gray, w, h, &cfg).unwrap()
        );
    }

    #[test]
    fn gray_alpha_discards_alpha() {
        let (w, h) = (16u32, 16u32);
        let ya: Vec<u8> = (0..w * h * 2).map(|i| (i % 256) as u8).collect();
        let y: Vec<u8> = ya.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
        let cfg = EncodeConfig::new().with_quality(70);
        assert_eq!(
            encode_gray_alpha8_266(&ya, w, h, &cfg).unwrap(),
            encode_gray_266(&y, w, h, &cfg).unwrap()
        );
    }

    #[test]
    fn native_10bit_decodes_as_ten() {
        let (w, h) = (16u32, 16u32);
        let rgb: Vec<u16> = (0..w * h * 3).map(|i| ((i * 11) % 1024) as u16).collect();
        let cfg = EncodeConfig::new()
            .with_quality(80)
            .with_chroma(ChromaFormat::Yuv444);
        let img = decode_266(&encode_rgb10_266(&rgb, w, h, &cfg).unwrap()).unwrap();
        assert_eq!(img.bit_depth, BitDepth::Ten);
        assert_eq!(img.chroma, ChromaFormat::Yuv444);
    }

    #[test]
    fn ten_bit_range_is_enforced() {
        let (w, h) = (8u32, 8u32);
        let mut rgb: Vec<u16> = vec![100; (w * h * 3) as usize];
        rgb[5] = 1024; // out of 10-bit range
        assert!(encode_rgb10_266(&rgb, w, h, &EncodeConfig::new()).is_err());
    }

    #[test]
    fn decode_yuv_is_decode() {
        let (w, h) = (16u32, 16u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
        let s = encode_rgb_266(&rgb, w, h, &EncodeConfig::new().with_quality(70)).unwrap();
        assert_eq!(
            decode_yuv_266(&s).unwrap().planes,
            decode_266(&s).unwrap().planes
        );
    }

    #[test]
    fn rgba10_with_alpha_auto_embeds_alpha() {
        let (w, h) = (16u32, 16u32);
        let rgba: Vec<u16> = (0..w * h * 4).map(|i| ((i * 9) % 1024) as u16).collect();
        let cfg = EncodeConfig::new()
            .with_lossless(true)
            .with_chroma(ChromaFormat::Yuv444);
        let heif = encode_rgba10_with_alpha(&rgba, w, h, &cfg).unwrap();
        let img = decode(&heif).unwrap();
        let a_exp: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[3].to_le_bytes())
            .collect();
        assert_eq!(img.bit_depth, BitDepth::Ten);
        assert_eq!(
            img.alpha.as_ref().expect("alpha auto-embedded").planes,
            a_exp
        );
    }

    #[test]
    fn decode_heif_recovers_orientation_and_color() {
        let (w, h) = (16u32, 16u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
        let cicp = Cicp {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Smpte2084,
            matrix: MatrixCoefficients::Bt2020Ncl,
            full_range: false,
        };
        let cfg = EncodeConfig::new()
            .with_quality(80)
            .with_orientation(Orientation::Rotate90)
            .with_cicp(cicp)
            .with_icc_profile(vec![1, 2, 3, 4]);
        let img = decode(&encode_rgb(&rgb, w, h, &cfg).unwrap()).unwrap();
        assert_eq!(img.orientation, Orientation::Rotate90);
        assert_eq!(img.color.cicp, Some(cicp));
        assert_eq!(img.color.icc.as_deref(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn quality_maps_monotonically() {
        let lo = EncodeConfig::new().with_quality(10).luma_qp();
        let hi = EncodeConfig::new().with_quality(95).luma_qp();
        assert!(hi < lo, "higher quality must give lower QP ({hi} < {lo})");
        assert!(EncodeConfig::new().with_quality(100).luma_qp() <= 10);
    }

    #[test]
    fn rejects_bad_dims_and_quality() {
        let cfg = EncodeConfig::default();
        assert_eq!(
            encode_rgb_266(&[], 0, 1, &cfg),
            Err(EncodeError::InvalidDimensions {
                width: 0,
                height: 1
            })
        );
        let bad = EncodeConfig::default().with_quality(0);
        assert_eq!(
            encode_rgb_266(&[0; 3], 1, 1, &bad),
            Err(EncodeError::InvalidQuality(0))
        );
    }

    #[test]
    fn encodes_odd_dimensions_with_subsampled_chroma() {
        let (w, h) = (9u32, 11u32);
        let rgb = vec![128; (w * h * 3) as usize];
        for cfg in [
            EncodeConfig::default(),
            EncodeConfig::default().with_chroma(ChromaFormat::Yuv422),
            EncodeConfig::default().with_lossless(true),
        ] {
            let stream = encode_rgb_266(&rgb, w, h, &cfg).unwrap();
            let raw = decode_266(&stream).unwrap();
            let coded_chroma = cfg.chroma.for_dimensions(w, h);
            assert_eq!((raw.width, raw.height), (w, h));
            assert_eq!(raw.chroma, coded_chroma);

            let heif = encode_rgb(&rgb, w, h, &cfg).unwrap();
            let decoded = decode(&heif).unwrap();
            assert_eq!((decoded.width, decoded.height), (w, h));
            assert_eq!(decoded.luma_plane().samples(), (w * h) as usize);
            let (cb, cr) = decoded.chroma_planes().unwrap();
            let chroma_samples = (w.div_ceil(coded_chroma.sub_w() as u32)
                * h.div_ceil(coded_chroma.sub_h() as u32))
                as usize;
            assert_eq!(cb.samples(), chroma_samples);
            assert_eq!(cr.samples(), chroma_samples);
        }

        let (w, h) = (10u32, 11u32);
        let cfg = EncodeConfig::default().with_lossless(true);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y = vec![64u8; (w * h) as usize];
        let cb: Vec<u8> = (0..cw * ch).map(|i| (i / cw) as u8).collect();
        let cr: Vec<u8> = (0..cw * ch).map(|i| (100 + i / cw) as u8).collect();
        let planes: Vec<u8> = y.iter().chain(&cb).chain(&cr).copied().collect();
        let decoded = decode_266(&encode_yuv8_266(&planes, w, h, &cfg).unwrap()).unwrap();
        assert_eq!((decoded.width, decoded.height), (w, h));
        assert_eq!(decoded.chroma, ChromaFormat::Yuv422);
        let (got_cb, got_cr) = decoded.chroma_planes().unwrap();
        for row in 0..h as usize {
            assert_eq!(
                &got_cb.data[row * cw as usize..(row + 1) * cw as usize],
                &cb[row / 2 * cw as usize..(row / 2 + 1) * cw as usize]
            );
            assert_eq!(
                &got_cr.data[row * cw as usize..(row + 1) * cw as usize],
                &cr[row / 2 * cw as usize..(row / 2 + 1) * cw as usize]
            );
        }
    }

    #[test]
    fn enforces_level_6_2_picture_limits() {
        // MaxLumaPs for the declared Level 6.2 is 35 651 584 (≈ 8192×4352).
        // The limit is on the *coded* picture (display padded to a multiple of 8).
        assert!(
            validate_dims(8192, 4352).is_ok(),
            "exactly MaxLumaPs must pass"
        );
        assert!(
            validate_dims(6000, 4000).is_ok(),
            "24 Mpx is within level 6.2"
        );
        assert!(
            validate_dims(16_888, 8).is_ok(),
            "max coded width must pass"
        );
        // One luma column past the max coded area must be rejected.
        assert!(
            matches!(
                validate_dims(8193, 4352),
                Err(EncodeError::InvalidDimensions { .. })
            ),
            "coded 8200×4352 exceeds MaxLumaPs"
        );
        // Square 8K (67 Mpx) far exceeds level 6.2.
        assert!(matches!(
            validate_dims(8192, 8192),
            Err(EncodeError::InvalidDimensions { .. })
        ));
        // A single dimension past floor(sqrt(8·MaxLumaPs)) = 16 888.
        assert!(matches!(
            validate_dims(16_896, 8),
            Err(EncodeError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn rejects_wrong_buffer_size() {
        let cfg = EncodeConfig::default();
        assert_eq!(
            encode_rgb_266(&[0; 5], 2, 2, &cfg),
            Err(EncodeError::BufferSize {
                expected: 12,
                found: 5
            })
        );
    }

    #[test]
    fn encodes_a_valid_annexb_still_picture() {
        let cfg = EncodeConfig::default();
        // A small gradient so prediction and residual both do real work.
        let (w, h) = (40u32, 24u32);
        let mut rgb = vec![0u8; (w * h) as usize * 3];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) as usize) * 3;
                let v = ((x * 6 + y * 3) & 0xff) as u8;
                rgb[o] = v;
                rgb[o + 1] = v.wrapping_add(20);
                rgb[o + 2] = 255 - v;
            }
        }
        let out = encode_rgb_266(&rgb, w, h, &cfg).expect("encode");
        // Must start with an Annex-B start code and contain at least SPS, PPS,
        // and the IDR slice (three start codes).
        assert_eq!(&out[..4], &[0, 0, 0, 1], "missing start code");
        let starts = out.windows(3).filter(|wnd| wnd == &[0, 0, 1]).count();
        assert!(starts >= 3, "expected >=3 NAL units, found {starts}");
    }

    #[test]
    fn lossless_reconstruction_equals_source() {
        // Lossless mode must reconstruct the coded YCbCr planes exactly. The
        // encoder's reconstruction is what a conformant decoder reproduces, so
        // checking it against the source-derived planes proves true losslessness
        // for the codec (independent of the RGB->YCbCr input conversion).
        let (w, h) = (24usize, 16usize);
        let mut s = 0xc0ffee11u32;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut rgb = vec![0u8; w * h * 3];
        for b in rgb.iter_mut() {
            *b = (rng() & 0xff) as u8;
        }
        let cfg = EncodeConfig::default().with_lossless(true);
        let (bs, recon) = encode_rgb_with_reconstruction(&rgb, w as u32, h as u32, &cfg).unwrap();
        assert!(!bs.is_empty());

        // Independently derive the coded YCbCr (BT.601 full-range, 4:2:0) from
        // the source and compare against the reconstruction byte-for-byte.
        let sample = |x: usize, y: usize| {
            let o = (y * w + x) * 3;
            (rgb[o] as i32, rgb[o + 1] as i32, rgb[o + 2] as i32)
        };
        let mut expected = Vec::with_capacity(recon.len());
        // Mirror the encoder's yuv-crate-identical full-range BT.601 conversion.
        const BIAS: i32 = 4095;
        let yq = |r: i32, g: i32, b: i32| (2449 * r + 4809 * g + 934 * b + BIAS) >> 13;
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = sample(x, y);
                expected.push(yq(r, g, b) as u8);
            }
        }
        let (cw, ch) = (w / 2, h / 2);
        let mut chroma = |to_cb: bool| {
            for cy in 0..ch {
                for cx in 0..cw {
                    let (mut sr, mut sg, mut sb) = (0i32, 0i32, 0i32);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (r, g, b) = sample(cx * 2 + dx, cy * 2 + dy);
                            sr += r;
                            sg += g;
                            sb += b;
                        }
                    }
                    // rounded 2×2 RGB average, then direct-form chroma
                    let (r, g, b) = ((sr + 2) >> 2, (sg + 2) >> 2, (sb + 2) >> 2);
                    let v = if to_cb {
                        128 + ((-1382 * r - 2714 * g + 4096 * b + BIAS) >> 13)
                    } else {
                        128 + ((4096 * r - 3430 * g - 666 * b + BIAS) >> 13)
                    };
                    expected.push(v.clamp(0, 255) as u8);
                }
            }
        };
        chroma(true);
        chroma(false);
        assert_eq!(
            recon, expected,
            "lossless reconstruction differs from source"
        );
    }

    #[test]
    fn typed_heif_wrap_is_transparent() {
        // A typed HEIF encoder must produce a file whose embedded stream decodes
        // to exactly what the matching raw `_266` encoder produces.
        let (w, h) = (24u32, 16u32);
        let cfg = EncodeConfig::new().with_quality(85);

        let rgb10: Vec<u16> = (0..w * h * 3).map(|i| ((i * 7) % 1024) as u16).collect();
        let heif = encode_rgb10(&rgb10, w, h, &cfg).unwrap();
        let raw = encode_rgb10_266(&rgb10, w, h, &cfg).unwrap();
        assert_eq!(
            decode(&heif).unwrap().planes,
            decode_266(&raw).unwrap().planes
        );
        assert_eq!(decode(&heif).unwrap().bit_depth, BitDepth::Ten);

        let gray12: Vec<u16> = (0..w * h).map(|i| ((i * 13) % 4096) as u16).collect();
        let heif = encode_gray12(&gray12, w, h, &cfg).unwrap();
        let raw = encode_gray12_266(&gray12, w, h, &cfg).unwrap();
        assert_eq!(
            decode(&heif).unwrap().planes,
            decode_266(&raw).unwrap().planes
        );
        assert_eq!(decode(&heif).unwrap().chroma, ChromaFormat::Monochrome);
    }

    #[test]
    fn gray_alpha_heif_preserves_alpha() {
        // The HEIF gray+alpha encoders keep alpha as a monochrome aux image;
        // lossless makes both luma and alpha exact for the assertion.
        let (w, h) = (20u32, 12u32);
        let cfg = EncodeConfig::new().with_lossless(true);

        // 8-bit interleaved Y,A.
        let ya8: Vec<u8> = (0..w * h * 2).map(|i| (i % 256) as u8).collect();
        let img = decode(&encode_gray_alpha8(&ya8, w, h, &cfg).unwrap()).unwrap();
        let y_exp: Vec<u8> = ya8.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
        let a_exp: Vec<u8> = ya8.as_chunks::<2>().0.iter().map(|p| p[1]).collect();
        assert_eq!(img.planes, y_exp, "8-bit gray luma");
        assert_eq!(
            img.alpha.as_ref().expect("8-bit alpha embedded").planes,
            a_exp
        );

        // 10-bit interleaved Y,A (u16, little-endian planes out).
        let ya10: Vec<u16> = (0..w * h * 2).map(|i| ((i * 5) % 1024) as u16).collect();
        let img = decode(&encode_gray_alpha10(&ya10, w, h, &cfg).unwrap()).unwrap();
        let y_exp: Vec<u8> = ya10
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| p[0].to_le_bytes())
            .collect();
        let a_exp: Vec<u8> = ya10
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| p[1].to_le_bytes())
            .collect();
        assert_eq!(img.bit_depth, BitDepth::Ten);
        assert_eq!(img.planes, y_exp, "10-bit gray luma");
        assert_eq!(
            img.alpha.as_ref().expect("10-bit alpha embedded").planes,
            a_exp
        );
    }

    #[test]
    fn plane_views_match_packed_layout() {
        // 10-bit 4:2:0 planar in, lossless → decoded planes are exact.
        let (w, h) = (16u32, 12u32);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y_exp: Vec<u16> = (0..w * h).map(|i| (i % 1024) as u16).collect();
        let cb_exp: Vec<u16> = (0..cw * ch).map(|i| (i * 3 % 1024) as u16).collect();
        let cr_exp: Vec<u16> = (0..cw * ch).map(|i| (i * 7 % 1024) as u16).collect();
        let planes: Vec<u16> = y_exp
            .iter()
            .chain(&cb_exp)
            .chain(&cr_exp)
            .copied()
            .collect();
        let cfg = EncodeConfig::new()
            .with_lossless(true)
            .with_chroma(ChromaFormat::Yuv420);
        let img = decode(&encode_yuv10(&planes, w, h, &cfg).unwrap()).unwrap();

        let y = img.luma_plane();
        assert_eq!((y.width, y.height, y.bytes_per_sample), (w, h, 2));
        assert_eq!(y.to_u16(), y_exp);
        let (cb, cr) = img.chroma_planes().expect("4:2:0 has chroma");
        assert_eq!((cb.width, cb.height), (cw, ch));
        assert_eq!(cb.to_u16(), cb_exp);
        assert_eq!(cr.to_u16(), cr_exp);
        assert!(img.alpha_plane().is_none());

        // 8-bit gray + alpha: luma is 1 byte/sample, no chroma, alpha exact.
        let ya: Vec<u8> = (0..w * h * 2).map(|i| (i % 256) as u8).collect();
        let img = decode(
            &encode_gray_alpha8(&ya, w, h, &EncodeConfig::new().with_lossless(true)).unwrap(),
        )
        .unwrap();
        assert_eq!(img.luma_plane().bytes_per_sample, 1);
        assert!(img.chroma_planes().is_none(), "monochrome has no chroma");
        let a_exp: Vec<u8> = ya.as_chunks::<2>().0.iter().map(|p| p[1]).collect();
        assert_eq!(
            img.alpha_plane().expect("alpha present").data,
            a_exp.as_slice()
        );
    }
}
