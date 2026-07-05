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

//! HEIF / ISO base media file format wrapping for a single VVC still image.
//!
//! garnetash's encoder emits a raw Annex-B byte stream (SPS, PPS and one IDR
//! slice). To produce a file that image viewers recognise, this module wraps
//! that stream in a HEIF container (ISO/IEC 23008-12) built on ISOBMFF boxes
//! (ISO/IEC 14496-12), with the VVC carriage defined by ISO/IEC 14496-15: the
//! item type `vvc1` and the decoder-configuration box `vvcC`
//! (`VvcDecoderConfigurationRecord`).
//!
//! The primary item is a single coded image. Parameter sets (SPS/PPS) are stored
//! out-of-band in the `vvcC` record; the `mdat` sample carries only the VCL
//! (slice) NAL units, length-prefixed with a 4-byte length. The structural brand
//! is `mif1` (the codec-agnostic HEIF still-image brand) with `miaf` compatible.
//!
//! Layout produced:
//! ```text
//! ftyp                     mif1 / [mif1, miaf]
//! meta
//!   hdlr   'pict'
//!   pitm   primary item id = 1
//!   iloc   item 1 -> offset/length into mdat
//!   iinf   infe item 1, type 'vvc1'
//!   iprp
//!     ipco  1:vvcC(essential) 2:colr 3:ispe 4:pixi
//!     ipma  item 1 -> {1*,2,3,4}
//! mdat                     length-prefixed VCL NAL units
//! ```

use crate::color::{Cicp, ColorMetadata};
use crate::error::EncodeError;
use crate::fmt::{BitDepth, ChromaFormat};
use crate::metadata::{ImageMetadata, Orientation};

/// VVC NAL unit types used here (H.266 Table 7-1). Slice NAL types (the VCL
/// range) go in the `mdat` sample; SPS/PPS go in the `vvcC` arrays.
const NUT_SPS: u8 = 15;
const NUT_PPS: u8 = 16;

/// One parsed NAL unit from the Annex-B stream (header byte 0 onward; the
/// emulation-prevention bytes are already present, as the encoder wrote them).
struct Nal<'a> {
    nal_type: u8,
    data: &'a [u8],
}

/// Split an Annex-B byte stream into NAL units, dropping the 3- or 4-byte start
/// codes. The VVC NAL type is bits [7:3] of the second header byte.
fn split_annexb(stream: &[u8]) -> Vec<Nal<'_>> {
    // Find all start-code positions (00 00 01, optionally preceded by 00).
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        // The NAL ends where the next start code begins. The next start code is
        // 3 bytes (00 00 01) possibly with a leading 00; trim one trailing zero
        // that belonged to the next 4-byte start code.
        let mut end = if k + 1 < starts.len() {
            starts[k + 1] - 3
        } else {
            stream.len()
        };
        if end > s && k + 1 < starts.len() && stream[end - 1] == 0 {
            end -= 1;
        }
        if end >= s + 2 {
            let data = &stream[s..end];
            let nal_type = (data[1] >> 3) & 0x1f;
            nals.push(Nal { nal_type, data });
        }
    }
    nals
}

fn w32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn w16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// Open a plain box: 4-byte size placeholder + 4-char type. Returns the start
/// offset to be passed to [`patch`].
fn write_box(buf: &mut Vec<u8>, cc: &[u8; 4]) -> usize {
    let s = buf.len();
    w32(buf, 0); // size placeholder
    buf.extend_from_slice(cc);
    s
}

/// Open a full box: size + type + version(1) + flags(3).
fn write_fullbox(buf: &mut Vec<u8>, cc: &[u8; 4], ver: u8, flags: u32) -> usize {
    let s = write_box(buf, cc);
    buf.push(ver);
    buf.extend_from_slice(&flags.to_be_bytes()[1..]); // low 3 bytes
    s
}

/// Back-patch a box's size field (the u32 at `start`) to the bytes written since.
fn patch(buf: &mut [u8], start: usize) {
    let size = (buf.len() - start) as u32;
    buf[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

/// `general_profile_idc` for the VVC profile garnetash signals: Main 10 4:4:4
/// (33) for the 4:2:2 / 4:4:4 family, Main 10 (1) otherwise. Mirrors
/// `headers::Sps::profile_idc`.
fn profile_idc(chroma: ChromaFormat) -> u8 {
    match chroma {
        ChromaFormat::Yuv422 | ChromaFormat::Yuv444 => 33,
        _ => 1,
    }
}

/// Build the `VvcDecoderConfigurationRecord` payload (ISO/IEC 14496-15 §11.2.4),
/// carrying the SPS and PPS out-of-band. `ptl_present_flag = 1`; the
/// profile/tier/level values match the in-band SPS garnetash writes (tier main,
/// level 6.2, frame-only, no general constraints, no sub-profiles).
fn build_vvcc(
    sps: &[&[u8]],
    pps: &[&[u8]],
    chroma_idc: u8,
    bit_depth: u8,
    width: u16,
    height: u16,
    profile: u8,
) -> Vec<u8> {
    let mut r: Vec<u8> = Vec::new();

    // bit(5) reserved=11111, unsigned(2) LengthSizeMinusOne=3, unsigned(1) ptl_present_flag=1.
    r.push(0xf8 | (3 << 1) | 1); // 0xff

    // ols_idx(9)=0, num_sublayers(3)=1, constant_frame_rate(2)=1, chroma_format_idc(2).
    w16(&mut r, (1 << 4) | (1 << 2) | (chroma_idc as u16 & 0x3));
    // bit_depth_minus8(3), reserved(5)=11111.
    r.push(((bit_depth - 8) << 5) | 0x1f);

    // reserved(2)=0, num_bytes_constraint_info(6)=1.
    r.push(0x01);
    // general_profile_idc(7), general_tier_flag(1)=0.
    r.push(profile << 1);
    // general_level_idc(8) = 102 (level 6.2), matching headers::LEVEL_IDC.
    r.push(crate::headers::LEVEL_IDC);
    // ptl_frame_only_constraint_flag(1)=1, ptl_multilayer_enabled_flag(1)=0,
    // general_constraint_info(8*1-2 = 6 bits)=0  ->  0b1000_0000.
    r.push(0x80);
    // (no sublayer level bytes: num_sublayers == 1)
    // num_sub_profiles(8) = 0.
    r.push(0x00);
    // max_picture_width(16), max_picture_height(16), avg_frame_rate(16)=0.
    w16(&mut r, width);
    w16(&mut r, height);
    w16(&mut r, 0);

    // num_of_arrays(8): SPS and PPS.
    let arrays: &[(u8, &[&[u8]])] = &[(NUT_SPS, sps), (NUT_PPS, pps)];
    let present = arrays.iter().filter(|(_, l)| !l.is_empty()).count();
    r.push(present as u8);
    for &(nut, list) in arrays {
        if list.is_empty() {
            continue;
        }
        // array_completeness(1)=1, reserved(2)=0, NAL_unit_type(5).
        r.push(0x80 | (nut & 0x1f));
        // num_nalus(16) (SPS/PPS are not DCI/OPI, so the count is present).
        w16(&mut r, list.len() as u16);
        for &nalu in list {
            w16(&mut r, nalu.len() as u16);
            r.extend_from_slice(nalu);
        }
    }
    r
}

/// Write the `ftyp` box: major brand `mif1` (codec-agnostic HEIF still image),
/// compatible brands `mif1` and `miaf`.
fn write_ftyp(f: &mut Vec<u8>) {
    let s = write_box(f, b"ftyp");
    f.extend_from_slice(b"mif1"); // major_brand
    w32(f, 0); // minor_version
    f.extend_from_slice(b"mif1");
    f.extend_from_slice(b"miaf");
    patch(f, s);
}

/// Write an `nclx` `colr` box carrying a CICP description.
fn write_colr_nclx(f: &mut Vec<u8>, cicp: &crate::color::Cicp) {
    let s = write_box(f, b"colr");
    f.extend_from_slice(&cicp.nclx_payload());
    patch(f, s);
}

/// Write a `prof` `colr` box carrying an embedded ICC profile.
fn write_colr_icc(f: &mut Vec<u8>, icc: &[u8]) {
    let s = write_box(f, b"colr");
    f.extend_from_slice(b"prof");
    f.extend_from_slice(icc);
    patch(f, s);
}

/// Wrap a raw Annex-B VVC still-image stream (SPS + PPS + one IDR slice) into a
/// HEIF file. `width`/`height` are the display dimensions, `bit_depth` the
/// sample bit depth and `chroma` the chroma format used by the encoder.
/// `color` describes the color space (`colr` boxes) and `meta` carries optional
/// orientation, content-light-level and Exif metadata.
pub(crate) fn wrap_vvc_still(
    annexb: &[u8],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    chroma: ChromaFormat,
    color: &ColorMetadata,
    meta: &ImageMetadata,
) -> Result<Vec<u8>, EncodeError> {
    let nals = split_annexb(annexb);
    if nals.is_empty() {
        return Err(EncodeError::Unsupported(
            "empty VVC stream: nothing to wrap",
        ));
    }

    let sps: Vec<&[u8]> = nals
        .iter()
        .filter(|n| n.nal_type == NUT_SPS)
        .map(|n| n.data)
        .collect();
    let pps: Vec<&[u8]> = nals
        .iter()
        .filter(|n| n.nal_type == NUT_PPS)
        .map(|n| n.data)
        .collect();
    // VCL NAL units (slice data): VVC VCL NAL unit types are 0..=11.
    let vcl: Vec<&[u8]> = nals
        .iter()
        .filter(|n| n.nal_type <= 11)
        .map(|n| n.data)
        .collect();
    if sps.is_empty() || pps.is_empty() || vcl.is_empty() {
        return Err(EncodeError::Unsupported(
            "VVC stream missing SPS, PPS or slice data; cannot wrap",
        ));
    }

    let bd = bit_depth.bits();
    let chroma_idc = chroma.idc() as u8;
    let vvcc = build_vvcc(
        &sps,
        &pps,
        chroma_idc,
        bd,
        width.min(0xffff) as u16,
        height.min(0xffff) as u16,
        profile_idc(chroma),
    );

    // The mdat image sample: VCL NAL units, each prefixed by a 4-byte length.
    let mut sample: Vec<u8> = Vec::new();
    for &nalu in &vcl {
        w32(&mut sample, nalu.len() as u32);
        sample.extend_from_slice(nalu);
    }

    // Optional Exif item payload: a 4-byte exif_tiff_header_offset (0) then the
    // raw Exif/TIFF block.
    let has_exif = meta.exif.is_some();
    let exif_payload: Vec<u8> = meta
        .exif
        .as_ref()
        .map(|e| {
            let mut p = Vec::with_capacity(e.len() + 4);
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(e);
            p
        })
        .unwrap_or_default();

    let chans: u8 = if matches!(chroma, ChromaFormat::Monochrome) {
        1
    } else {
        3
    };

    let mut f: Vec<u8> = Vec::new();
    write_ftyp(&mut f);

    let meta_start = write_fullbox(&mut f, b"meta", 0, 0);

    // hdlr: handler type 'pict'.
    {
        let s = write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0); // pre_defined
        f.extend_from_slice(b"pict");
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0);
        f.push(0); // name (empty, NUL-terminated)
        patch(&mut f, s);
    }

    // pitm: primary item id = 1.
    {
        let s = write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }

    // iloc: image item (1) and optional Exif item (2); offsets patched once the
    // mdat positions are known.
    let img_offset_patch_pos;
    let mut exif_offset_patch_pos = 0usize;
    {
        let s = write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, if has_exif { 2 } else { 1 }); // item_count
        // Image item 1.
        w16(&mut f, 1); // item_id
        w16(&mut f, 0); // data_reference_index
        w16(&mut f, 1); // extent_count
        img_offset_patch_pos = f.len();
        w32(&mut f, 0); // extent_offset (patched)
        w32(&mut f, sample.len() as u32); // extent_length
        if has_exif {
            w16(&mut f, 2); // item_id
            w16(&mut f, 0); // data_reference_index
            w16(&mut f, 1); // extent_count
            exif_offset_patch_pos = f.len();
            w32(&mut f, 0); // extent_offset (patched)
            w32(&mut f, exif_payload.len() as u32); // extent_length
        }
        patch(&mut f, s);
    }

    // iinf / infe: image item (type 'vvc1') + optional Exif item (type 'Exif').
    {
        let s = write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, if has_exif { 2 } else { 1 }); // entry_count
        {
            let si = write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 1); // item_id
            w16(&mut f, 0); // item_protection_index
            f.extend_from_slice(b"vvc1"); // item_type
            f.push(0); // item_name (empty)
            patch(&mut f, si);
        }
        if has_exif {
            let si = write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 2); // item_id
            w16(&mut f, 0); // item_protection_index
            f.extend_from_slice(b"Exif"); // item_type
            f.push(0); // item_name (empty)
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iref: the Exif item describes (cdsc) the image item.
    if has_exif {
        let s = write_fullbox(&mut f, b"iref", 0, 0);
        {
            let si = write_box(&mut f, b"cdsc");
            w16(&mut f, 2); // from_item_id (Exif)
            w16(&mut f, 1); // reference_count
            w16(&mut f, 1); // to_item_id (image)
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iprp { ipco { vvcC, colr, ispe, pixi, [prof, irot, imir, clli] }, ipma }.
    {
        let s = write_box(&mut f, b"iprp");
        // Optional property indices (0 = absent).
        let mut colr2_idx = 0u8;
        let mut irot_idx = 0u8;
        let mut imir_idx = 0u8;
        let mut clli_idx = 0u8;
        {
            let si = write_box(&mut f, b"ipco");
            // 1: vvcC (essential decoder config).
            {
                let sh = write_box(&mut f, b"vvcC");
                f.extend_from_slice(&vvcc);
                patch(&mut f, sh);
            }
            // 2: colr — nclx (CICP), or the ICC profile if that is all we have.
            if color.cicp.is_some() || color.icc.is_none() {
                write_colr_nclx(&mut f, &color.effective_cicp());
            } else if let Some(icc) = &color.icc {
                write_colr_icc(&mut f, icc);
            }
            // 3: ispe (image spatial extents).
            {
                let sh = write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, width);
                w32(&mut f, height);
                patch(&mut f, sh);
            }
            // 4: pixi (bits per channel).
            {
                let sh = write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(chans);
                for _ in 0..chans {
                    f.push(bd);
                }
                patch(&mut f, sh);
            }
            // 5+: optional properties.
            let mut next: u8 = 5;
            if color.has_secondary_colr()
                && let Some(icc) = &color.icc
            {
                write_colr_icc(&mut f, icc);
                colr2_idx = next;
                next += 1;
            }
            if meta.orientation.irot_steps() != 0 {
                let sh = write_box(&mut f, b"irot");
                f.push(meta.orientation.irot_steps() & 0x03);
                patch(&mut f, sh);
                irot_idx = next;
                next += 1;
            }
            if let Some(axis) = meta.orientation.imir_axis() {
                let sh = write_box(&mut f, b"imir");
                f.push(if axis { 1 } else { 0 });
                patch(&mut f, sh);
                imir_idx = next;
                next += 1;
            }
            if let Some(cll) = meta.content_light_level {
                let sh = write_box(&mut f, b"clli");
                f.extend_from_slice(&cll.clli_payload());
                patch(&mut f, sh);
                clli_idx = next;
                next += 1;
            }
            let _ = next;
            patch(&mut f, si);
        }
        // ipma: associate properties with item 1.
        {
            let mut assoc: Vec<u8> = vec![0x80 | 1, 2, 3, 4]; // vvcC* colr ispe pixi
            if colr2_idx != 0 {
                assoc.push(colr2_idx);
            }
            if irot_idx != 0 {
                assoc.push(0x80 | irot_idx); // transformative -> essential
            }
            if imir_idx != 0 {
                assoc.push(0x80 | imir_idx);
            }
            if clli_idx != 0 {
                assoc.push(clli_idx);
            }
            let si = write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 1); // entry_count
            w16(&mut f, 1); // item_id
            f.push(assoc.len() as u8);
            f.extend_from_slice(&assoc);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // mdat: the image sample, then the optional Exif payload.
    let mdat_start = write_box(&mut f, b"mdat");
    let sample_abs = f.len() as u32;
    f.extend_from_slice(&sample);
    let exif_abs = f.len() as u32;
    if has_exif {
        f.extend_from_slice(&exif_payload);
    }
    patch(&mut f, mdat_start);

    f[img_offset_patch_pos..img_offset_patch_pos + 4].copy_from_slice(&sample_abs.to_be_bytes());
    if has_exif {
        f[exif_offset_patch_pos..exif_offset_patch_pos + 4]
            .copy_from_slice(&exif_abs.to_be_bytes());
    }

    Ok(f)
}

/// Extract `(sps, pps, length-prefixed VCL sample)` from an Annex-B VVC stream.
#[allow(clippy::type_complexity)]
fn split_for_container(annexb: &[u8]) -> Result<(Vec<&[u8]>, Vec<&[u8]>, Vec<u8>), EncodeError> {
    let nals = split_annexb(annexb);
    let sps: Vec<&[u8]> = nals
        .iter()
        .filter(|n| n.nal_type == NUT_SPS)
        .map(|n| n.data)
        .collect();
    let pps: Vec<&[u8]> = nals
        .iter()
        .filter(|n| n.nal_type == NUT_PPS)
        .map(|n| n.data)
        .collect();
    let vcl: Vec<&[u8]> = nals
        .iter()
        .filter(|n| n.nal_type <= 11)
        .map(|n| n.data)
        .collect();
    if sps.is_empty() || pps.is_empty() || vcl.is_empty() {
        return Err(EncodeError::Unsupported(
            "VVC stream missing SPS, PPS or slice data",
        ));
    }
    let mut sample = Vec::new();
    for &n in &vcl {
        w32(&mut sample, n.len() as u32);
        sample.extend_from_slice(n);
    }
    Ok((sps, pps, sample))
}

/// Wrap a color VVC still **and** a monochrome alpha VVC still into one HEIF
/// file. The alpha plane is stored as a separate auxiliary image item (item 2)
/// of type `vvc1`, linked to the master image (item 1) by an `auxl` item
/// reference and tagged with an `auxC` property carrying the MIAF alpha URN
/// `urn:mpeg:mpegB:cicp:systems:auxiliary:alpha` (the codec-independent alpha
/// auxiliary type, correct for VVC). This carries the master's color metadata;
/// orientation/HDR side data and Exif are not attached in the alpha path.
pub(crate) fn wrap_vvc_still_with_alpha(
    master: &[u8],
    alpha: &[u8],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    chroma: ChromaFormat,
    color: &ColorMetadata,
) -> Result<Vec<u8>, EncodeError> {
    const ALPHA_URN: &[u8] = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";
    let (m_sps, m_pps, m_sample) = split_for_container(master)?;
    let (a_sps, a_pps, a_sample) = split_for_container(alpha)?;

    let bd = bit_depth.bits();
    let w16cap = width.min(0xffff) as u16;
    let h16cap = height.min(0xffff) as u16;
    let m_vvcc = build_vvcc(
        &m_sps,
        &m_pps,
        chroma.idc() as u8,
        bd,
        w16cap,
        h16cap,
        profile_idc(chroma),
    );
    // Alpha is monochrome (chroma_idc 0), Main-10/Main intra profile (1).
    let a_vvcc = build_vvcc(&a_sps, &a_pps, 0, bd, w16cap, h16cap, 1);

    let m_chans: u8 = if matches!(chroma, ChromaFormat::Monochrome) {
        1
    } else {
        3
    };

    let mut f: Vec<u8> = Vec::new();
    write_ftyp(&mut f);
    let meta_start = write_fullbox(&mut f, b"meta", 0, 0);

    // hdlr 'pict'.
    {
        let s = write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0);
        f.extend_from_slice(b"pict");
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0);
        f.push(0);
        patch(&mut f, s);
    }
    // pitm: primary = master item 1.
    {
        let s = write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }
    // iloc: item 1 (master), item 2 (alpha).
    let m_off_patch;
    let a_off_patch;
    {
        let s = write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, 2);
        w16(&mut f, 1);
        w16(&mut f, 0);
        w16(&mut f, 1);
        m_off_patch = f.len();
        w32(&mut f, 0);
        w32(&mut f, m_sample.len() as u32);
        w16(&mut f, 2);
        w16(&mut f, 0);
        w16(&mut f, 1);
        a_off_patch = f.len();
        w32(&mut f, 0);
        w32(&mut f, a_sample.len() as u32);
        patch(&mut f, s);
    }
    // iinf: master + alpha, both 'vvc1'.
    {
        let s = write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, 2);
        for id in [1u16, 2] {
            let si = write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, id);
            w16(&mut f, 0);
            f.extend_from_slice(b"vvc1");
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }
    // iref: auxl from alpha (2) to master (1).
    {
        let s = write_fullbox(&mut f, b"iref", 0, 0);
        let si = write_box(&mut f, b"auxl");
        w16(&mut f, 2); // from = alpha
        w16(&mut f, 1); // reference_count
        w16(&mut f, 1); // to = master
        patch(&mut f, si);
        patch(&mut f, s);
    }
    // iprp { ipco { (master) vvcC,colr,ispe,pixi ; (alpha) vvcC,ispe,pixi,auxC }, ipma }.
    {
        let s = write_box(&mut f, b"iprp");
        {
            let si = write_box(&mut f, b"ipco");
            // 1: master vvcC
            let sh = write_box(&mut f, b"vvcC");
            f.extend_from_slice(&m_vvcc);
            patch(&mut f, sh);
            // 2: colr
            if color.cicp.is_some() || color.icc.is_none() {
                write_colr_nclx(&mut f, &color.effective_cicp());
            } else if let Some(icc) = &color.icc {
                write_colr_icc(&mut f, icc);
            }
            // 3: ispe
            let sh = write_fullbox(&mut f, b"ispe", 0, 0);
            w32(&mut f, width);
            w32(&mut f, height);
            patch(&mut f, sh);
            // 4: pixi (master channels)
            let sh = write_fullbox(&mut f, b"pixi", 0, 0);
            f.push(m_chans);
            for _ in 0..m_chans {
                f.push(bd);
            }
            patch(&mut f, sh);
            // 5: alpha vvcC
            let sh = write_box(&mut f, b"vvcC");
            f.extend_from_slice(&a_vvcc);
            patch(&mut f, sh);
            // 6: alpha ispe (same extents)
            let sh = write_fullbox(&mut f, b"ispe", 0, 0);
            w32(&mut f, width);
            w32(&mut f, height);
            patch(&mut f, sh);
            // 7: alpha pixi (1 channel)
            let sh = write_fullbox(&mut f, b"pixi", 0, 0);
            f.push(1);
            f.push(bd);
            patch(&mut f, sh);
            // 8: auxC
            let sh = write_fullbox(&mut f, b"auxC", 0, 0);
            f.extend_from_slice(ALPHA_URN);
            patch(&mut f, sh);
            patch(&mut f, si);
        }
        // ipma: item1 -> {1,2,3,4}, item2 -> {5,6,7,8}.
        {
            let si = write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 2); // entry_count
            w16(&mut f, 1);
            f.push(4);
            f.extend_from_slice(&[0x80 | 1, 2, 3, 4]);
            w16(&mut f, 2);
            f.push(4);
            f.extend_from_slice(&[0x80 | 5, 6, 7, 0x80 | 8]);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }
    patch(&mut f, meta_start);

    // mdat: master sample, then alpha sample.
    let mdat_start = write_box(&mut f, b"mdat");
    let m_abs = f.len() as u32;
    f.extend_from_slice(&m_sample);
    let a_abs = f.len() as u32;
    f.extend_from_slice(&a_sample);
    patch(&mut f, mdat_start);
    f[m_off_patch..m_off_patch + 4].copy_from_slice(&m_abs.to_be_bytes());
    f[a_off_patch..a_off_patch + 4].copy_from_slice(&a_abs.to_be_bytes());
    Ok(f)
}

/// Read a big-endian `u32` at `p`, or `None` if out of range.
fn rd32(b: &[u8], p: usize) -> Option<u32> {
    b.get(p..p + 4)
        .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd16(b: &[u8], p: usize) -> Option<u16> {
    b.get(p..p + 2).map(|s| u16::from_be_bytes([s[0], s[1]]))
}

/// Find the first child box of the given 4CC within `[start, end)`, returning
/// its payload range `(payload_start, payload_end)`. `full` skips the 4-byte
/// version/flags of a FullBox.
fn find_child(
    b: &[u8],
    start: usize,
    end: usize,
    fourcc: &[u8; 4],
    full: bool,
) -> Option<(usize, usize)> {
    let mut p = start;
    while p + 8 <= end {
        let size = rd32(b, p)? as usize;
        let (hdr, box_end) = match size {
            1 => (16, p + rd32(b, p + 12)? as usize), // 64-bit size (low 32 bits suffice here)
            0 => (8, end),
            _ => (8, p + size),
        };
        if box_end > end || box_end <= p {
            return None;
        }
        if &b[p + 4..p + 8] == fourcc {
            let ps = p + hdr + if full { 4 } else { 0 };
            return Some((ps, box_end));
        }
        p = box_end;
    }
    None
}

/// Extract a raw Annex-B VVC stream (SPS, PPS, then the VCL slice) from a HEIF
/// file produced by [`wrap_vvc_still`]. The inverse of the container: it reads
/// the parameter sets from the `vvcC` property and the slice sample via
/// `iloc` → `mdat`, and concatenates them as start-code-prefixed NAL units.
/// Like [`find_child`] but returns the `nth` (0-based) matching child box.
fn find_nth_child(
    b: &[u8],
    start: usize,
    end: usize,
    fourcc: &[u8; 4],
    nth: usize,
) -> Option<(usize, usize)> {
    let mut p = start;
    let mut seen = 0usize;
    while p + 8 <= end {
        let size = rd32(b, p)? as usize;
        let box_end = match size {
            1 => p + rd32(b, p + 12)? as usize,
            0 => end,
            _ => p + size,
        };
        if box_end > end || box_end <= p {
            return None;
        }
        if &b[p + 4..p + 8] == fourcc {
            if seen == nth {
                return Some((p + 8, box_end));
            }
            seen += 1;
        }
        p = box_end;
    }
    None
}

/// Extract the Annex-B VVC stream for the `want`-th image item (0-based) of a
/// garnetash HEIF file. Item 0 is the primary color image; item 1, when
/// present, is the alpha auxiliary. The matching `vvcC` is paired by box order
/// in `ipco`, which holds for files written by garnetash.
fn extract_item(heif: &[u8], want: usize) -> Result<Vec<u8>, EncodeError> {
    let n = heif.len();
    let err = |m| EncodeError::Decode(m);

    let (meta_s, meta_e) = find_child(heif, 0, n, b"meta", true).ok_or(err("HEIF: no meta box"))?;
    let (iloc_s, _e) =
        find_child(heif, meta_s, meta_e, b"iloc", true).ok_or(err("HEIF: no iloc box"))?;
    let sizes = *heif.get(iloc_s).ok_or(err("HEIF: short iloc"))?;
    let offset_size = (sizes >> 4) as usize;
    let length_size = (sizes & 0xf) as usize;
    let base_offset_size = (*heif.get(iloc_s + 1).ok_or(err("HEIF: short iloc"))? >> 4) as usize;
    let item_count = rd16(heif, iloc_s + 2).ok_or(err("HEIF: short iloc"))? as usize;
    if want >= item_count {
        return Err(err("HEIF: item index out of range"));
    }
    let read_sized = |b: &[u8], at: usize, sz: usize| -> Option<u64> {
        let mut v = 0u64;
        for i in 0..sz {
            v = (v << 8) | *b.get(at + i)? as u64;
        }
        Some(v)
    };
    // Walk every item entry (version 0: no construction_method / index), keeping
    // the first extent of the requested one.
    let mut p = iloc_s + 4;
    let mut found: Option<(usize, usize)> = None;
    for idx in 0..item_count {
        p += 2 + 2 + base_offset_size; // item_id, data_reference_index, base_offset
        let extent_count = rd16(heif, p).ok_or(err("HEIF: short iloc extent"))? as usize;
        p += 2;
        for e in 0..extent_count {
            let off =
                read_sized(heif, p, offset_size).ok_or(err("HEIF: bad extent offset"))? as usize;
            p += offset_size;
            let len =
                read_sized(heif, p, length_size).ok_or(err("HEIF: bad extent length"))? as usize;
            p += length_size;
            if idx == want && e == 0 {
                found = Some((off, len));
            }
        }
    }
    let (ext_off, ext_len) = found.ok_or(err("HEIF: item has no extent"))?;
    let end = ext_off
        .checked_add(ext_len)
        .ok_or(err("HEIF: extent overflow"))?;
    if end > n {
        return Err(err("HEIF: extent overruns file"));
    }
    let sample = &heif[ext_off..end];

    let (iprp_s, iprp_e) =
        find_child(heif, meta_s, meta_e, b"iprp", false).ok_or(err("HEIF: no iprp box"))?;
    let (ipco_s, ipco_e) =
        find_child(heif, iprp_s, iprp_e, b"ipco", false).ok_or(err("HEIF: no ipco box"))?;
    let (vvcc_s, vvcc_e) =
        find_nth_child(heif, ipco_s, ipco_e, b"vvcC", want).ok_or(err("HEIF: no vvcC for item"))?;
    let (sps_nals, pps_nals) = read_vvcc_arrays(&heif[vvcc_s..vvcc_e])?;
    if sps_nals.is_empty() || pps_nals.is_empty() {
        return Err(err("HEIF: vvcC missing SPS/PPS"));
    }

    let start = [0u8, 0, 0, 1];
    let mut out = Vec::with_capacity(ext_len + 64);
    for nal in sps_nals.iter().chain(pps_nals.iter()) {
        out.extend_from_slice(&start);
        out.extend_from_slice(nal);
    }
    let mut q = 0usize;
    while q + 4 <= sample.len() {
        let ln = rd32(sample, q).ok_or(err("HEIF: bad NAL length"))? as usize;
        q += 4;
        if q + ln > sample.len() {
            return Err(err("HEIF: NAL length overruns sample"));
        }
        out.extend_from_slice(&start);
        out.extend_from_slice(&sample[q..q + ln]);
        q += ln;
    }
    Ok(out)
}

/// Extract the primary (color) image's Annex-B VVC stream from a HEIF file.
pub(crate) fn extract_vvc_stream(heif: &[u8]) -> Result<Vec<u8>, EncodeError> {
    extract_item(heif, 0)
}

/// Extract the alpha auxiliary image's stream, if the file has one (item 1).
pub(crate) fn extract_alpha_stream(heif: &[u8]) -> Option<Vec<u8>> {
    extract_item(heif, 1).ok()
}

/// Read the `ispe` display extent associated with an image item.
pub(crate) fn extract_spatial_extents(heif: &[u8], want: usize) -> Option<(u32, u32)> {
    let (meta_s, meta_e) = find_child(heif, 0, heif.len(), b"meta", true)?;
    let (iprp_s, iprp_e) = find_child(heif, meta_s, meta_e, b"iprp", false)?;
    let (ipco_s, ipco_e) = find_child(heif, iprp_s, iprp_e, b"ipco", false)?;
    let (ispe_s, ispe_e) = find_nth_child(heif, ipco_s, ipco_e, b"ispe", want)?;
    if ispe_s + 12 > ispe_e {
        return None;
    }
    Some((rd32(heif, ispe_s + 4)?, rd32(heif, ispe_s + 8)?))
}

/// Read the master image's display orientation and color metadata from the
/// container property store (`iprp`/`ipco`): the first `irot`/`imir` boxes give
/// orientation, and the `colr` boxes give the CICP description (`nclx`) and/or
/// the embedded ICC profile (`prof`). Missing properties yield the defaults.
pub(crate) fn extract_metadata(heif: &[u8]) -> (Orientation, ColorMetadata) {
    let mut steps = 0u8;
    let mut axis: Option<bool> = None;
    let mut color = ColorMetadata::default();
    let n = heif.len();
    if let Some((meta_s, meta_e)) = find_child(heif, 0, n, b"meta", true)
        && let Some((iprp_s, iprp_e)) = find_child(heif, meta_s, meta_e, b"iprp", false)
        && let Some((ipco_s, ipco_e)) = find_child(heif, iprp_s, iprp_e, b"ipco", false)
    {
        if let Some((s, _)) = find_child(heif, ipco_s, ipco_e, b"irot", false)
            && let Some(&b) = heif.get(s)
        {
            steps = b & 3;
        }
        if let Some((s, _)) = find_child(heif, ipco_s, ipco_e, b"imir", false)
            && let Some(&b) = heif.get(s)
        {
            axis = Some(b & 1 == 1);
        }
        // There may be up to two colr boxes (nclx and/or prof); read all.
        let mut i = 0;
        while let Some((s, e)) = find_nth_child(heif, ipco_s, ipco_e, b"colr", i) {
            match heif.get(s..s + 4) {
                Some(b"nclx") => color.cicp = Cicp::from_nclx_payload(&heif[s..e]),
                Some(b"prof") | Some(b"rICC") => color.icc = Some(heif[s + 4..e].to_vec()),
                _ => {}
            }
            i += 1;
        }
    }
    (Orientation::from_irot_imir(steps, axis), color)
}

/// Parse the SPS (NAL type 15) and PPS (16) arrays out of a `vvcC` payload.
#[allow(clippy::type_complexity)]
fn read_vvcc_arrays(v: &[u8]) -> Result<(Vec<Vec<u8>>, Vec<Vec<u8>>), EncodeError> {
    let err = |m| EncodeError::Decode(m);
    let mut p = 0usize;
    let b0 = *v.first().ok_or(err("vvcC: empty"))?;
    p += 1;
    let ptl_present = b0 & 1;
    if ptl_present == 1 {
        p += 2; // ols_idx/num_sublayers/cfr/chroma_idc
        p += 1; // bit_depth/reserved
        let num_bytes_ci = (*v.get(p).ok_or(err("vvcC: short"))? & 0x3f) as usize;
        p += 1;
        p += 1; // profile/tier
        p += 1; // level
        p += num_bytes_ci; // frame_only + multilayer + constraint info
        // num_sublayers == 1 -> no sublayer level bytes.
        let nsp = *v.get(p).ok_or(err("vvcC: short"))? as usize;
        p += 1;
        p += 4 * nsp;
        p += 6; // max_w, max_h, avg_frame_rate
    }
    let num_arrays = *v.get(p).ok_or(err("vvcC: short"))?;
    p += 1;
    let (mut sps, mut pps) = (Vec::new(), Vec::new());
    for _ in 0..num_arrays {
        let hdr = *v.get(p).ok_or(err("vvcC: short array"))?;
        p += 1;
        let nut = hdr & 0x1f;
        let num_nalus = rd16(v, p).ok_or(err("vvcC: short array"))?;
        p += 2;
        for _ in 0..num_nalus {
            let ln = rd16(v, p).ok_or(err("vvcC: short nalu"))? as usize;
            p += 2;
            let nal = v.get(p..p + ln).ok_or(err("vvcC: nalu overrun"))?.to_vec();
            p += ln;
            match nut {
                15 => sps.push(nal),
                16 => pps.push(nal),
                _ => {}
            }
        }
    }
    Ok((sps, pps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_annexb_into_nals() {
        // Two fake NALs with 4-byte and 3-byte start codes.
        let mut s = vec![0, 0, 0, 1, 0x00, 0x78, 0xaa]; // type (0x78>>3)=15 SPS
        s.extend_from_slice(&[0, 0, 1, 0x00, 0x80, 0xbb]); // type (0x80>>3)=16 PPS
        let nals = split_annexb(&s);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].nal_type, 15);
        assert_eq!(nals[1].nal_type, 16);
    }

    #[test]
    fn vvcc_starts_with_expected_fixed_bytes() {
        let sps: Vec<&[u8]> = vec![&[0x00, 0x78, 0x11]];
        let pps: Vec<&[u8]> = vec![&[0x00, 0x80, 0x22]];
        let v = build_vvcc(&sps, &pps, 1, 8, 64, 64, 1);
        assert_eq!(v[0], 0xff); // reserved|LenM1=3|ptl_present=1
        assert_eq!(&v[1..3], &[0x00, 0x15]); // ols/sublayers/cfr/chroma_idc=1
        assert_eq!(v[3], 0x1f); // bit_depth_minus8=0
        assert_eq!(v[4], 0x01); // num_bytes_constraint_info=1
        assert_eq!(v[5], 0x02); // profile 1 << 1 | tier 0
        assert_eq!(v[6], 102); // level 6.2
        assert_eq!(v[7], 0x80); // frame_only=1, multilayer=0, gci=0
    }
}
