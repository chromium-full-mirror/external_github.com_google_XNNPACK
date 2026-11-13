// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-avx512vnnigfni.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_4x32c4__avx512vnnigfni_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(row_sum != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  const __m512i vmask = _mm512_set1_epi8(0x03);
  XNN_FORCE_REALIZATION(vmask);
  const __m512i vshr2 = _mm512_set1_epi64(0x0408000000000000);
  const __m512i vshr4 = _mm512_set1_epi64(0x1020000000000000);
  const __m512i vshr6 = _mm512_set1_epi64(0x4080000000000000);
  XNN_FORCE_REALIZATION(vshr2);
  XNN_FORCE_REALIZATION(vshr4);
  XNN_FORCE_REALIZATION(vshr6);

  do {
    const __m512i vksum0 = _mm512_load_epi32((const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_load_epi32((const int32_t*) w + 16);
    const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
    __m512i vacc0x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point0);
    __m512i vacc0x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point0);
    const __m512i vinput_zero_point1 = _mm512_set1_epi32((int) quantization_params[1].zero_point);
    __m512i vacc1x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point1);
    __m512i vacc1x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point1);
    const __m512i vinput_zero_point2 = _mm512_set1_epi32((int) quantization_params[2].zero_point);
    __m512i vacc2x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point2);
    __m512i vacc2x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point2);
    const __m512i vinput_zero_point3 = _mm512_set1_epi32((int) quantization_params[3].zero_point);
    __m512i vacc3x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point3);
    __m512i vacc3x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point3);
    __m512i vacc1x0x0 = _mm512_setzero_epi32();
    __m512i vacc1x0x1 = _mm512_setzero_epi32();
    __m512i vacc1x1x0 = _mm512_setzero_epi32();
    __m512i vacc1x1x1 = _mm512_setzero_epi32();
    __m512i vacc1x2x0 = _mm512_setzero_epi32();
    __m512i vacc1x2x1 = _mm512_setzero_epi32();
    __m512i vacc1x3x0 = _mm512_setzero_epi32();
    __m512i vacc1x3x1 = _mm512_setzero_epi32();
    w = (const int32_t*) w + 32;
    const float* kzp = (const float*) w;
    w = (const float*) w + 32;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m512i vbb0 = _mm512_load_si512((const int8_t*) w + 0);
      const __m512i vbb1 = _mm512_load_si512((const int8_t*) w + 64);

      const __m512i vb0x0 = _mm512_and_si512(vbb0, vmask);
      const __m512i vb1x0 = _mm512_and_si512(vbb1, vmask);
      const __m512i va0x0 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
      vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x0, va0x0);
      vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x0, va0x0);
      const __m512i va1x0 = _mm512_set1_epi32((int) unaligned_load_u32(a1));
      vacc1x0 = _mm512_dpbusd_epi32(vacc1x0, vb0x0, va1x0);
      vacc1x1 = _mm512_dpbusd_epi32(vacc1x1, vb1x0, va1x0);
      const __m512i va2x0 = _mm512_set1_epi32((int) unaligned_load_u32(a2));
      vacc2x0 = _mm512_dpbusd_epi32(vacc2x0, vb0x0, va2x0);
      vacc2x1 = _mm512_dpbusd_epi32(vacc2x1, vb1x0, va2x0);
      const __m512i va3x0 = _mm512_set1_epi32((int) unaligned_load_u32(a3));
      vacc3x0 = _mm512_dpbusd_epi32(vacc3x0, vb0x0, va3x0);
      vacc3x1 = _mm512_dpbusd_epi32(vacc3x1, vb1x0, va3x0);

      const __m512i vb0x1 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr2, 0);
      const __m512i vb1x1 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr2, 0);
      const __m512i va0x1 = _mm512_set1_epi32((int) unaligned_load_u32(a0 + 4));
      vacc1x0x0 = _mm512_dpbusd_epi32(vacc1x0x0, vb0x1, va0x1);
      vacc1x0x1 = _mm512_dpbusd_epi32(vacc1x0x1, vb1x1, va0x1);
      const __m512i va1x1 = _mm512_set1_epi32((int) unaligned_load_u32(a1 + 4));
      vacc1x1x0 = _mm512_dpbusd_epi32(vacc1x1x0, vb0x1, va1x1);
      vacc1x1x1 = _mm512_dpbusd_epi32(vacc1x1x1, vb1x1, va1x1);
      const __m512i va2x1 = _mm512_set1_epi32((int) unaligned_load_u32(a2 + 4));
      vacc1x2x0 = _mm512_dpbusd_epi32(vacc1x2x0, vb0x1, va2x1);
      vacc1x2x1 = _mm512_dpbusd_epi32(vacc1x2x1, vb1x1, va2x1);
      const __m512i va3x1 = _mm512_set1_epi32((int) unaligned_load_u32(a3 + 4));
      vacc1x3x0 = _mm512_dpbusd_epi32(vacc1x3x0, vb0x1, va3x1);
      vacc1x3x1 = _mm512_dpbusd_epi32(vacc1x3x1, vb1x1, va3x1);

      xnn_prefetch_to_l1((const int8_t*) w + 768);
      xnn_prefetch_to_l1((const int8_t*) w + 832);

      const __m512i vb0x2 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr4, 0);
      const __m512i vb1x2 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr4, 0);
      const __m512i va0x2 = _mm512_set1_epi32((int) unaligned_load_u32(a0 + 8));
      vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x2, va0x2);
      vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x2, va0x2);
      const __m512i va1x2 = _mm512_set1_epi32((int) unaligned_load_u32(a1 + 8));
      vacc1x0 = _mm512_dpbusd_epi32(vacc1x0, vb0x2, va1x2);
      vacc1x1 = _mm512_dpbusd_epi32(vacc1x1, vb1x2, va1x2);
      const __m512i va2x2 = _mm512_set1_epi32((int) unaligned_load_u32(a2 + 8));
      vacc2x0 = _mm512_dpbusd_epi32(vacc2x0, vb0x2, va2x2);
      vacc2x1 = _mm512_dpbusd_epi32(vacc2x1, vb1x2, va2x2);
      const __m512i va3x2 = _mm512_set1_epi32((int) unaligned_load_u32(a3 + 8));
      vacc3x0 = _mm512_dpbusd_epi32(vacc3x0, vb0x2, va3x2);
      vacc3x1 = _mm512_dpbusd_epi32(vacc3x1, vb1x2, va3x2);

      const __m512i vb0x3 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr6, 0);
      const __m512i vb1x3 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr6, 0);
      const __m512i va0x3 = _mm512_set1_epi32((int) unaligned_load_u32(a0 + 12));
      a0 += 16;
      vacc1x0x0 = _mm512_dpbusd_epi32(vacc1x0x0, vb0x3, va0x3);
      vacc1x0x1 = _mm512_dpbusd_epi32(vacc1x0x1, vb1x3, va0x3);
      const __m512i va1x3 = _mm512_set1_epi32((int) unaligned_load_u32(a1 + 12));
      a1 += 16;
      vacc1x1x0 = _mm512_dpbusd_epi32(vacc1x1x0, vb0x3, va1x3);
      vacc1x1x1 = _mm512_dpbusd_epi32(vacc1x1x1, vb1x3, va1x3);
      const __m512i va2x3 = _mm512_set1_epi32((int) unaligned_load_u32(a2 + 12));
      a2 += 16;
      vacc1x2x0 = _mm512_dpbusd_epi32(vacc1x2x0, vb0x3, va2x3);
      vacc1x2x1 = _mm512_dpbusd_epi32(vacc1x2x1, vb1x3, va2x3);
      const __m512i va3x3 = _mm512_set1_epi32((int) unaligned_load_u32(a3 + 12));
      a3 += 16;
      vacc1x3x0 = _mm512_dpbusd_epi32(vacc1x3x0, vb0x3, va3x3);
      vacc1x3x1 = _mm512_dpbusd_epi32(vacc1x3x1, vb1x3, va3x3);

      w = (const int8_t*) w + 128;
      k -= 16 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m512i vbb0 = _mm512_load_si512((const int8_t*) w + 0);
      const __m512i vbb1 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb0x0 = _mm512_and_si512(vbb0, vmask);
      const __m512i vb1x0 = _mm512_and_si512(vbb1, vmask);
      const __m512i va0x0 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
      a0 += 4;
      vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x0, va0x0);
      vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x0, va0x0);
      const __m512i va1x0 = _mm512_set1_epi32((int) unaligned_load_u32(a1));
      a1 += 4;
      vacc1x0 = _mm512_dpbusd_epi32(vacc1x0, vb0x0, va1x0);
      vacc1x1 = _mm512_dpbusd_epi32(vacc1x1, vb1x0, va1x0);
      const __m512i va2x0 = _mm512_set1_epi32((int) unaligned_load_u32(a2));
      a2 += 4;
      vacc2x0 = _mm512_dpbusd_epi32(vacc2x0, vb0x0, va2x0);
      vacc2x1 = _mm512_dpbusd_epi32(vacc2x1, vb1x0, va2x0);
      const __m512i va3x0 = _mm512_set1_epi32((int) unaligned_load_u32(a3));
      a3 += 4;
      vacc3x0 = _mm512_dpbusd_epi32(vacc3x0, vb0x0, va3x0);
      vacc3x1 = _mm512_dpbusd_epi32(vacc3x1, vb1x0, va3x0);

      if (k > 4 * sizeof(int8_t)) {
        const __m512i vb0x1 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr2, 0);
        const __m512i vb1x1 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr2, 0);
        const __m512i va0x1 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
        a0 += 4;
        vacc1x0x0 = _mm512_dpbusd_epi32(vacc1x0x0, vb0x1, va0x1);
        vacc1x0x1 = _mm512_dpbusd_epi32(vacc1x0x1, vb1x1, va0x1);
        const __m512i va1x1 = _mm512_set1_epi32((int) unaligned_load_u32(a1));
        a1 += 4;
        vacc1x1x0 = _mm512_dpbusd_epi32(vacc1x1x0, vb0x1, va1x1);
        vacc1x1x1 = _mm512_dpbusd_epi32(vacc1x1x1, vb1x1, va1x1);
        const __m512i va2x1 = _mm512_set1_epi32((int) unaligned_load_u32(a2));
        a2 += 4;
        vacc1x2x0 = _mm512_dpbusd_epi32(vacc1x2x0, vb0x1, va2x1);
        vacc1x2x1 = _mm512_dpbusd_epi32(vacc1x2x1, vb1x1, va2x1);
        const __m512i va3x1 = _mm512_set1_epi32((int) unaligned_load_u32(a3));
        a3 += 4;
        vacc1x3x0 = _mm512_dpbusd_epi32(vacc1x3x0, vb0x1, va3x1);
        vacc1x3x1 = _mm512_dpbusd_epi32(vacc1x3x1, vb1x1, va3x1);

        if (k > 8 * sizeof(int8_t)) {
          const __m512i vb0x2 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr4, 0);
          const __m512i vb1x2 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr4, 0);
          const __m512i va0x2 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
          a0 += 4;
          vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x2, va0x2);
          vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x2, va0x2);
          const __m512i va1x2 = _mm512_set1_epi32((int) unaligned_load_u32(a1));
          a1 += 4;
          vacc1x0 = _mm512_dpbusd_epi32(vacc1x0, vb0x2, va1x2);
          vacc1x1 = _mm512_dpbusd_epi32(vacc1x1, vb1x2, va1x2);
          const __m512i va2x2 = _mm512_set1_epi32((int) unaligned_load_u32(a2));
          a2 += 4;
          vacc2x0 = _mm512_dpbusd_epi32(vacc2x0, vb0x2, va2x2);
          vacc2x1 = _mm512_dpbusd_epi32(vacc2x1, vb1x2, va2x2);
          const __m512i va3x2 = _mm512_set1_epi32((int) unaligned_load_u32(a3));
          a3 += 4;
          vacc3x0 = _mm512_dpbusd_epi32(vacc3x0, vb0x2, va3x2);
          vacc3x1 = _mm512_dpbusd_epi32(vacc3x1, vb1x2, va3x2);
        }
      }
      w = (const int8_t*) w + 128;
    }

    vacc0x0 = _mm512_add_epi32(vacc0x0, vacc1x0x0);
    vacc0x1 = _mm512_add_epi32(vacc0x1, vacc1x0x1);
    vacc1x0 = _mm512_add_epi32(vacc1x0, vacc1x1x0);
    vacc1x1 = _mm512_add_epi32(vacc1x1, vacc1x1x1);
    vacc2x0 = _mm512_add_epi32(vacc2x0, vacc1x2x0);
    vacc2x1 = _mm512_add_epi32(vacc2x1, vacc1x2x1);
    vacc3x0 = _mm512_add_epi32(vacc3x0, vacc1x3x0);
    vacc3x1 = _mm512_add_epi32(vacc3x1, vacc1x3x1);

    __m512 vscaled0x0 = _mm512_cvtepi32_ps(vacc0x0);
    __m512 vscaled0x1 = _mm512_cvtepi32_ps(vacc0x1);
    __m512 vscaled1x0 = _mm512_cvtepi32_ps(vacc1x0);
    __m512 vscaled1x1 = _mm512_cvtepi32_ps(vacc1x1);
    __m512 vscaled2x0 = _mm512_cvtepi32_ps(vacc2x0);
    __m512 vscaled2x1 = _mm512_cvtepi32_ps(vacc2x1);
    __m512 vscaled3x0 = _mm512_cvtepi32_ps(vacc3x0);
    __m512 vscaled3x1 = _mm512_cvtepi32_ps(vacc3x1);

    const __m512 vtwo = _mm512_set1_ps(2.0f);
    const __m512 rh_zero_points_0 = _mm512_load_ps(kzp + 0);
    const __m512 biased_rh_zero_points_0 = _mm512_add_ps(rh_zero_points_0, vtwo);
    const __m512 rh_zero_points_1 = _mm512_load_ps(kzp + 16);
    const __m512 biased_rh_zero_points_1 = _mm512_add_ps(rh_zero_points_1, vtwo);

    const __m512 vlh_row_sum_0 = _mm512_set1_ps(row_sum[0]);
    vscaled0x0 = _mm512_fnmadd_ps(biased_rh_zero_points_0, vlh_row_sum_0, vscaled0x0);
    vscaled0x1 = _mm512_fnmadd_ps(biased_rh_zero_points_1, vlh_row_sum_0, vscaled0x1);
    const __m512 vlh_row_sum_1 = _mm512_set1_ps(row_sum[1]);
    vscaled1x0 = _mm512_fnmadd_ps(biased_rh_zero_points_0, vlh_row_sum_1, vscaled1x0);
    vscaled1x1 = _mm512_fnmadd_ps(biased_rh_zero_points_1, vlh_row_sum_1, vscaled1x1);
    const __m512 vlh_row_sum_2 = _mm512_set1_ps(row_sum[2]);
    vscaled2x0 = _mm512_fnmadd_ps(biased_rh_zero_points_0, vlh_row_sum_2, vscaled2x0);
    vscaled2x1 = _mm512_fnmadd_ps(biased_rh_zero_points_1, vlh_row_sum_2, vscaled2x1);
    const __m512 vlh_row_sum_3 = _mm512_set1_ps(row_sum[3]);
    vscaled3x0 = _mm512_fnmadd_ps(biased_rh_zero_points_0, vlh_row_sum_3, vscaled3x0);
    vscaled3x1 = _mm512_fnmadd_ps(biased_rh_zero_points_1, vlh_row_sum_3, vscaled3x1);

    const __m512 vscaled_lh_zero_point_0 = _mm512_set1_ps((float) kc * quantization_params[0].zero_point);
    vscaled0x0 = _mm512_fmadd_ps(rh_zero_points_0, vscaled_lh_zero_point_0, vscaled0x0);
    vscaled0x1 = _mm512_fmadd_ps(rh_zero_points_1, vscaled_lh_zero_point_0, vscaled0x1);
    const __m512 vscaled_lh_zero_point_1 = _mm512_set1_ps((float) kc * quantization_params[1].zero_point);
    vscaled1x0 = _mm512_fmadd_ps(rh_zero_points_0, vscaled_lh_zero_point_1, vscaled1x0);
    vscaled1x1 = _mm512_fmadd_ps(rh_zero_points_1, vscaled_lh_zero_point_1, vscaled1x1);
    const __m512 vscaled_lh_zero_point_2 = _mm512_set1_ps((float) kc * quantization_params[2].zero_point);
    vscaled2x0 = _mm512_fmadd_ps(rh_zero_points_0, vscaled_lh_zero_point_2, vscaled2x0);
    vscaled2x1 = _mm512_fmadd_ps(rh_zero_points_1, vscaled_lh_zero_point_2, vscaled2x1);
    const __m512 vscaled_lh_zero_point_3 = _mm512_set1_ps((float) kc * quantization_params[3].zero_point);
    vscaled3x0 = _mm512_fmadd_ps(rh_zero_points_0, vscaled_lh_zero_point_3, vscaled3x0);
    vscaled3x1 = _mm512_fmadd_ps(rh_zero_points_1, vscaled_lh_zero_point_3, vscaled3x1);

    const __m512 vinv_scale_0 = _mm512_set1_ps(quantization_params[0].inv_scale);
    vscaled0x0 = _mm512_mul_ps(vscaled0x0, vinv_scale_0);
    vscaled0x1 = _mm512_mul_ps(vscaled0x1, vinv_scale_0);
    const __m512 vinv_scale_1 = _mm512_set1_ps(quantization_params[1].inv_scale);
    vscaled1x0 = _mm512_mul_ps(vscaled1x0, vinv_scale_1);
    vscaled1x1 = _mm512_mul_ps(vscaled1x1, vinv_scale_1);
    const __m512 vinv_scale_2 = _mm512_set1_ps(quantization_params[2].inv_scale);
    vscaled2x0 = _mm512_mul_ps(vscaled2x0, vinv_scale_2);
    vscaled2x1 = _mm512_mul_ps(vscaled2x1, vinv_scale_2);
    const __m512 vinv_scale_3 = _mm512_set1_ps(quantization_params[3].inv_scale);
    vscaled3x0 = _mm512_mul_ps(vscaled3x0, vinv_scale_3);
    vscaled3x1 = _mm512_mul_ps(vscaled3x1, vinv_scale_3);

    const __m512 vfilter_output_scale0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;
    const __m512 vbias0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0 = _mm512_fmadd_ps(vscaled0x0, vfilter_output_scale0, vbias0);
    vscaled0x1 = _mm512_fmadd_ps(vscaled0x1, vfilter_output_scale1, vbias1);
    vscaled1x0 = _mm512_fmadd_ps(vscaled1x0, vfilter_output_scale0, vbias0);
    vscaled1x1 = _mm512_fmadd_ps(vscaled1x1, vfilter_output_scale1, vbias1);
    vscaled2x0 = _mm512_fmadd_ps(vscaled2x0, vfilter_output_scale0, vbias0);
    vscaled2x1 = _mm512_fmadd_ps(vscaled2x1, vfilter_output_scale1, vbias1);
    vscaled3x0 = _mm512_fmadd_ps(vscaled3x0, vfilter_output_scale0, vbias0);
    vscaled3x1 = _mm512_fmadd_ps(vscaled3x1, vfilter_output_scale1, vbias1);

    vscaled0x0 = _mm512_max_ps(vscaled0x0, voutput_min);
    vscaled0x1 = _mm512_max_ps(vscaled0x1, voutput_min);
    vscaled1x0 = _mm512_max_ps(vscaled1x0, voutput_min);
    vscaled1x1 = _mm512_max_ps(vscaled1x1, voutput_min);
    vscaled2x0 = _mm512_max_ps(vscaled2x0, voutput_min);
    vscaled2x1 = _mm512_max_ps(vscaled2x1, voutput_min);
    vscaled3x0 = _mm512_max_ps(vscaled3x0, voutput_min);
    vscaled3x1 = _mm512_max_ps(vscaled3x1, voutput_min);

    vscaled0x0 = _mm512_min_ps(vscaled0x0, voutput_max);
    vscaled0x1 = _mm512_min_ps(vscaled0x1, voutput_max);
    vscaled1x0 = _mm512_min_ps(vscaled1x0, voutput_max);
    vscaled1x1 = _mm512_min_ps(vscaled1x1, voutput_max);
    vscaled2x0 = _mm512_min_ps(vscaled2x0, voutput_max);
    vscaled2x1 = _mm512_min_ps(vscaled2x1, voutput_max);
    vscaled3x0 = _mm512_min_ps(vscaled3x0, voutput_max);
    vscaled3x1 = _mm512_min_ps(vscaled3x1, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c0 + 0, vscaled0x0);
      _mm512_storeu_ps(c0 + 16, vscaled0x1);
      _mm512_storeu_ps(c1 + 0, vscaled1x0);
      _mm512_storeu_ps(c1 + 16, vscaled1x1);
      _mm512_storeu_ps(c2 + 0, vscaled2x0);
      _mm512_storeu_ps(c2 + 16, vscaled2x1);
      _mm512_storeu_ps(c3 + 0, vscaled3x0);
      _mm512_storeu_ps(c3 + 16, vscaled3x1);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 32;
    } else {
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0x0);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vscaled1x0);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vscaled2x0);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vscaled3x0);
      if (nc > 16) {
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled0x1);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vscaled1x1);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vscaled2x1);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vscaled3x1);
      }
      nc = 0;
    }
  } while (nc != 0);
}
