/*************************************************************************
* Copyright (C) 2021 by Cambricon, Inc. All rights reserved
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*************************************************************************/

#ifdef __SSE__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "distances.h"

namespace cnindex {

static float fvec_L2sqr_c(const float *x, const float *y, size_t d);
static float fvec_L2sqr_sse(const float *x, const float *y, size_t d);
static float fvec_L2sqr_avx(const float *x, const float *y, size_t d);
static float fvec_L2sqr_avx512(const float *x, const float *y, size_t d);
static float fvec_L2sqr_asimd(const float *x, const float *y, size_t d);

static float fvec_inner_product_c(const float *x, const float *y, size_t d);
static float fvec_inner_product_sse(const float *x, const float *y, size_t d);
static float fvec_inner_product_avx(const float *x, const float *y, size_t d);
static float fvec_inner_product_avx512(const float *x, const float *y, size_t d);
static float fvec_inner_product_asimd(const float *x, const float *y, size_t d);

#if (defined(__AVX__) && defined(__SSE__))
fvec_func_ptr fvec_L2sqr = fvec_L2sqr_avx;
#elif defined(__SSE__)
fvec_func_ptr fvec_L2sqr = fvec_L2sqr_sse;
#elif defined(__aarch64__)
fvec_func_ptr fvec_L2sqr = fvec_L2sqr_asimd;
#else
fvec_func_ptr fvec_L2sqr = fvec_L2sqr_c;
#endif

#if (defined(__AVX__) && defined(__SSE__))
fvec_func_ptr fvec_inner_product = fvec_inner_product_avx;
#elif defined(__SSE__)
fvec_func_ptr fvec_inner_product = fvec_inner_product_sse;
#elif defined(__aarch64__)
fvec_func_ptr fvec_inner_product = fvec_inner_product_asimd;
#else
fvec_func_ptr fvec_inner_product = fvec_inner_product_c;
#endif

static float fvec_L2sqr_c(const float *x, const float *y, size_t d) {
  float dif = 0.0;
  float sum = 0.0;
  for (size_t i = 0; i < d; i++) {
    dif = x[i] - y[i];
    sum += dif * dif;
  }
  return sum;
}

#if defined(__SSE__)
// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x) {
  __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
  switch (d) {
    case 3 : buf[2] = x[2];
    case 2 : buf[1] = x[1];
    case 1 : buf[0] = x[0];
  }
  return _mm_load_ps(buf);
  // cannot use AVX2 _mm_mask_set1_epi32
}

static float fvec_L2sqr_sse(const float *x, const float *y, size_t d) {
  __m128 msum1 = _mm_setzero_ps();

  while (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 a_m_b1 = mx - my;
    msum1 += a_m_b1 * a_m_b1;
    d -= 4;
  }

  if (d > 0) {
    // add the last 1, 2 or 3 values
    __m128 mx = masked_read(d, x);
    __m128 my = masked_read(d, y);
    __m128 a_m_b1 = mx - my;
    msum1 += a_m_b1 * a_m_b1;
  }

  msum1 = _mm_hadd_ps(msum1, msum1);
  msum1 = _mm_hadd_ps(msum1, msum1);
  return  _mm_cvtss_f32(msum1);
}
#endif

#if (defined(__AVX__) && defined(__SSE__))
static float fvec_L2sqr_avx(const float *x, const float *y, size_t d) {
  __m256 msum1 = _mm256_setzero_ps();

  while (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x); x += 8;
    __m256 my = _mm256_loadu_ps(y); y += 8;
    const __m256 a_m_b1 = mx - my;
    msum1 += a_m_b1 * a_m_b1;
    d -= 8;
  }

  __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
  msum2 +=       _mm256_extractf128_ps(msum1, 0);

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 a_m_b1 = mx - my;
    msum2 += a_m_b1 * a_m_b1;
    d -= 4;
  }

  if (d > 0) {
    __m128 mx = masked_read(d, x);
    __m128 my = masked_read(d, y);
    __m128 a_m_b1 = mx - my;
    msum2 += a_m_b1 * a_m_b1;
  }

  msum2 = _mm_hadd_ps(msum2, msum2);
  msum2 = _mm_hadd_ps(msum2, msum2);
  return  _mm_cvtss_f32(msum2);
}
#endif

/*
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX__) && defined(__SSE__))
static float fvec_L2sqr_avx512(const float *x, const float *y, size_t d) {
  __m512 msum0 = _mm512_setzero_ps();

  while (d >= 16) {
    __m512 mx = _mm512_loadu_ps(x); x += 16;
    __m512 my = _mm512_loadu_ps(y); y += 16;
    const __m512 a_m_b1 = mx - my;
    msum0 += a_m_b1 * a_m_b1;
    d -= 16;
  }

  __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
  msum1 +=       _mm512_extractf32x8_ps(msum0, 0);

  if (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x); x += 8;
    __m256 my = _mm256_loadu_ps(y); y += 8;
    const __m256 a_m_b1 = mx - my;
    msum1 += a_m_b1 * a_m_b1;
    d -= 8;
  }

  __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
  msum2 +=       _mm256_extractf128_ps(msum1, 0);

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 a_m_b1 = mx - my;
    msum2 += a_m_b1 * a_m_b1;
    d -= 4;
  }

  if (d > 0) {
    __m128 mx = masked_read(d, x);
    __m128 my = masked_read(d, y);
    __m128 a_m_b1 = mx - my;
    msum2 += a_m_b1 * a_m_b1;
  }

  msum2 = _mm_hadd_ps(msum2, msum2);
  msum2 = _mm_hadd_ps(msum2, msum2);
  return  _mm_cvtss_f32(msum2);
}
#endif
*/

#if defined(__aarch64__)
static float fvec_L2sqr_asimd(const float* x, const float* y, size_t d) {
  float32x4_t accux4 = vdupq_n_f32(0);
  const size_t d_simd = d - (d & 3);
  size_t i;
  for (i = 0; i < d_simd; i += 4) {
    float32x4_t xi = vld1q_f32(x + i);
    float32x4_t yi = vld1q_f32(y + i);
    float32x4_t sq = vsubq_f32(xi, yi);
    accux4 = vfmaq_f32(accux4, sq, sq);
  }
  float32x4_t accux2 = vpaddq_f32(accux4, accux4);
  float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
  for (; i < d; ++i) {
    float32_t sq = x[i] - y[i];
    accux1 += sq * sq;
  }
  return accux1;
}
#endif

static float fvec_inner_product_c(const float *x, const float *y, size_t d) {
  size_t i;
  float res = 0;
  for (i = 0; i < d; i++)
    res += x[i] * y[i];
  return res;
}

#if defined(__SSE__)
static float fvec_inner_product_sse(const float *x, const float *y, size_t d) {
  __m128 mx, my;
  __m128 msum1 = _mm_setzero_ps();

  while (d >= 4) {
    mx = _mm_loadu_ps(x); x += 4;
    my = _mm_loadu_ps(y); y += 4;
    msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
    d -= 4;
  }

  if (d > 0) {
    // add the last 1, 2, or 3 values
    mx = masked_read (d, x);
    my = masked_read (d, y);
    msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
  }

  msum1 = _mm_hadd_ps (msum1, msum1);
  msum1 = _mm_hadd_ps (msum1, msum1);
  return  _mm_cvtss_f32 (msum1);
}
#endif

#if (defined(__AVX__) && defined(__SSE__))
static float fvec_inner_product_avx(const float *x, const float *y, size_t d) {
  __m256 msum1 = _mm256_setzero_ps();

  while (d >= 8) {
    __m256 mx = _mm256_loadu_ps (x); x += 8;
    __m256 my = _mm256_loadu_ps (y); y += 8;
    msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (mx, my));
    d -= 8;
  }

  __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
  msum2 +=       _mm256_extractf128_ps(msum1, 0);

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps (x); x += 4;
    __m128 my = _mm_loadu_ps (y); y += 4;
    msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
    d -= 4;
  }

  if (d > 0) {
    __m128 mx = masked_read (d, x);
    __m128 my = masked_read (d, y);
    msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
  }

  msum2 = _mm_hadd_ps (msum2, msum2);
  msum2 = _mm_hadd_ps (msum2, msum2);
  return  _mm_cvtss_f32 (msum2);
}
#endif

/*
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX__) && defined(__SSE__))
static float fvec_inner_product_avx512(const float *x, const float *y, size_t d) {
  __m512 msum0 = _mm512_setzero_ps();

  while (d >= 16) {
    __m512 mx = _mm512_loadu_ps (x); x += 16;
    __m512 my = _mm512_loadu_ps (y); y += 16;
    msum0 = _mm512_add_ps (msum0, _mm512_mul_ps (mx, my));
    d -= 16;
  }

  __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
  msum1 +=       _mm512_extractf32x8_ps(msum0, 0);

  if (d >= 8) {
    __m256 mx = _mm256_loadu_ps (x); x += 8;
    __m256 my = _mm256_loadu_ps (y); y += 8;
    msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (mx, my));
    d -= 8;
  }

  __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
  msum2 +=       _mm256_extractf128_ps(msum1, 0);

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps (x); x += 4;
    __m128 my = _mm_loadu_ps (y); y += 4;
    msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
    d -= 4;
  }

  if (d > 0) {
    __m128 mx = masked_read (d, x);
    __m128 my = masked_read (d, y);
    msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
  }

  msum2 = _mm_hadd_ps (msum2, msum2);
  msum2 = _mm_hadd_ps (msum2, msum2);
  return  _mm_cvtss_f32 (msum2);
}
#endif
*/

#if defined(__aarch64__)
static float fvec_inner_product_asimd(const float *x, const float *y, size_t d) {
  float32x4_t accux4 = vdupq_n_f32 (0);
  const size_t d_simd = d - (d & 3);
  size_t i;
  for (i = 0; i < d_simd; i += 4) {
    float32x4_t xi = vld1q_f32 (x + i);
    float32x4_t yi = vld1q_f32 (y + i);
    accux4 = vfmaq_f32 (accux4, xi, yi);
  }
  float32x4_t accux2 = vpaddq_f32 (accux4, accux4);
  float32_t accux1 = vdups_laneq_f32 (accux2, 0) + vdups_laneq_f32 (accux2, 1);
  for (; i < d; ++i) {
    accux1 += x[i] * y[i];
  }
  return accux1;
}
#endif

} // cnindex
