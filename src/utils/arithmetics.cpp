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

#include "arithmetics.h"

namespace cnindex {

static float * fvec_add_c(const float *x, const float *y, size_t d, float *r);
static float * fvec_add_sse(const float *x, const float *y, size_t d, float *r);
static float * fvec_add_avx(const float *x, const float *y, size_t d, float *r);
static float * fvec_add_avx512(const float *x, const float *y, size_t d, float *r);
static float * fvec_add_asimd(const float *x, const float *y, size_t d, float *r);

static float * fvec_sub_c(const float *x, const float *y, size_t d, float *r);
static float * fvec_sub_sse(const float *x, const float *y, size_t d, float *r);
static float * fvec_sub_avx(const float *x, const float *y, size_t d, float *r);
static float * fvec_sub_avx512(const float *x, const float *y, size_t d, float *r);
static float * fvec_sub_asimd(const float *x, const float *y, size_t d, float *r);

#if (defined(__AVX__) && defined(__SSE__))
fvec_arith_func_ptr fvec_add = fvec_add_avx;
#elif defined(__SSE__)
fvec_arith_func_ptr fvec_add = fvec_add_sse;
#elif defined(__aarch64__)
fvec_arith_func_ptr fvec_add = fvec_add_asimd;
#else
fvec_arith_func_ptr fvec_add = fvec_add_c;
#endif

#if (defined(__AVX__) && defined(__SSE__))
fvec_arith_func_ptr fvec_sub = fvec_sub_avx;
#elif defined(__SSE__)
fvec_arith_func_ptr fvec_sub = fvec_sub_sse;
#elif defined(__aarch64__)
fvec_arith_func_ptr fvec_sub = fvec_sub_asimd;
#else
fvec_arith_func_ptr fvec_sub = fvec_sub_c;
#endif


float * fvec_add_c(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  for (size_t i = 0; i < d; i++) {
    r[i] = x[i] + y[i];
  }

  return r;
}

#if defined(__SSE__)
float * fvec_add_sse(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  while (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 mres = mx + my;
    _mm_storeu_ps(r, mres); r += 4;
    d -= 4;
  }

  if (d > 0) {
    for (size_t i = 0; i < d; i++) {
      r[i] = x[i] + y[i];
    }
  }

  return r;
}
#endif

#if (defined(__AVX__) && defined(__SSE__))
float * fvec_add_avx(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  while (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x); x += 8;
    __m256 my = _mm256_loadu_ps(y); y += 8;
    const __m256 mres = mx + my;
    _mm256_storeu_ps(r, mres); r += 8;
    d -= 8;
  }

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 mres = mx + my;
    _mm_storeu_ps(r, mres); r += 4;
    d -= 4;
  }

  if (d > 0) {
    for (size_t i = 0; i < d; i++) {
      r[i] = x[i] + y[i];
    }
  }

  return r;
}
#endif

/*
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX__) && defined(__SSE__))
float * fvec_add_avx512(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  while (d >= 16) {
    __m512 mx = _mm512_loadu_ps(x); x += 16;
    __m512 my = _mm512_loadu_ps(y); y += 16;
    const __m512 mres = mx + my;
    _mm512_storeu_ps(r, mres); r += 16;
    d -= 16;
  }

  if (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x); x += 8;
    __m256 my = _mm256_loadu_ps(y); y += 8;
    const __m256 mres = mx + my;
    _mm256_storeu_ps(r, mres); r += 8;
    d -= 8;
  }

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 mres = mx + my;
    _mm_storeu_ps(r, mres); r += 4;
    d -= 4;
  }

  if (d > 0) {
    for (size_t i = 0; i < d; i++) {
      r[i] = x[i] + y[i];
    }
  }

  return r;
}
#endif
*/

#if defined(__aarch64__)
float * fvec_add_asimd(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  const size_t d_simd = d - (d & 3);
  size_t i;
  for (i = 0; i < d_simd; i += 4) {
    float32x4_t xi = vld1q_f32(x + i);
    float32x4_t yi = vld1q_f32(y + i);
    float32x4_t res = vaddq_f32(xi, yi);
    vst1q_f32(r + i, res);
  }
  for (; i < d; ++i) {
    r[i] = x[i] + y[i];
  }

  return r;
}
#endif

float * fvec_sub_c(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  for (size_t i = 0; i < d; i++) {
    r[i] = x[i] - y[i];
  }

  return r;
}

#if defined(__SSE__)
float * fvec_sub_sse(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  while (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 mres = mx - my;
    _mm_storeu_ps(r, mres); r += 4;
    d -= 4;
  }

  if (d > 0) {
    for (size_t i = 0; i < d; i++) {
      r[i] = x[i] - y[i];
    }
  }

  return r;
}
#endif

#if (defined(__AVX__) && defined(__SSE__))
float * fvec_sub_avx(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  while (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x); x += 8;
    __m256 my = _mm256_loadu_ps(y); y += 8;
    const __m256 mres = mx - my;
    _mm256_storeu_ps(r, mres); r += 8;
    d -= 8;
  }

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 mres = mx - my;
    _mm_storeu_ps(r, mres); r += 4;
    d -= 4;
  }

  if (d > 0) {
    for (size_t i = 0; i < d; i++) {
      r[i] = x[i] - y[i];
    }
  }

  return r;
}
#endif

/*
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX__) && defined(__SSE__))
float * fvec_sub_avx512(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  while (d >= 16) {
    __m512 mx = _mm512_loadu_ps(x); x += 16;
    __m512 my = _mm512_loadu_ps(y); y += 16;
    const __m512 mres = mx - my;
    _mm512_storeu_ps(r, mres); r += 16;
    d -= 16;
  }

  if (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x); x += 8;
    __m256 my = _mm256_loadu_ps(y); y += 8;
    const __m256 mres = mx - my;
    _mm256_storeu_ps(r, mres); r += 8;
    d -= 8;
  }

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps(x); x += 4;
    __m128 my = _mm_loadu_ps(y); y += 4;
    const __m128 mres = mx - my;
    _mm_storeu_ps(r, mres); r += 4;
    d -= 4;
  }

  if (d > 0) {
    for (size_t i = 0; i < d; i++) {
      r[i] = x[i] - y[i];
    }
  }

  return r;
}
#endif
*/

#if defined(__aarch64__)
float * fvec_sub_asimd(const float *x, const float *y, size_t d, float *r) {
  if (!r) r = const_cast<float *>(x);

  const size_t d_simd = d - (d & 3);
  size_t i;
  for (i = 0; i < d_simd; i += 4) {
    float32x4_t xi = vld1q_f32(x + i);
    float32x4_t yi = vld1q_f32(y + i);
    float32x4_t res = vsubq_f32(xi, yi);
    vst1q_f32(r + i, res);
  }
  for (; i < d; ++i) {
    r[i] = x[i] - y[i];
  }

  return r;
}
#endif

} // cnindex
